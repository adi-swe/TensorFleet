package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	orchestratorpb "github.com/tensorfleet/api-gateway/proto/orchestrator"
)

// Helper function to get minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Helper function to get maximum of two int64 values (ensures non-negative uptime)
func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// JobRecord is the canonical representation of a job stored in Redis.
// Its JSON tags match exactly what handleSubmitJob writes, so the same
// struct is used for both storing and reading job metadata.
type JobRecord struct {
	JobID          string `json:"job_id"`
	UserID         string `json:"user_id"`
	ModelType      string `json:"model_type"`
	DatasetPath    string `json:"dataset_path"`
	Status         string `json:"status"`
	TotalTasks     int32  `json:"total_tasks"`
	CompletedTasks int32  `json:"completed_tasks"`
	CreatedAt      int64  `json:"created_at"` // Unix seconds
}

type GatewayServer struct {
	orchestratorClient orchestratorpb.OrchestratorServiceClient
	grpcConn           *grpc.ClientConn
	redisClient        *redis.Client
	router             *gin.Engine
}

func NewGatewayServer() (*GatewayServer, error) {
	// Connect to orchestrator via gRPC
	orchestratorAddr := os.Getenv("ORCHESTRATOR_ADDR")
	if orchestratorAddr == "" {
		orchestratorAddr = "orchestrator:50051"
	}

	log.Printf("Connecting to orchestrator at %s", orchestratorAddr)
	conn, err := grpc.NewClient(orchestratorAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	)
	if err != nil {
		return nil, err
	}

	client := orchestratorpb.NewOrchestratorServiceClient(conn)

	// Connect to Redis
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "redis:6379"
	}

	log.Printf("Connecting to Redis at %s", redisAddr)
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
		DB:   0,
	})

	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Printf("Warning: Redis connection failed: %v", err)
	} else {
		log.Println("Connected to Redis successfully")
	}

	router := gin.Default()

	// Allowed CORS origins, configurable via CORS_ALLOWED_ORIGINS
	// (comma-separated). Defaults to "*" to preserve dev behavior.
	allowedOrigins := []string{"*"}
	if envOrigins := os.Getenv("CORS_ALLOWED_ORIGINS"); envOrigins != "" {
		allowedOrigins = allowedOrigins[:0]
		for _, o := range strings.Split(envOrigins, ",") {
			if o = strings.TrimSpace(o); o != "" {
				allowedOrigins = append(allowedOrigins, o)
			}
		}
	}
	allowAllOrigins := len(allowedOrigins) == 1 && allowedOrigins[0] == "*"

	// Add CORS middleware
	router.Use(func(c *gin.Context) {
		if allowAllOrigins {
			c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		} else {
			origin := c.GetHeader("Origin")
			for _, allowed := range allowedOrigins {
				if origin == allowed {
					c.Writer.Header().Set("Access-Control-Allow-Origin", origin)
					c.Writer.Header().Set("Vary", "Origin")
					break
				}
			}
		}
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-User-ID, x-user-id")
		c.Writer.Header().Set("Access-Control-Expose-Headers", "Content-Type")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	gs := &GatewayServer{
		orchestratorClient: client,
		grpcConn:           conn,
		redisClient:        rdb,
		router:             router,
	}

	gs.setupRoutes()
	return gs, nil
}

func (gs *GatewayServer) setupRoutes() {
	// Health check
	gs.router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "healthy"})
	})

	// Worker activity endpoint (public, no auth required for demo)
	gs.router.GET("/worker-activity", gs.handleWorkerActivity)

	// API routes
	api := gs.router.Group("/api/v1")
	{
		api.POST("/jobs", gs.handleSubmitJob)
		api.GET("/jobs/:id", gs.handleGetJobStatus)
		api.GET("/jobs/:id/logs", gs.handleGetJobLogs)
		api.GET("/jobs", gs.handleListJobs)
		api.DELETE("/jobs/:id", gs.handleCancelJob)
		api.GET("/workers", gs.handleGetWorkers)
	}
}

type JobSubmitRequest struct {
	ModelType       string            `json:"model_type" binding:"required"`
	DatasetPath     string            `json:"dataset_path" binding:"required"`
	Hyperparameters map[string]string `json:"hyperparameters"`
	NumWorkers      int32             `json:"num_workers"`
	Epochs          int32             `json:"epochs"`
}

func (gs *GatewayServer) handleSubmitJob(c *gin.Context) {
	var req JobSubmitRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Generate job ID
	jobID := uuid.New().String()
	userID := c.GetHeader("X-User-ID")
	if userID == "" {
		userID = "anonymous"
	}

	// Set defaults
	if req.NumWorkers == 0 {
		req.NumWorkers = 2
	}
	if req.Epochs == 0 {
		req.Epochs = 10
		if strings.HasPrefix(req.ModelType, "fleet_") {
			req.Epochs = 1
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Forward to orchestrator
	resp, err := gs.orchestratorClient.CreateTrainingJob(ctx, &orchestratorpb.TrainingJobRequest{
		JobId:           jobID,
		UserId:          userID,
		ModelType:       req.ModelType,
		DatasetPath:     req.DatasetPath,
		Hyperparameters:  req.Hyperparameters,
		NumWorkers:      req.NumWorkers,
		Epochs:          req.Epochs,
	})

	if err != nil {
		log.Printf("Error creating job: %v", err)
		if status.Code(err) == codes.InvalidArgument {
			c.JSON(http.StatusBadRequest, gin.H{"error": status.Convert(err).Message()})
			return
		}
		if status.Code(err) == codes.ResourceExhausted {
			c.Header("Retry-After", "1")
			c.JSON(http.StatusTooManyRequests, gin.H{"error": status.Convert(err).Message()})
			return
		}
		if status.Code(err) == codes.Unavailable {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": status.Convert(err).Message()})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create job"})
		return
	}

	// Store job metadata in Redis for history
	jobMetadata := JobRecord{
		JobID:          jobID,
		UserID:         userID,
		ModelType:      req.ModelType,
		DatasetPath:    req.DatasetPath,
		Status:         resp.Status,
		TotalTasks:     resp.NumTasks,
		CompletedTasks: 0,
		CreatedAt:      time.Now().Unix(),
	}

	jobJSON, err := json.Marshal(jobMetadata)
	if err == nil {
		// Store job in Redis with 7 day expiration
		gs.redisClient.Set(ctx, fmt.Sprintf("job:%s", jobID), jobJSON, 7*24*time.Hour)
		log.Printf("Stored job %s metadata in Redis", jobID)
	} else {
		log.Printf("Warning: Failed to store job metadata in Redis: %v", err)
	}

	c.JSON(http.StatusAccepted, gin.H{
		"job_id":    resp.JobId,
		"status":    resp.Status,
		"num_tasks": resp.NumTasks,
		"message":   resp.Message,
	})
}

func (gs *GatewayServer) handleGetJobStatus(c *gin.Context) {
	jobID := c.Param("id")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := gs.orchestratorClient.GetJobStatus(ctx, &orchestratorpb.GetJobStatusRequest{
		JobId: jobID,
	})

	if err != nil {
		log.Printf("Error getting job status: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get job status"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"job_id":          resp.JobId,
		"status":          resp.Status,
		"progress":        resp.Progress,
		"completed_tasks": resp.CompletedTasks,
		"total_tasks":     resp.TotalTasks,
		"current_loss":    resp.CurrentLoss,
		"current_accuracy": resp.CurrentAccuracy,
		"message":         resp.Message,
	})
}

func (gs *GatewayServer) handleGetJobLogs(c *gin.Context) {
	jobID := c.Param("id")

	// Verify job exists first (before setting SSE headers)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	resp, err := gs.orchestratorClient.GetJobStatus(ctx, &orchestratorpb.GetJobStatusRequest{
		JobId: jobID,
	})
	cancel()

	if err != nil {
		log.Printf("Error getting job status for logs: %v", err)
		c.JSON(http.StatusNotFound, gin.H{"error": "Job not found"})
		return
	}

	// Set headers for Server-Sent Events AFTER verification
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("X-Accel-Buffering", "no") // Disable nginx buffering

	// Function to send a log message
	sendLog := func(level, message string) {
		logEntry := fmt.Sprintf("[%s] %s: %s", time.Now().Format("15:04:05"), level, message)
		c.SSEvent("message", logEntry)
		c.Writer.Flush()
	}

	// Send initial logs
	sendLog("INFO", fmt.Sprintf("Job %s created", jobID))
	time.Sleep(100 * time.Millisecond)
	
	sendLog("INFO", fmt.Sprintf("Initializing training for model: %s", resp.JobId))
	time.Sleep(100 * time.Millisecond)
	
	sendLog("INFO", fmt.Sprintf("Distributing %d tasks across workers", resp.TotalTasks))
	time.Sleep(100 * time.Millisecond)

	// Stream logs with updates every 2 seconds
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	clientGone := c.Request.Context().Done()
	
	// Stream continuously until job completes, fails, is cancelled, or client disconnects
	for {
		select {
		case <-clientGone:
			log.Printf("Client disconnected from log stream for job %s", jobID)
			return
		case <-ticker.C:
			// Get updated status
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			resp, err := gs.orchestratorClient.GetJobStatus(ctx, &orchestratorpb.GetJobStatusRequest{
				JobId: jobID,
			})
			cancel()

			if err != nil {
				sendLog("ERROR", fmt.Sprintf("Failed to get job status: %v", err))
				return
			}

			// Send progress updates
			if resp.CompletedTasks > 0 {
				sendLog("INFO", fmt.Sprintf("Training in progress: %d/%d tasks completed (%.1f%%)", 
					resp.CompletedTasks, resp.TotalTasks, float64(resp.Progress)))
				
				if resp.CurrentLoss > 0 || resp.CurrentAccuracy > 0 {
					sendLog("INFO", fmt.Sprintf("Current metrics - Loss: %.4f, Accuracy: %.4f", 
						resp.CurrentLoss, resp.CurrentAccuracy))
				}
			}

			// Check if job is completed, failed, or cancelled
			if resp.Status == "COMPLETED" {
				sendLog("INFO", fmt.Sprintf("Job %s completed successfully!", jobID))
				sendLog("INFO", fmt.Sprintf("Final metrics - Loss: %.4f, Accuracy: %.4f", 
					resp.CurrentLoss, resp.CurrentAccuracy))
				sendLog("INFO", "Log streaming ended")
				return
			} else if resp.Status == "FAILED" {
				sendLog("ERROR", fmt.Sprintf("Job %s failed", jobID))
				sendLog("INFO", "Log streaming ended")
				return
			} else if resp.Status == "CANCELLED" {
				sendLog("WARN", fmt.Sprintf("Job %s was cancelled", jobID))
				sendLog("INFO", "Log streaming ended")
				return
			} else if resp.Status == "RUNNING" {
				// Get worker activity for detailed logging
				workerCtx, workerCancel := context.WithTimeout(context.Background(), 5*time.Second)
				workerResp, err := gs.orchestratorClient.GetWorkerActivity(workerCtx, &orchestratorpb.WorkerActivityRequest{})
				workerCancel()
				activeWorkers := 0
				var workerDetails []string
				
				if err == nil && workerResp != nil {
					// Count workers for this job (BUSY = actively processing, IDLE = available but assigned to job)
					for _, worker := range workerResp.Workers {
						if worker.CurrentJobId == jobID {
							// Count as active if worker is BUSY or recently worked on this job (within last 10s)
							timeSinceActivity := time.Since(time.Unix(worker.LastActivityTime, 0))
							if worker.Status == "BUSY" || (worker.Status == "IDLE" && timeSinceActivity < 10*time.Second) {
								activeWorkers++
								statusIcon := "⚡"
								if worker.Status == "IDLE" {
									statusIcon = "⏸️"
								}
								
								taskDisplay := "completed"
								if worker.CurrentTaskId != "" && len(worker.CurrentTaskId) >= 8 {
									taskDisplay = worker.CurrentTaskId[:8]
								}
								
								workerDetails = append(workerDetails, fmt.Sprintf(
									"  %s Worker %s: %s (task: %s, completed: %d)", 
									statusIcon,
									worker.WorkerId[:8],
									worker.Status,
									taskDisplay,
									worker.TasksCompleted))
							}
						}
					}
					
					// If no workers found but job is running, show debug info
					if activeWorkers == 0 && len(workerResp.Workers) > 0 {
						sendLog("DEBUG", fmt.Sprintf("Total workers in system: %d", len(workerResp.Workers)))
						for _, w := range workerResp.Workers {
							sendLog("DEBUG", fmt.Sprintf("  Worker %s: status=%s, job=%s", 
								w.WorkerId[:min(8, len(w.WorkerId))], w.Status, w.CurrentJobId[:min(8, len(w.CurrentJobId))]))
						}
					}
				} else if err != nil {
					sendLog("WARN", fmt.Sprintf("Failed to get worker activity: %v", err))
				}
				
				// Log progress summary
				sendLog("INFO", fmt.Sprintf("Job running - Progress: %.1f%%, Tasks: %d/%d, Active workers: %d", 
					float64(resp.Progress), resp.CompletedTasks, resp.TotalTasks, activeWorkers))
				
				// Log individual worker details
				if len(workerDetails) > 0 {
					for _, detail := range workerDetails {
						sendLog("INFO", detail)
					}
				}
			}
		}
	}
}

func (gs *GatewayServer) handleListJobs(c *gin.Context) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Pagination params: limit (default 50, max 200) and offset (default 0)
	limit := 50
	if v := c.Query("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			limit = n
		}
	}
	if limit > 200 {
		limit = 200
	}
	offset := 0
	if v := c.Query("offset"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			offset = n
		}
	}

	// Get all job keys from Redis using SCAN (non-blocking, unlike KEYS)
	var keys []string
	var cursor uint64
	for {
		batch, nextCursor, err := gs.redisClient.Scan(ctx, cursor, "job:*", 100).Result()
		if err != nil {
			log.Printf("Error scanning job keys from Redis: %v", err)
			c.JSON(http.StatusOK, gin.H{
				"jobs":   []interface{}{},
				"total":  0,
				"limit":  limit,
				"offset": offset,
			})
			return
		}
		keys = append(keys, batch...)
		cursor = nextCursor
		if cursor == 0 {
			break
		}
	}

	type JobSummary struct {
		JobID          string `json:"job_id"`
		ModelType      string `json:"model_type"`
		Status         string `json:"status"`
		Progress       int32  `json:"progress"`
		TotalTasks     int32  `json:"total_tasks"`
		CompletedTasks int32  `json:"completed_tasks"`
		CreatedAt      int64  `json:"created_at"`
		UserID         string `json:"user_id,omitempty"`
	}

	// Load and parse all job records from Redis
	records := []JobRecord{}
	for _, key := range keys {
		jobData, err := gs.redisClient.Get(ctx, key).Result()
		if err != nil {
			log.Printf("Error fetching job %s: %v", key, err)
			continue
		}

		var job JobRecord
		if err := json.Unmarshal([]byte(jobData), &job); err != nil {
			log.Printf("Error parsing job %s: %v", key, err)
			continue
		}
		records = append(records, job)
	}

	// Sort jobs by creation time (newest first) before paginating
	sort.Slice(records, func(i, j int) bool {
		return records[i].CreatedAt > records[j].CreatedAt
	})

	total := len(records)

	// Apply pagination
	start := offset
	if start > total {
		start = total
	}
	end := start + limit
	if end > total {
		end = total
	}
	page := records[start:end]

	// Concurrently refresh statuses from the orchestrator for the current
	// page, bounded by a semaphore. On error, keep the Redis-stored status.
	jobs := make([]JobSummary, len(page))
	sem := make(chan struct{}, 8)
	var wg sync.WaitGroup
	for i, rec := range page {
		wg.Add(1)
		go func(i int, job JobRecord) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			// Progress calculated from the Redis-stored counts as fallback
			progress := int32(0)
			if job.TotalTasks > 0 {
				progress = int32(float64(job.CompletedTasks) / float64(job.TotalTasks) * 100)
			}

			statusCtx, statusCancel := context.WithTimeout(ctx, 3*time.Second)
			statusResp, err := gs.orchestratorClient.GetJobStatus(statusCtx, &orchestratorpb.GetJobStatusRequest{
				JobId: job.JobID,
			})
			statusCancel()

			if err == nil && statusResp != nil {
				job.Status = statusResp.Status
				job.CompletedTasks = statusResp.CompletedTasks
				job.TotalTasks = statusResp.TotalTasks
				progress = statusResp.Progress
			}

			jobs[i] = JobSummary{
				JobID:          job.JobID,
				ModelType:      job.ModelType,
				Status:         job.Status,
				Progress:       progress,
				TotalTasks:     job.TotalTasks,
				CompletedTasks: job.CompletedTasks,
				CreatedAt:      job.CreatedAt,
				UserID:         job.UserID,
			}
		}(i, rec)
	}
	wg.Wait()

	c.JSON(http.StatusOK, gin.H{
		"jobs":   jobs,
		"total":  total,
		"limit":  limit,
		"offset": offset,
	})
}

func (gs *GatewayServer) handleWorkerActivity(c *gin.Context) {
	// Get worker activity directly from orchestrator via gRPC
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := gs.orchestratorClient.GetWorkerActivity(ctx, &orchestratorpb.WorkerActivityRequest{})
	if err != nil {
		log.Printf("Error fetching worker activity from orchestrator: %v", err)
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "Failed to fetch worker activity",
			"details": err.Error(),
		})
		return
	}

	log.Printf("DEBUG: Got %d workers from orchestrator", resp.TotalWorkers)

	// Convert gRPC response to JSON format expected by frontend
	workers := make([]map[string]interface{}, 0)
	activeWorkers := 0
	busyWorkers := 0

	for _, worker := range resp.Workers {
		isActive := worker.Status == "BUSY" || worker.Status == "IDLE"
		if isActive {
			activeWorkers++
		}
		if worker.Status == "BUSY" {
			busyWorkers++
		}

		// Calculate uptime (assuming workers started when they first contacted orchestrator)
		// Ensure uptime is never negative by using max(0, calculated_uptime)
		uptime := max(0, time.Now().Unix() - worker.LastActivityTime)

		workers = append(workers, map[string]interface{}{
			"worker_id":           worker.WorkerId,
			"status":              worker.Status,
			"current_task_id":     worker.CurrentTaskId,
			"current_job_id":      worker.CurrentJobId,
			"tasks_completed":     worker.TasksCompleted,
			"last_activity_time":  worker.LastActivityTime,
			"cpu_usage":           nil, // No hardware sample is carried by this RPC.
			"memory_usage":        nil,
            "telemetry_available": false,
			"uptime":              uptime,
			"is_active":           isActive,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"workers":       workers,
		"total_workers": resp.TotalWorkers,
		"active_workers": activeWorkers,
		"busy_workers":   busyWorkers,
		"timestamp":      time.Now().Unix(),
	})
}

func (gs *GatewayServer) handleGetWorkers(c *gin.Context) {
	// Get worker activity from orchestrator via gRPC
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := gs.orchestratorClient.GetWorkerActivity(ctx, &orchestratorpb.WorkerActivityRequest{})
	if err != nil {
		log.Printf("Error fetching workers from orchestrator: %v", err)
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "Failed to fetch workers",
			"details": err.Error(),
		})
		return
	}

	log.Printf("DEBUG: Got %d workers from orchestrator for /api/v1/workers", resp.TotalWorkers)

	// Convert gRPC response to API format
	workers := make([]map[string]interface{}, 0)

	for _, worker := range resp.Workers {
		isActive := worker.Status == "BUSY" || worker.Status == "IDLE"
		
		workers = append(workers, map[string]interface{}{
			"id":                 worker.WorkerId,
			"worker_id":          worker.WorkerId,
			"status":             worker.Status,
			"current_task_id":    worker.CurrentTaskId,
			"current_job_id":     worker.CurrentJobId,
			"tasks_completed":    worker.TasksCompleted,
			"last_activity":      worker.LastActivityTime,
			"is_active":          isActive,
		})
	}

	c.JSON(http.StatusOK, workers)
}

func (gs *GatewayServer) handleCancelJob(c *gin.Context) {
	jobID := c.Param("id")
	
	log.Printf("Cancel request received for job: %s", jobID)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	// Call the orchestrator's CancelJob RPC
	resp, err := gs.orchestratorClient.CancelJob(ctx, &orchestratorpb.CancelJobRequest{
		JobId: jobID,
	})
	
	if err != nil {
		log.Printf("Error cancelling job: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to cancel job",
			"details": err.Error(),
		})
		return
	}
	
	if !resp.Success {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"message": resp.Message,
			"previous_status": resp.PreviousStatus,
		})
		return
	}
	
	log.Printf("Job %s successfully cancelled", jobID)
	
	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": resp.Message,
		"job_id": jobID,
		"previous_status": resp.PreviousStatus,
	})
}

func (gs *GatewayServer) Run() error {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           gs.router,
		ReadHeaderTimeout: 10 * time.Second,
	}

	// Serve in the background so we can wait for shutdown signals
	errCh := make(chan error, 1)
	go func() {
		log.Printf("API Gateway starting on port %s", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
		close(errCh)
	}()

	// Wait for SIGINT/SIGTERM (or a server error)
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errCh:
		gs.closeClients()
		return err
	case sig := <-quit:
		log.Printf("Received signal %s, shutting down gracefully...", sig)
	}

	// Gracefully drain in-flight requests with a 15s deadline
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	err := srv.Shutdown(shutdownCtx)
	if err != nil {
		log.Printf("Error during server shutdown: %v", err)
	}

	gs.closeClients()
	log.Println("API Gateway stopped")
	return err
}

// closeClients closes the gRPC connection and Redis client.
func (gs *GatewayServer) closeClients() {
	if gs.grpcConn != nil {
		if err := gs.grpcConn.Close(); err != nil {
			log.Printf("Error closing gRPC connection: %v", err)
		}
	}
	if gs.redisClient != nil {
		if err := gs.redisClient.Close(); err != nil {
			log.Printf("Error closing Redis client: %v", err)
		}
	}
}

func main() {
	server, err := NewGatewayServer()
	if err != nil {
		log.Fatalf("Failed to create gateway server: %v", err)
	}

	if err := server.Run(); err != nil {
		log.Fatalf("Failed to run server: %v", err)
	}
}
