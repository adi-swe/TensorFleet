package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	orchestratorpb "github.com/tensorfleet/worker/proto/orchestrator"
	workerpb "github.com/tensorfleet/worker/proto/worker"
)

const (
	// rpcTimeout bounds every unary RPC to the orchestrator.
	rpcTimeout = 5 * time.Second
	// reportTimeout bounds ReportTaskCompletion during task execution.
	reportTimeout = 10 * time.Second
	// cancelPollInterval is how often a running task polls the
	// orchestrator to see whether its job was cancelled.
	cancelPollInterval = 3 * time.Second
	// shutdownGracePeriod is how long we wait for in-flight tasks on shutdown.
	shutdownGracePeriod = 30 * time.Second
	// defaultMaxConcurrentTasks is used when MAX_CONCURRENT_TASKS is unset/invalid.
	defaultMaxConcurrentTasks = 2
)

var (
	taskDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name: "worker_task_duration_seconds",
		Help: "Time taken to complete a task",
	})
	tasksCompleted = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "worker_tasks_completed_total",
		Help: "Total number of tasks completed",
	})
	tasksFailed = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "worker_tasks_failed_total",
		Help: "Total number of tasks failed",
	})
)

func init() {
	prometheus.MustRegister(taskDuration)
	prometheus.MustRegister(tasksCompleted)
	prometheus.MustRegister(tasksFailed)
}

type WorkerServer struct {
	workerpb.UnimplementedWorkerServiceServer
	workerID           string
	capabilities       []string
	orchestratorClient orchestratorpb.OrchestratorServiceClient
	currentTasks       atomic.Int64
	completedTasks     atomic.Int64
	taskSem            chan struct{}  // bounds concurrent task execution
	taskWG             sync.WaitGroup // tracks in-flight fetched tasks for shutdown
}

func maxConcurrentTasks() int {
	if v := os.Getenv("MAX_CONCURRENT_TASKS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
		log.Printf("Invalid MAX_CONCURRENT_TASKS=%q, using default %d", v, defaultMaxConcurrentTasks)
	}
	return defaultMaxConcurrentTasks
}

func NewWorkerServer() (*WorkerServer, error) {
	workerID := uuid.New().String()

	orchestratorAddr := os.Getenv("ORCHESTRATOR_ADDR")
	if orchestratorAddr == "" {
		orchestratorAddr = "orchestrator:50051"
	}

	log.Printf("Connecting to orchestrator at %s", orchestratorAddr)
	conn, err := grpc.Dial(orchestratorAddr,
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

	ws := &WorkerServer{
		workerID:           workerID,
		capabilities:       detectCapabilities(),
		orchestratorClient: client,
		taskSem:            make(chan struct{}, maxConcurrentTasks()),
	}

	return ws, nil
}

// isJobCancelled checks if a job has been cancelled or failed
func (ws *WorkerServer) isJobCancelled(ctx context.Context, jobID string) (bool, error) {
	// Create a context with timeout for the status check
	checkCtx, cancel := context.WithTimeout(ctx, rpcTimeout)
	defer cancel()

	resp, err := ws.orchestratorClient.GetJobStatus(checkCtx, &orchestratorpb.GetJobStatusRequest{
		JobId: jobID,
	})

	if err != nil {
		log.Printf("Failed to check job status: %v", err)
		return false, err
	}

	// Consider job cancelled if status is CANCELLED or FAILED
	return resp.Status == "CANCELLED" || resp.Status == "FAILED", nil
}

func (ws *WorkerServer) ExecuteTask(ctx context.Context, req *workerpb.TaskRequest) (*workerpb.TaskResponse, error) {
	start := time.Now()
	log.Printf("Worker %s executing task %s (epoch %d, batches %d-%d)",
		ws.workerID, req.TaskId, req.Epoch, req.BatchStart, req.BatchEnd)

	ws.currentTasks.Add(1)
	defer ws.currentTasks.Add(-1)

	// Check if job is cancelled before starting
	if cancelled, err := ws.isJobCancelled(ctx, req.JobId); err == nil && cancelled {
		log.Printf("Task %s aborted - job %s was cancelled", req.TaskId, req.JobId)
		return &workerpb.TaskResponse{
			TaskId:  req.TaskId,
			Success: false,
			Message: "Task aborted - job was cancelled",
		}, nil
	}

	var success bool
	var loss, accuracy float64
	message := "Task completed successfully"
	if strings.HasPrefix(req.ModelType, "fleet_") {
		var err error
		loss, accuracy, message, err = ws.trainRealModel(ctx, req)
		success = err == nil
		if err != nil {
			message = err.Error()
		}
	} else {
		success, loss, accuracy = ws.simulateTraining(ctx, req)
	}

	duration := time.Since(start).Seconds()
	taskDuration.Observe(duration)

	if success {

		// Report completion to orchestrator
		reportCtx, cancel := context.WithTimeout(ctx, reportTimeout)
		_, err := ws.orchestratorClient.ReportTaskCompletion(reportCtx, &orchestratorpb.TaskCompletionRequest{
			TaskId:     req.TaskId,
			JobId:      req.JobId,
			WorkerId:   ws.workerID,
			LeaseToken: req.LeaseToken,
			Success:    true,
			Loss:       loss,
			Accuracy:   accuracy,
			// Real artifacts travel through object storage, not gRPC payloads.
			ModelWeights: []byte{},
		})
		cancel()

		if err != nil {
			log.Printf("Failed to report task completion: %v", err)
			tasksFailed.Inc()
			return &workerpb.TaskResponse{TaskId: req.TaskId, Success: false, Message: "Completion acknowledgement failed: " + err.Error()}, nil
		}
		tasksCompleted.Inc()
		ws.completedTasks.Add(1)

		log.Printf("Task %s completed successfully. Loss: %.4f, Accuracy: %.4f",
			req.TaskId, loss, accuracy)

		return &workerpb.TaskResponse{
			TaskId:       req.TaskId,
			Success:      true,
			Message:      message,
			Loss:         loss,
			Accuracy:     accuracy,
			ModelWeights: []byte{},
		}, nil
	}

	tasksFailed.Inc()
	if strings.HasPrefix(req.ModelType, "fleet_") {
		reportCtx, cancel := context.WithTimeout(context.Background(), reportTimeout)
		defer cancel()
		if _, err := ws.orchestratorClient.ReportTaskCompletion(reportCtx, &orchestratorpb.TaskCompletionRequest{
			TaskId: req.TaskId, JobId: req.JobId, WorkerId: ws.workerID, LeaseToken: req.LeaseToken,
			Success: false, ErrorMessage: message,
		}); err != nil {
			log.Printf("Failed to report training failure: %v", err)
		}
	}
	return &workerpb.TaskResponse{
		TaskId:  req.TaskId,
		Success: false,
		Message: message,
	}, nil
}

func (ws *WorkerServer) simulateTraining(ctx context.Context, req *workerpb.TaskRequest) (bool, float64, float64) {
	// Simulate ML training with periodic status checks
	totalDuration := time.Duration(rand.Intn(3000)+1000) * time.Millisecond
	checkInterval := cancelPollInterval
	elapsed := time.Duration(0)

	// Check job status periodically during training
	for elapsed < totalDuration {
		sleepTime := checkInterval
		if totalDuration-elapsed < checkInterval {
			sleepTime = totalDuration - elapsed
		}

		time.Sleep(sleepTime)
		elapsed += sleepTime

		// Stop early if training is complete; no need for a final cancel poll
		if elapsed >= totalDuration {
			break
		}

		// Check if job was cancelled during training
		if cancelled, err := ws.isJobCancelled(ctx, req.JobId); err == nil && cancelled {
			log.Printf("Training interrupted - job %s was cancelled", req.JobId)
			return false, 0, 0
		}
	}

	// Simulate convergence: loss decreases, accuracy increases over epochs
	baseLoss := 2.5
	baseAccuracy := 0.1

	loss := baseLoss/(1+float64(req.Epoch)*0.2) + (rand.Float64()-0.5)*0.1
	accuracy := baseAccuracy + float64(req.Epoch)*0.08 + (rand.Float64()-0.5)*0.02

	if loss < 0 {
		loss = 0.01
	}
	if accuracy > 1.0 {
		accuracy = 0.99
	}

	return true, loss, accuracy
}

func (ws *WorkerServer) GetWorkerStatus(ctx context.Context, req *workerpb.WorkerStatusRequest) (*workerpb.WorkerStatusResponse, error) {
	return &workerpb.WorkerStatusResponse{
		WorkerId:       ws.workerID,
		Status:         "ACTIVE",
		CurrentTasks:   int32(ws.currentTasks.Load()),
		CompletedTasks: int32(ws.completedTasks.Load()),
		CpuUsage:       -1, // Unknown: never fabricate utilization.
		MemoryUsage:    -1,
	}, nil
}

func (ws *WorkerServer) CancelTask(ctx context.Context, req *workerpb.CancelTaskRequest) (*workerpb.CancelTaskResponse, error) {
	return &workerpb.CancelTaskResponse{
		Success: true,
		Message: fmt.Sprintf("Task %s cancelled", req.TaskId),
	}, nil
}

// fetchInterval returns how often the worker polls for new tasks.
// FETCH_INTERVAL_MS tunes work-pickup latency vs orchestrator load.
func fetchInterval() time.Duration {
	if v := os.Getenv("FETCH_INTERVAL_MS"); v != "" {
		if ms, err := strconv.Atoi(v); err == nil && ms >= 100 {
			return time.Duration(ms) * time.Millisecond
		}
		log.Printf("Invalid FETCH_INTERVAL_MS=%q, using default 5s", v)
	}
	return 5 * time.Second
}

func (ws *WorkerServer) startTaskFetcher(ctx context.Context) {
	ticker := time.NewTicker(fetchInterval())
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ws.fetchAndExecuteTask(ctx)
		}
	}
}

func (ws *WorkerServer) fetchAndExecuteTask(ctx context.Context) {
	// Acquire a concurrency slot; if all slots are busy, skip this tick.
	select {
	case ws.taskSem <- struct{}{}:
	default:
		return
	}

	taskCtx, cancel := context.WithTimeout(ctx, rpcTimeout)
	defer cancel()

	resp, err := ws.orchestratorClient.AssignTask(taskCtx, &orchestratorpb.AssignTaskRequest{
		WorkerId:     ws.workerID,
		Capabilities: ws.capabilities,
	})

	if err != nil {
		// No tasks available or error
		<-ws.taskSem
		return
	}

	// Execute task
	taskReq := &workerpb.TaskRequest{
		TaskId:          resp.TaskId,
		JobId:           resp.JobId,
		ModelType:       resp.ModelType,
		DatasetPath:     resp.DatasetPath,
		Hyperparameters: resp.Hyperparameters,
		Epoch:           resp.Epoch,
		BatchStart:      resp.BatchStart,
		BatchEnd:        resp.BatchEnd,
		LeaseToken:      resp.LeaseToken,
	}

	ws.taskWG.Add(1)
	go func() {
		defer ws.taskWG.Done()
		defer func() { <-ws.taskSem }()
		if _, err := ws.ExecuteTask(context.Background(), taskReq); err != nil {
			log.Printf("Task %s execution error: %v", taskReq.TaskId, err)
		}
	}()
}

func main() {
	worker, err := NewWorkerServer()
	if err != nil {
		log.Fatalf("Failed to create worker: %v", err)
	}

	// Start Prometheus metrics server. METRICS_PORT allows multiple workers
	// on one host (e.g. local benchmarking); set to "0" for an ephemeral port.
	metricsPort := os.Getenv("METRICS_PORT")
	if metricsPort == "" {
		metricsPort = "2112"
	}
	metricsMux := http.NewServeMux()
	metricsMux.Handle("/metrics", promhttp.Handler())
	metricsServer := &http.Server{
		Addr:              ":" + metricsPort,
		Handler:           metricsMux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
	}
	go func() {
		log.Printf("Metrics server listening on :%s", metricsPort)
		if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("Metrics server error: %v", err)
		}
	}()

	// Start task fetcher; cancelled on shutdown to stop pulling new tasks.
	fetchCtx, stopFetcher := context.WithCancel(context.Background())
	go worker.startTaskFetcher(fetchCtx)

	// Start gRPC server
	port := os.Getenv("PORT")
	if port == "" {
		port = "50052"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	workerpb.RegisterWorkerServiceServer(grpcServer, worker)

	go func() {
		log.Printf("Worker %s listening on port %s", worker.workerID, port)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve: %v", err)
		}
	}()

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigCh
	log.Printf("Received signal %v, shutting down", sig)

	// Stop fetching new tasks, then wait for in-flight tasks to finish.
	stopFetcher()
	tasksDone := make(chan struct{})
	go func() {
		worker.taskWG.Wait()
		close(tasksDone)
	}()
	select {
	case <-tasksDone:
		log.Println("All in-flight tasks completed")
	case <-time.After(shutdownGracePeriod):
		log.Println("Timed out waiting for in-flight tasks")
	}

	// Drain gRPC connections, then stop the metrics server.
	grpcServer.GracefulStop()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := metricsServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("Metrics server shutdown error: %v", err)
	}

	log.Println("Worker shut down cleanly")
}
