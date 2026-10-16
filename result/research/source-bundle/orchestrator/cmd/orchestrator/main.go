package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	orchestratorpb "github.com/tensorfleet/orchestrator/proto/orchestrator"
)

// storageHTTPClient is shared by all autoSaveModel calls so we reuse
// connections instead of creating a new http.Client per call.
var storageHTTPClient = &http.Client{Timeout: 30 * time.Second}

const (
	defaultLeaseTimeout = 5 * time.Minute
	reaperInterval      = 30 * time.Second
)

// leaseInfo tracks a task that has been handed to a worker via AssignTask but
// whose completion has not yet been reported.
type leaseInfo struct {
	Task       *Task
	WorkerID   string
	AssignedAt time.Time
}

type OrchestratorServer struct {
	orchestratorpb.UnimplementedOrchestratorServiceServer
	redisClient    *redis.Client
	allowEphemeral bool
	jobs           map[string]*Job
	taskQueue      chan *Task
	deferredTasks  []*Task
	workers        map[string]*WorkerActivity // Track worker activity
	mu             sync.RWMutex

	// assigned tracks leased tasks by task ID so the reaper can requeue tasks
	// held by dead workers. completedTaskIDs records tasks whose completion has
	// been reported, so a late duplicate report (e.g. after a lease-expiry
	// requeue) never double-counts. Both are guarded by mu.
	assigned         map[string]*leaseInfo
	completedTaskIDs map[string]bool
	leaseTimeout     time.Duration

	// enqueueMu serializes all producers on taskQueue (CreateTrainingJob and
	// the lease reaper). Consumers only ever remove items, so a capacity check
	// performed while holding enqueueMu remains valid for subsequent sends.
	enqueueMu sync.Mutex
}

type Job struct {
	JobID           string
	UserID          string
	ModelType       string
	DatasetPath     string
	Hyperparameters map[string]string
	NumWorkers      int32
	Epochs          int32
	Status          string
	Tasks           []*Task
	CompletedTasks  int
	TotalTasks      int
	CurrentLoss     float64
	CurrentAccuracy float64
	ErrorMessage    string
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

type Task struct {
	LeaseToken  string
	Stage       string
	ShardIndex  int32
	TaskID      string
	JobID       string
	WorkerID    string
	Status      string
	Epoch       int32
	BatchStart  int32
	BatchEnd    int32
	Loss        float64
	Accuracy    float64
	CreatedAt   time.Time
	CompletedAt *time.Time
}

type WorkerActivity struct {
	WorkerID         string
	CurrentTaskID    string
	CurrentJobID     string
	TasksCompleted   int
	LastActivityTime time.Time
	Status           string // "IDLE", "BUSY"
}

// autoSaveModel triggers automatic model saving when job completes
func (s *OrchestratorServer) autoSaveModel(ctx context.Context, jobID string, job *Job) {
	storageURL := os.Getenv("STORAGE_SERVICE_URL")
	if storageURL == "" {
		storageURL = "http://storage:8081"
	}

	// Prepare job data to send to storage service
	jobData := map[string]interface{}{
		"job_id":           job.JobID,
		"job_name":         job.JobID, // Using JobID as job name for now
		"model_type":       job.ModelType,
		"dataset_path":     job.DatasetPath,
		"hyperparameters":  job.Hyperparameters,
		"current_accuracy": job.CurrentAccuracy,
		"current_loss":     job.CurrentLoss,
		"completed_tasks":  job.CompletedTasks,
		"total_tasks":      job.TotalTasks,
		"epochs":           job.Epochs,
		"num_workers":      job.NumWorkers,
		"status":           job.Status,
		"created_at":       job.CreatedAt.Format(time.RFC3339),
		"updated_at":       job.UpdatedAt.Format(time.RFC3339),
	}

	// Convert to JSON
	jsonData, err := json.Marshal(jobData)
	if err != nil {
		log.Printf("Warning: Failed to marshal job data for auto-save: %v", err)
		return
	}

	url := fmt.Sprintf("%s/api/v1/jobs/%s/auto-save-model", storageURL, jobID)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Warning: Failed to build auto-save request for job %s: %v", jobID, err)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := storageHTTPClient.Do(httpReq)
	if err != nil {
		log.Printf("Warning: Failed to auto-save model for job %s: %v", jobID, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == 201 {
		log.Printf("✅ Successfully auto-saved model for completed job %s", jobID)
	} else if resp.StatusCode == 200 {
		log.Printf("ℹ️  Model already exists for job %s", jobID)
	} else {
		log.Printf("⚠️  Failed to auto-save model for job %s (status: %d)", jobID, resp.StatusCode)
	}
}

func NewOrchestratorServer() (*OrchestratorServer, error) {
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "redis:6379"
	}

	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Printf("Warning: Redis connection failed: %v", err)
	} else {
		log.Println("Connected to Redis successfully")
	}

	leaseTimeout := defaultLeaseTimeout
	if v := os.Getenv("TASK_LEASE_TIMEOUT_SECONDS"); v != "" {
		if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
			leaseTimeout = time.Duration(secs) * time.Second
		}
	}

	server := &OrchestratorServer{
		allowEphemeral:   os.Getenv("ALLOW_EPHEMERAL_JOBS") == "true",
		redisClient:      rdb,
		jobs:             make(map[string]*Job),
		taskQueue:        make(chan *Task, 1000),
		workers:          make(map[string]*WorkerActivity),
		assigned:         make(map[string]*leaseInfo),
		completedTaskIDs: make(map[string]bool),
		leaseTimeout:     leaseTimeout,
	}
	if err := server.restoreFromRedis(ctx); err != nil {
		log.Printf("Warning: durable task recovery unavailable: %v", err)
	}
	return server, nil
}

// startLeaseReaper periodically requeues tasks whose lease expired — e.g.
// the assigned worker crashed and never reported completion. Without this,
// a single lost task leaves its job incomplete forever.
func (s *OrchestratorServer) startLeaseReaper(stop <-chan struct{}) {
	go func() {
		ticker := time.NewTicker(reaperInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				s.reapExpiredLeases()
			}
		}
	}()
}

func (s *OrchestratorServer) reapExpiredLeases() {
	now := time.Now()
	var toRequeue []*Task

	s.mu.Lock()
	for taskID, lease := range s.assigned {
		if now.Sub(lease.AssignedAt) < s.leaseTimeout {
			continue
		}
		delete(s.assigned, taskID)
		if s.completedTaskIDs[taskID] {
			continue
		}
		job := s.jobs[lease.Task.JobID]
		if job == nil || job.Status == "CANCELLED" || job.Status == "COMPLETED" || job.Status == "FAILED" {
			continue
		}
		lease.Task.Status = "PENDING"
		lease.Task.WorkerID = ""
		toRequeue = append(toRequeue, lease.Task)
	}
	s.mu.Unlock()

	if len(toRequeue) == 0 {
		return
	}

	s.enqueueMu.Lock()
	defer s.enqueueMu.Unlock()
	for _, task := range toRequeue {
		select {
		case s.taskQueue <- task:
			log.Printf("Requeued task %s (job %s) after lease expiry", task.TaskID, task.JobID)
		default:
			s.mu.Lock()
			s.deferredTasks = append(s.deferredTasks, task)
			s.mu.Unlock()
			log.Printf("Deferred retry of task %s until assignment capacity is available", task.TaskID)
		}
	}
}

func (s *OrchestratorServer) CreateTrainingJob(ctx context.Context, req *orchestratorpb.TrainingJobRequest) (*orchestratorpb.TrainingJobResponse, error) {
	log.Printf("Creating training job: %s for user: %s", req.JobId, req.UserId)
	if req.Epochs < 1 {
		return nil, status.Error(codes.InvalidArgument, "epochs must be positive")
	}
	if strings.HasPrefix(req.ModelType, "fleet_") {
		switch req.ModelType {
		case "fleet_random_forest", "fleet_logistic_regression", "fleet_decision_tree", "fleet_distributed_random_forest", "fleet_accelerated_logistic":
		default:
			return nil, status.Error(codes.InvalidArgument, "unsupported fleet algorithm")
		}
		if req.Epochs != 1 || req.Hyperparameters["target_column"] == "" {
			return nil, status.Error(codes.InvalidArgument, "fleet training requires epochs=1 and hyperparameters.target_column")
		}
		for key := range req.Hyperparameters {
			if strings.HasPrefix(key, "_fleet_") {
				return nil, status.Error(codes.InvalidArgument, "_fleet_ parameters are reserved for the scheduler")
			}
		}
		device := req.Hyperparameters["device"]
		if device != "" && device != "auto" && device != "cpu" && device != "mps" && device != "cuda" {
			return nil, status.Error(codes.InvalidArgument, "device must be auto, cpu, mps, or cuda")
		}
		if req.ModelType != "fleet_accelerated_logistic" && device != "" && device != "auto" && device != "cpu" {
			return nil, status.Error(codes.InvalidArgument, "this sklearn backend supports CPU only")
		}
	}

	job := &Job{
		JobID:           req.JobId,
		UserID:          req.UserId,
		ModelType:       req.ModelType,
		DatasetPath:     req.DatasetPath,
		Hyperparameters: req.Hyperparameters,
		NumWorkers:      req.NumWorkers,
		Epochs:          req.Epochs,
		Status:          "PENDING",
		Tasks:           []*Task{},
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	// Create tasks - split training across epochs and batches
	numBatches := int32(10) // Simulate 10 batches per epoch
	if strings.HasPrefix(req.ModelType, "fleet_") {
		numBatches = 1 // A complete independent fit, never independent epoch/batch fits.
	}
	for epoch := int32(0); epoch < req.Epochs; epoch++ {
		for batch := int32(0); batch < numBatches; batch++ {
			task := &Task{
				TaskID:     uuid.New().String(),
				JobID:      req.JobId,
				Status:     "PENDING",
				Epoch:      epoch,
				BatchStart: batch * 100,
				BatchEnd:   (batch + 1) * 100,
				CreatedAt:  time.Now(),
			}
			job.Tasks = append(job.Tasks, task)
		}
	}

	if req.ModelType == "fleet_distributed_random_forest" {
		var err error
		job.Tasks, err = distributedTasks(req)
		if err != nil {
			return nil, status.Error(codes.InvalidArgument, err.Error())
		}
	}
	job.TotalTasks = len(job.Tasks)
	job.Status = "RUNNING"

	// Backpressure: reserve queue capacity for the whole job up front, before
	// registering it. If the queue can't hold all tasks the job is rejected
	// with ResourceExhausted so the caller can retry later, instead of a
	// detached goroutine silently blocking on a full channel.
	s.enqueueMu.Lock()
	s.mu.RLock()
	deferred := len(s.deferredTasks)
	s.mu.RUnlock()
	if len(s.taskQueue)+deferred+len(job.Tasks) > cap(s.taskQueue) {
		queued := len(s.taskQueue)
		s.enqueueMu.Unlock()
		return nil, status.Errorf(codes.ResourceExhausted,
			"task queue saturated (%d/%d queued, job needs %d): retry later",
			queued, cap(s.taskQueue), len(job.Tasks))
	}

	// Persist initial state before exposing tasks. Failed durable admission
	// must not leave a running orphan behind an HTTP error.
	if err := s.saveJobToRedis(ctx, job); err != nil {
		if !s.allowEphemeral {
			s.enqueueMu.Unlock()
			return nil, status.Error(codes.Unavailable, "durable job admission unavailable: Redis persistence failed")
		}
		log.Printf("Explicit ephemeral admission for job %s: %v", job.JobID, err)
	}
	s.mu.Lock()
	s.jobs[req.JobId] = job
	s.mu.Unlock()
	for _, task := range job.Tasks {
		if task.Status != "WAITING" {
			s.taskQueue <- task
		}
	}
	s.enqueueMu.Unlock()

	log.Printf("Created job %s with %d tasks", req.JobId, job.TotalTasks)

	return &orchestratorpb.TrainingJobResponse{
		JobId:    req.JobId,
		Status:   "RUNNING",
		NumTasks: int32(job.TotalTasks),
		Message:  fmt.Sprintf("Job created with %d tasks", job.TotalTasks),
	}, nil
}

func (s *OrchestratorServer) GetJobStatus(ctx context.Context, req *orchestratorpb.GetJobStatusRequest) (*orchestratorpb.GetJobStatusResponse, error) {
	s.mu.RLock()
	job, exists := s.jobs[req.JobId]
	s.mu.RUnlock()

	if !exists {
		// Try to load from Redis
		var err error
		job, err = s.loadJobFromRedis(ctx, req.JobId)
		if err != nil {
			return nil, fmt.Errorf("job not found: %s", req.JobId)
		}
		s.mu.Lock()
		s.jobs[req.JobId] = job
		s.mu.Unlock()
	}

	// Completion and cancellation update job fields concurrently with polling.
	s.mu.RLock()
	defer s.mu.RUnlock()
	progress := int32(0)
	if job.TotalTasks > 0 {
		progress = int32(float64(job.CompletedTasks) / float64(job.TotalTasks) * 100)
	}

	message := fmt.Sprintf("Completed %d/%d tasks", job.CompletedTasks, job.TotalTasks)
	if job.ErrorMessage != "" {
		message = job.ErrorMessage
	}
	if strings.HasPrefix(job.ModelType, "fleet_") && job.Status == "COMPLETED" {
		message = "s3://artifacts/fleet/" + job.JobID + "/result.json"
	}
	return &orchestratorpb.GetJobStatusResponse{
		JobId:           job.JobID,
		Status:          job.Status,
		Progress:        progress,
		CompletedTasks:  int32(job.CompletedTasks),
		TotalTasks:      int32(job.TotalTasks),
		CurrentLoss:     job.CurrentLoss,
		CurrentAccuracy: job.CurrentAccuracy,
		Message:         message,
	}, nil
}

func (s *OrchestratorServer) AssignTask(ctx context.Context, req *orchestratorpb.AssignTaskRequest) (*orchestratorpb.AssignTaskResponse, error) {
	deadline := time.After(5 * time.Second)
	for {
		s.mu.Lock()
		deferredTask := s.takeDeferred(req.Capabilities)
		s.mu.Unlock()
		task := deferredTask
		if task == nil {
			select {
			case task = <-s.taskQueue:
			case <-deadline:
				return nil, fmt.Errorf("no compatible tasks available")
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		{
			s.mu.Lock()
			job := s.jobs[task.JobID]

			// Drain tasks that no longer have work to do: orphaned jobs and
			// jobs cancelled after their tasks were queued.
			if job == nil {
				s.mu.Unlock()
				log.Printf("Dropping task %s: job %s not found", task.TaskID, task.JobID)
				continue
			}
			if job.Status == "CANCELLED" || job.Status == "FAILED" || job.Status == "COMPLETED" {
				jobStatus := job.Status
				s.mu.Unlock()
				log.Printf("Dropping task %s: job %s is %s", task.TaskID, task.JobID, jobStatus)
				continue
			}
			if !matchesWorker(job, req.Capabilities) {
				s.deferredTasks = append(s.deferredTasks, task)
				s.mu.Unlock()
				continue
			}

			task.WorkerID = req.WorkerId
			task.Status = "ASSIGNED"
			task.LeaseToken = uuid.New().String()

			// Update worker activity and record the lease so a dead worker's
			// task gets requeued by the reaper.
			workerActivity, ok := s.workers[req.WorkerId]
			if !ok {
				workerActivity = &WorkerActivity{
					WorkerID:         req.WorkerId,
					Status:           "BUSY",
					TasksCompleted:   0,
					LastActivityTime: time.Now(),
				}
				s.workers[req.WorkerId] = workerActivity
			}
			workerActivity.CurrentTaskID = task.TaskID
			workerActivity.CurrentJobID = task.JobID
			workerActivity.Status = "BUSY"
			workerActivity.LastActivityTime = time.Now()
			s.assigned[task.TaskID] = &leaseInfo{
				Task:       task,
				WorkerID:   req.WorkerId,
				AssignedAt: time.Now(),
			}
			parameters := make(map[string]string, len(job.Hyperparameters)+3)
			for key, value := range job.Hyperparameters {
				parameters[key] = value
			}
			if job.ModelType == "fleet_distributed_random_forest" {
				parameters["_fleet_stage"] = task.Stage
				parameters["_fleet_shard_index"] = strconv.Itoa(int(task.ShardIndex))
				parameters["_fleet_shards"] = strconv.Itoa(int(job.NumWorkers))
			}
			response := &orchestratorpb.AssignTaskResponse{
				TaskId:          task.TaskID,
				JobId:           task.JobID,
				ModelType:       job.ModelType,
				DatasetPath:     job.DatasetPath,
				Hyperparameters: parameters,
				Epoch:           task.Epoch,
				BatchStart:      task.BatchStart,
				BatchEnd:        task.BatchEnd,
				LeaseToken:      task.LeaseToken,
			}
			s.mu.Unlock()
			log.Printf("Assigned task %s (epoch %d) to worker %s", response.TaskId, response.Epoch, req.WorkerId)
			return response, nil
		}
	}
}

func (s *OrchestratorServer) ReportTaskCompletion(ctx context.Context, req *orchestratorpb.TaskCompletionRequest) (*orchestratorpb.TaskCompletionResponse, error) {
	log.Printf("Task %s completed by worker %s: success=%v, loss=%.4f, accuracy=%.4f",
		req.TaskId, req.WorkerId, req.Success, req.Loss, req.Accuracy)

	s.mu.Lock()
	defer s.mu.Unlock()

	job, exists := s.jobs[req.JobId]
	if !exists {
		return nil, fmt.Errorf("job not found")
	}
	if strings.HasPrefix(job.ModelType, "fleet_") && (job.Status == "CANCELLED" || job.Status == "FAILED") {
		return &orchestratorpb.TaskCompletionResponse{Acknowledged: true, Message: "Job already terminal"}, nil
	}

	// Release the lease and ignore duplicate reports (a slow worker may report
	// after the reaper already requeued and another worker completed the task).
	if s.completedTaskIDs[req.TaskId] {
		return &orchestratorpb.TaskCompletionResponse{
			Acknowledged: true,
			Message:      "Duplicate task completion ignored",
		}, nil
	}
	lease := s.assigned[req.TaskId]
	if lease == nil || lease.WorkerID != req.WorkerId || lease.Task.LeaseToken != req.LeaseToken || req.LeaseToken == "" {
		return nil, status.Error(codes.FailedPrecondition, "task lease is stale or not owned by this worker")
	}
	delete(s.assigned, req.TaskId)
	task := lease.Task

	if req.Success {
		task.Status = "COMPLETED"
		finished := time.Now()
		task.CompletedAt = &finished
		s.completedTaskIDs[req.TaskId] = true
		job.CompletedTasks++
		job.CurrentLoss = req.Loss
		job.CurrentAccuracy = req.Accuracy
		job.UpdatedAt = time.Now()
		if job.ModelType == "fleet_distributed_random_forest" && job.CompletedTasks == job.TotalTasks-1 {
			go s.enqueueAggregation(job.JobID)
		}

		if job.CompletedTasks >= job.TotalTasks {
			job.Status = "COMPLETED"
			log.Printf("Job %s completed!", req.JobId)

			// Real workers saved and verified weights before reporting success;
			// the legacy auto-save endpoint only creates a JSON summary.
			if !strings.HasPrefix(job.ModelType, "fleet_") {
				saveCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				go func() {
					defer cancel()
					s.autoSaveModel(saveCtx, req.JobId, job)
				}()
			}
		}
	} else if strings.HasPrefix(job.ModelType, "fleet_") {
		task.Status = "FAILED"
		job.Status = "FAILED"
		job.ErrorMessage = req.ErrorMessage
		job.UpdatedAt = time.Now()
	} else if job.Status == "RUNNING" {
		task.Status = "PENDING"
		// Failed attempt: requeue the task for another worker to retry.
		s.mu.Unlock()
		s.enqueueMu.Lock()
		requeued := false
		select {
		case s.taskQueue <- task:
			requeued = true
		default:
			s.mu.Lock()
			s.deferredTasks = append(s.deferredTasks, task)
			s.mu.Unlock()
			requeued = true
		}
		s.enqueueMu.Unlock()
		s.mu.Lock()
		if requeued {
			log.Printf("Task %s failed on worker %s, requeued for retry", req.TaskId, req.WorkerId)
		} else {
			log.Printf("Warning: task %s failed and queue is full, not requeued", req.TaskId)
		}
	}

	// Update worker activity
	workerActivity, ok := s.workers[req.WorkerId]
	if ok {
		if req.Success {
			workerActivity.TasksCompleted++
		}
		workerActivity.Status = "IDLE"
		workerActivity.LastActivityTime = time.Now()
	}

	// Save to Redis
	if err := s.saveJobToRedis(ctx, job); err != nil {
		log.Printf("Warning: Failed to save job to Redis: %v", err)
	}

	return &orchestratorpb.TaskCompletionResponse{
		Acknowledged: true,
		Message:      "Task completion recorded",
	}, nil
}

func (s *OrchestratorServer) UpdateJobMetrics(ctx context.Context, req *orchestratorpb.JobMetricsRequest) (*orchestratorpb.JobMetricsResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, exists := s.jobs[req.JobId]
	if !exists {
		return &orchestratorpb.JobMetricsResponse{Success: false}, nil
	}

	job.CurrentLoss = req.Loss
	job.CurrentAccuracy = req.Accuracy
	job.UpdatedAt = time.Now()

	return &orchestratorpb.JobMetricsResponse{Success: true}, nil
}

func (s *OrchestratorServer) CancelJob(ctx context.Context, req *orchestratorpb.CancelJobRequest) (*orchestratorpb.CancelJobResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, exists := s.jobs[req.JobId]
	if !exists {
		// Try to load from Redis
		var err error
		job, err = s.loadJobFromRedis(ctx, req.JobId)
		if err != nil {
			return &orchestratorpb.CancelJobResponse{
				Success: false,
				Message: fmt.Sprintf("Job not found: %s", req.JobId),
			}, nil
		}
		s.jobs[req.JobId] = job
	}

	previousStatus := job.Status

	// Check if job can be cancelled
	if job.Status == "COMPLETED" || job.Status == "FAILED" || job.Status == "CANCELLED" {
		return &orchestratorpb.CancelJobResponse{
			Success:        false,
			Message:        fmt.Sprintf("Cannot cancel job with status: %s", job.Status),
			PreviousStatus: previousStatus,
		}, nil
	}

	// Update job status to CANCELLED
	job.Status = "CANCELLED"
	job.UpdatedAt = time.Now()

	// Save to Redis
	if err := s.saveJobToRedis(ctx, job); err != nil {
		log.Printf("Failed to save cancelled job to Redis: %v", err)
	}

	log.Printf("Job %s cancelled (previous status: %s)", req.JobId, previousStatus)

	return &orchestratorpb.CancelJobResponse{
		Success:        true,
		Message:        fmt.Sprintf("Job %s has been cancelled", req.JobId),
		PreviousStatus: previousStatus,
	}, nil
}

func (s *OrchestratorServer) GetWorkerActivity(ctx context.Context, req *orchestratorpb.WorkerActivityRequest) (*orchestratorpb.WorkerActivityResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	workers := make([]*orchestratorpb.WorkerInfo, 0, len(s.workers))
	for _, worker := range s.workers {
		workers = append(workers, &orchestratorpb.WorkerInfo{
			WorkerId:         worker.WorkerID,
			Status:           worker.Status,
			CurrentTaskId:    worker.CurrentTaskID,
			CurrentJobId:     worker.CurrentJobID,
			TasksCompleted:   int32(worker.TasksCompleted),
			LastActivityTime: worker.LastActivityTime.Unix(),
		})
	}

	return &orchestratorpb.WorkerActivityResponse{
		Workers:      workers,
		TotalWorkers: int32(len(workers)),
	}, nil
}

func (s *OrchestratorServer) saveJobToRedis(ctx context.Context, job *Job) error {
	data, err := json.Marshal(job)
	if err != nil {
		return err
	}

	return s.redisClient.Set(ctx, "state:job:"+job.JobID, data, 24*time.Hour).Err()
}

func (s *OrchestratorServer) loadJobFromRedis(ctx context.Context, jobID string) (*Job, error) {
	data, err := s.redisClient.Get(ctx, "state:job:"+jobID).Bytes()
	if err == redis.Nil {
		data, err = s.redisClient.Get(ctx, "job:"+jobID).Bytes()
	}
	if err != nil {
		return nil, err
	}

	var job Job
	if err := json.Unmarshal(data, &job); err != nil {
		return nil, err
	}

	return &job, nil
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "50051"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	server, err := NewOrchestratorServer()
	if err != nil {
		log.Fatalf("Failed to create orchestrator: %v", err)
	}

	grpcServer := grpc.NewServer()
	orchestratorpb.RegisterOrchestratorServiceServer(grpcServer, server)

	reaperStop := make(chan struct{})
	server.startLeaseReaper(reaperStop)

	go func() {
		log.Printf("Orchestrator server listening on port %s", port)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve: %v", err)
		}
	}()

	// Graceful shutdown: let in-flight RPCs finish, force-stop after 15s.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigCh
	log.Printf("Received %s, shutting down gracefully...", sig)

	close(reaperStop)
	done := make(chan struct{})
	go func() {
		grpcServer.GracefulStop()
		close(done)
	}()
	select {
	case <-done:
		log.Println("gRPC server stopped gracefully")
	case <-time.After(15 * time.Second):
		log.Println("Graceful stop timed out, forcing shutdown")
		grpcServer.Stop()
	}

	if err := server.redisClient.Close(); err != nil {
		log.Printf("Error closing Redis client: %v", err)
	}
}
