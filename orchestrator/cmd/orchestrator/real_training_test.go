package main

import (
	"context"
	"testing"
	"time"

	"github.com/go-redis/redis/v8"
	pb "github.com/tensorfleet/orchestrator/proto/orchestrator"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func fleetTestServer(t *testing.T) *OrchestratorServer {
	t.Helper()
	rdb := redis.NewClient(&redis.Options{Addr: "127.0.0.1:1", MaxRetries: -1,
		DialTimeout: time.Millisecond, ReadTimeout: time.Millisecond, WriteTimeout: time.Millisecond})
	t.Cleanup(func() { rdb.Close() })
	return &OrchestratorServer{redisClient: rdb, jobs: map[string]*Job{},
		allowEphemeral: true,
		taskQueue:      make(chan *Task, 1000), workers: map[string]*WorkerActivity{},
		assigned: map[string]*leaseInfo{}, completedTaskIDs: map[string]bool{}}
}

func TestFleetJobIsOneWholeFit(t *testing.T) {
	s := fleetTestServer(t)
	ctx := context.Background()
	request := &pb.TrainingJobRequest{JobId: "real", ModelType: "fleet_random_forest",
		DatasetPath: "datasets/iris.csv", Epochs: 1, Hyperparameters: map[string]string{"target_column": "species"}}
	created, err := s.CreateTrainingJob(ctx, request)
	if err != nil || created.NumTasks != 1 {
		t.Fatalf("expected one fit, got %v, %v", created, err)
	}
	task, err := s.AssignTask(ctx, &pb.AssignTaskRequest{WorkerId: "worker"})
	if err != nil || task.ModelType != request.ModelType || task.Hyperparameters["target_column"] != "species" {
		t.Fatalf("training payload not preserved: %v, %v", task, err)
	}
	completion := &pb.TaskCompletionRequest{JobId: "real", TaskId: task.TaskId, WorkerId: "worker", LeaseToken: task.LeaseToken, Success: true, Accuracy: .9}
	if _, err := s.ReportTaskCompletion(ctx, completion); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ReportTaskCompletion(ctx, completion); err != nil {
		t.Fatal(err)
	}
	state, err := s.GetJobStatus(ctx, &pb.GetJobStatusRequest{JobId: "real"})
	if err != nil || state.CompletedTasks != 1 || state.Status != "COMPLETED" || state.Message != "s3://artifacts/fleet/real/result.json" {
		t.Fatalf("unexpected final state: %v, %v", state, err)
	}
}

func TestFleetFailureIsTerminalAndCannotBeOverwritten(t *testing.T) {
	s := fleetTestServer(t)
	s.jobs["real"] = &Job{JobID: "real", ModelType: "fleet_random_forest", Status: "RUNNING", TotalTasks: 1}
	s.assigned["task"] = &leaseInfo{Task: &Task{TaskID: "task", JobID: "real", LeaseToken: "token"}}
	ctx := context.Background()
	_, err := s.ReportTaskCompletion(ctx, &pb.TaskCompletionRequest{JobId: "real", TaskId: "task", LeaseToken: "token", ErrorMessage: "dataset missing"})
	if err != nil {
		t.Fatal(err)
	}
	s.ReportTaskCompletion(ctx, &pb.TaskCompletionRequest{JobId: "real", TaskId: "task", Success: true})
	state, _ := s.GetJobStatus(ctx, &pb.GetJobStatusRequest{JobId: "real"})
	if state.Status != "FAILED" || state.CompletedTasks != 0 || state.Message != "dataset missing" || len(s.taskQueue) != 0 {
		t.Fatalf("failed fit reported success or retried forever: %v", state)
	}
}

func TestFleetRejectsSyntheticBatchSemantics(t *testing.T) {
	s := fleetTestServer(t)
	for _, request := range []*pb.TrainingJobRequest{
		{ModelType: "fleet_unknown", Epochs: 1},
		{ModelType: "fleet_random_forest", Epochs: 10, Hyperparameters: map[string]string{"target_column": "target"}},
		{ModelType: "fleet_random_forest", Epochs: 1},
	} {
		if _, err := s.CreateTrainingJob(context.Background(), request); status.Code(err) != codes.InvalidArgument {
			t.Fatalf("invalid fleet request accepted: %v", err)
		}
	}
}

func TestLegacyBenchmarkRetainsSimulatedTaskCount(t *testing.T) {
	s := fleetTestServer(t)
	job, err := s.CreateTrainingJob(context.Background(), &pb.TrainingJobRequest{JobId: "simulation", ModelType: "benchmark", Epochs: 2})
	if err != nil || job.NumTasks != 20 {
		t.Fatalf("legacy benchmark changed: %v %v", job, err)
	}
}

func TestDistributedForestSchedulesAggregationAfterShards(t *testing.T) {
	s := fleetTestServer(t)
	ctx := context.Background()
	request := &pb.TrainingJobRequest{JobId: "distributed", ModelType: "fleet_distributed_random_forest",
		Epochs: 1, NumWorkers: 2, Hyperparameters: map[string]string{"target_column": "target", "estimator_params": `{"n_estimators":13}`}}
	created, err := s.CreateTrainingJob(ctx, request)
	if err != nil || created.NumTasks != 3 || len(s.taskQueue) != 2 {
		t.Fatalf("bad shard graph: %v %v", created, err)
	}
	for index := 0; index < 2; index++ {
		task, err := s.AssignTask(ctx, &pb.AssignTaskRequest{WorkerId: "cpu"})
		if err != nil || task.Hyperparameters["_fleet_stage"] != "shard" {
			t.Fatalf("unexpected task: %v %v", task, err)
		}
		if _, err := s.ReportTaskCompletion(ctx, &pb.TaskCompletionRequest{JobId: "distributed", TaskId: task.TaskId, WorkerId: "cpu", LeaseToken: task.LeaseToken, Success: true}); err != nil {
			t.Fatal(err)
		}
	}
	select {
	case task := <-s.taskQueue:
		if task.Stage != "aggregate" {
			t.Fatal("expected aggregation")
		}
	case <-time.After(time.Second):
		t.Fatal("aggregation never queued")
	}
	state, _ := s.GetJobStatus(ctx, &pb.GetJobStatusRequest{JobId: "distributed"})
	if state.Status != "RUNNING" || state.CompletedTasks != 2 {
		t.Fatalf("job completed before aggregation: %v", state)
	}
}

func TestGPUJobWaitsForCompatibleWorkerWithoutBlockingCPU(t *testing.T) {
	s := fleetTestServer(t)
	ctx := context.Background()
	for _, request := range []*pb.TrainingJobRequest{
		{JobId: "gpu", ModelType: "fleet_accelerated_logistic", Epochs: 1, Hyperparameters: map[string]string{"target_column": "target", "device": "mps"}},
		{JobId: "cpu", ModelType: "fleet_random_forest", Epochs: 1, Hyperparameters: map[string]string{"target_column": "target"}},
	} {
		if _, err := s.CreateTrainingJob(ctx, request); err != nil {
			t.Fatal(err)
		}
	}
	task, err := s.AssignTask(ctx, &pb.AssignTaskRequest{WorkerId: "docker"})
	if err != nil || task.JobId != "cpu" {
		t.Fatalf("CPU job blocked behind GPU work: %v %v", task, err)
	}
	task, err = s.AssignTask(ctx, &pb.AssignTaskRequest{WorkerId: "native", Capabilities: []string{"fleet_accelerated_logistic:mps"}})
	if err != nil || task.JobId != "gpu" {
		t.Fatalf("MPS job not assigned to compatible worker: %v %v", task, err)
	}
}

func TestRecoveryRequeuesOnlyUnfinishedWorkAndFencesOldLease(t *testing.T) {
	s := fleetTestServer(t)
	job := &Job{JobID: "recovery", ModelType: "fleet_distributed_random_forest", NumWorkers: 2,
		Status: "RUNNING", Tasks: []*Task{
			{TaskID: "done", JobID: "recovery", Stage: "shard", Status: "COMPLETED"},
			{TaskID: "lost", JobID: "recovery", Stage: "shard", ShardIndex: 1, Status: "ASSIGNED", LeaseToken: "old"},
			{TaskID: "final", JobID: "recovery", Stage: "aggregate", Status: "WAITING"},
		}}
	s.restoreJob(job)
	if len(s.taskQueue) != 1 || job.CompletedTasks != 1 || !s.completedTaskIDs["done"] {
		t.Fatal("invalid restored state")
	}
	task, err := s.AssignTask(context.Background(), &pb.AssignTaskRequest{WorkerId: "new"})
	if err != nil || task.TaskId != "lost" || task.LeaseToken == "old" || task.Hyperparameters["_fleet_shard_index"] != "1" {
		t.Fatalf("lost payload or lease: %v %v", task, err)
	}
	_, err = s.ReportTaskCompletion(context.Background(), &pb.TaskCompletionRequest{
		JobId: "recovery", TaskId: "lost", WorkerId: "old", LeaseToken: "old", Success: true})
	if status.Code(err) != codes.FailedPrecondition || s.assigned["lost"] == nil || job.CompletedTasks != 1 {
		t.Fatal("stale worker overwrote current attempt")
	}
}

func TestFullQueueRetainsExpiredLeasePayload(t *testing.T) {
	s := fleetTestServer(t)
	s.taskQueue = make(chan *Task, 1)
	s.taskQueue <- &Task{TaskID: "other"}
	s.leaseTimeout = time.Second
	task := &Task{TaskID: "expired", JobID: "job", Stage: "shard", ShardIndex: 3, Epoch: 4, BatchStart: 500}
	s.jobs["job"] = &Job{JobID: "job", Status: "RUNNING"}
	s.assigned[task.TaskID] = &leaseInfo{Task: task, AssignedAt: time.Now().Add(-time.Minute)}
	s.reapExpiredLeases()
	if len(s.deferredTasks) != 1 || s.deferredTasks[0].ShardIndex != 3 || s.deferredTasks[0].Epoch != 4 || s.deferredTasks[0].BatchStart != 500 {
		t.Fatal("retry lost a task or its training payload")
	}
}

func TestCompletionCannotConsumeAnotherJobsLease(t *testing.T) {
	s := fleetTestServer(t)
	s.jobs["a"] = &Job{JobID: "a", ModelType: "fleet_random_forest", Status: "RUNNING"}
	s.jobs["b"] = &Job{JobID: "b", ModelType: "fleet_random_forest", Status: "RUNNING"}
	s.assigned["task"] = &leaseInfo{WorkerID: "worker", Task: &Task{TaskID: "task", JobID: "a", LeaseToken: "token"}}
	_, err := s.ReportTaskCompletion(context.Background(), &pb.TaskCompletionRequest{JobId: "b", TaskId: "task", WorkerId: "worker", LeaseToken: "token", Success: true})
	if status.Code(err) != codes.FailedPrecondition || s.assigned["task"] == nil {
		t.Fatalf("cross-job completion consumed another lease: %v", err)
	}
}

func TestDurableAdmissionFailureDoesNotEnqueueJob(t *testing.T) {
	s := fleetTestServer(t)
	s.allowEphemeral = false
	_, err := s.CreateTrainingJob(context.Background(), &pb.TrainingJobRequest{JobId: "unpersisted", ModelType: "fleet_random_forest", DatasetPath: "datasets/iris.csv", Epochs: 1, Hyperparameters: map[string]string{"target_column": "target"}})
	if status.Code(err) != codes.Unavailable || len(s.taskQueue) != 0 || s.jobs["unpersisted"] != nil {
		t.Fatalf("failed persistence admitted orphan work: %v", err)
	}
}
