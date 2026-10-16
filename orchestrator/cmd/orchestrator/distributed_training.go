package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	pb "github.com/tensorfleet/orchestrator/proto/orchestrator"
)

func matchesWorker(job *Job, capabilities []string) bool {
	if job.ModelType != "fleet_accelerated_logistic" {
		if !strings.HasPrefix(job.ModelType, "fleet_") || len(capabilities) == 0 {
			return true
		}
		for _, capability := range capabilities {
			if capability == job.ModelType {
				return true
			}
		}
		return false
	}
	device := job.Hyperparameters["device"]
	for _, capability := range capabilities {
		if (device == "" || device == "auto") && strings.HasPrefix(capability, job.ModelType+":") {
			return true
		}
		if capability == job.ModelType+":"+device {
			return true
		}
	}
	return false
}

func distributedTasks(req *pb.TrainingJobRequest) ([]*Task, error) {
	if req.NumWorkers < 1 || req.NumWorkers > 32 {
		return nil, fmt.Errorf("distributed forest requires num_workers between 1 and 32")
	}
	parameters := map[string]interface{}{}
	if raw := req.Hyperparameters["estimator_params"]; raw != "" {
		if err := json.Unmarshal([]byte(raw), &parameters); err != nil {
			return nil, fmt.Errorf("estimator_params must be a JSON object")
		}
	}
	trees := float64(100)
	if value, exists := parameters["n_estimators"]; exists {
		var ok bool
		trees, ok = value.(float64)
		if !ok || trees != float64(int(trees)) {
			return nil, fmt.Errorf("n_estimators must be an integer")
		}
	}
	if trees < float64(req.NumWorkers) {
		return nil, fmt.Errorf("n_estimators must be at least num_workers")
	}
	tasks := make([]*Task, 0, req.NumWorkers+1)
	for index := int32(0); index < req.NumWorkers; index++ {
		tasks = append(tasks, &Task{TaskID: uuid.New().String(), JobID: req.JobId,
			Stage: "shard", ShardIndex: index, Status: "PENDING", CreatedAt: time.Now()})
	}
	return append(tasks, &Task{TaskID: uuid.New().String(), JobID: req.JobId,
		Stage: "aggregate", Status: "WAITING", CreatedAt: time.Now()}), nil
}

// Lock producers before state, matching admission's lock order.
func (s *OrchestratorServer) enqueueAggregation(jobID string) {
	s.enqueueMu.Lock()
	defer s.enqueueMu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	job := s.jobs[jobID]
	if job == nil || job.Status != "RUNNING" {
		return
	}
	last := job.Tasks[len(job.Tasks)-1]
	if last.Stage != "aggregate" || last.Status != "WAITING" {
		return
	}
	select {
	case s.taskQueue <- last:
		last.Status = "PENDING"
	default:
		s.deferredTasks = append(s.deferredTasks, last)
		last.Status = "PENDING"
	}
}

// Holds s.mu. Incompatible jobs wait here without being dropped or blocking
// CPU jobs behind an unavailable accelerator.
func (s *OrchestratorServer) takeDeferred(capabilities []string) *Task {
	for index := 0; index < len(s.deferredTasks); {
		task := s.deferredTasks[index]
		job := s.jobs[task.JobID]
		if job == nil || job.Status != "RUNNING" || matchesWorker(job, capabilities) {
			s.deferredTasks = append(s.deferredTasks[:index], s.deferredTasks[index+1:]...)
			if job != nil && job.Status == "RUNNING" {
				return task
			}
			continue
		}
		index++
	}
	return nil
}
