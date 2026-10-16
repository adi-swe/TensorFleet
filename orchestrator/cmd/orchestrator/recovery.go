package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
)

// Restore task identities and successful-task deduplication before serving.
// Unfinished work receives new leases. Artifact execution is at least once.
func (s *OrchestratorServer) restoreFromRedis(ctx context.Context) error {
	iterator := s.redisClient.Scan(ctx, 0, "state:job:*", 100).Iterator()
	for iterator.Next(ctx) {
		raw, err := s.redisClient.Get(ctx, iterator.Val()).Bytes()
		if err != nil {
			return err
		}
		var job Job
		if err := json.Unmarshal(raw, &job); err != nil {
			return err
		}
		if job.JobID == "" || len(job.Tasks) == 0 {
			return fmt.Errorf("invalid durable job record: %s", iterator.Val())
		}
		s.restoreJob(&job)
		log.Printf("Recovered job %s (%d/%d tasks)", job.JobID, job.CompletedTasks, job.TotalTasks)
	}
	return iterator.Err()
}

func (s *OrchestratorServer) restoreJob(job *Job) {
	s.jobs[job.JobID] = job
	completed := 0
	for _, task := range job.Tasks {
		if task.Status == "COMPLETED" {
			s.completedTaskIDs[task.TaskID] = true
			completed++
		}
	}
	job.CompletedTasks = completed
	job.TotalTasks = len(job.Tasks)
	if job.Status != "RUNNING" && job.Status != "PENDING" {
		return
	}
	job.Status = "RUNNING"
	if completed == job.TotalTasks {
		job.Status = "COMPLETED"
		return
	}
	for _, task := range job.Tasks {
		if task.Status == "COMPLETED" {
			continue
		}
		if task.Stage == "aggregate" && completed < job.TotalTasks-1 {
			task.Status = "WAITING"
			continue
		}
		task.Status, task.WorkerID, task.LeaseToken = "PENDING", "", ""
		select {
		case s.taskQueue <- task:
		default:
			s.deferredTasks = append(s.deferredTasks, task)
		}
	}
}
