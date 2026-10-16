package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"

	workerpb "github.com/tensorfleet/worker/proto/worker"
)

func detectCapabilities() []string {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	python, script := os.Getenv("TRAINING_PYTHON"), os.Getenv("FLEET_TRAINER_PATH")
	if python == "" {
		python = "python3"
	}
	if script == "" {
		script = "fleet_trainer.py"
	}
	output, err := exec.CommandContext(ctx, python, script, "--capabilities").Output()
	var capabilities []string
	if err != nil || json.Unmarshal(output, &capabilities) != nil {
		log.Printf("Could not detect optional trainer capabilities: %v", err)
		return nil
	}
	log.Printf("Trainer capabilities: %v", capabilities)
	return capabilities
}

// One complete independent model per task; no simulated batches or weight averaging.
func (ws *WorkerServer) trainRealModel(ctx context.Context, req *workerpb.TaskRequest) (float64, float64, string, error) {
	// Bound work below the default five-minute lease. Longer jobs require a
	// future lease-renewal protocol, rather than silently running past expiry.
	trainCtx, cancel := context.WithTimeout(ctx, 240*time.Second)
	defer cancel()
	python := os.Getenv("TRAINING_PYTHON")
	if python == "" {
		python = "python3"
	}
	script := os.Getenv("FLEET_TRAINER_PATH")
	if script == "" {
		script = "fleet_trainer.py"
	}
	payload, err := json.Marshal(map[string]interface{}{
		"job_id": req.JobId, "task_id": req.TaskId, "worker_id": ws.workerID,
		"model_type": req.ModelType, "dataset_path": req.DatasetPath,
		"hyperparameters": req.Hyperparameters,
	})
	if err != nil {
		return 0, 0, "", err
	}
	command := exec.CommandContext(trainCtx, python, script)
	command.Stdin = bytes.NewReader(payload)
	command.Env = append(os.Environ(), "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1")
	var stdout, stderr bytes.Buffer
	command.Stdout, command.Stderr = &stdout, &stderr
	// Real work observes cancellation while the Python process is running.
	go func() {
		ticker := time.NewTicker(cancelPollInterval)
		defer ticker.Stop()
		for {
			select {
			case <-trainCtx.Done():
				return
			case <-ticker.C:
				if cancelled, err := ws.isJobCancelled(trainCtx, req.JobId); err == nil && cancelled {
					cancel()
					return
				}
			}
		}
	}()
	if err := command.Run(); err != nil {
		return 0, 0, "", fmt.Errorf("real training failed: %w: %s", err, stderr.String())
	}
	var result struct {
		Status    string `json:"status"`
		Verified  bool   `json:"artifact_verified"`
		ResultURI string `json:"result_uri"`
		Metrics   struct {
			Loss     float64 `json:"test_log_loss"`
			Accuracy float64 `json:"test_accuracy"`
		} `json:"metrics"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return 0, 0, "", fmt.Errorf("invalid trainer response: %w", err)
	}
	if result.Status != "completed" || !result.Verified || result.ResultURI == "" {
		return 0, 0, "", fmt.Errorf("trainer did not verify a saved model")
	}
	return result.Metrics.Loss, result.Metrics.Accuracy, result.ResultURI, nil
}
