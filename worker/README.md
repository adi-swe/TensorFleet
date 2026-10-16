# ⚡ Worker Service

Real classical training, distributed forest aggregation and optional native
MPS/CUDA logistic execution are available through the explicit
`fleet_*` job types. See [Real fleet training](../docs/REAL_FLEET_TRAINING.md)
for submission, artifact verification, and current limitations. The simulation
described below remains the legacy path for other job types.

The Worker service is the **high-performance computational engine** of TensorFleet, executing **distributed ML training tasks** with advanced neural network simulation, **real-time metrics reporting**, **auto-scaling capabilities**, and seamless **gRPC integration** with the orchestrator's intelligent scheduling system.

## 🚀 Overview

Workers are the **elastic compute nodes** that perform intensive machine learning training workloads in a distributed fashion. Each worker **auto-registers** with the orchestrator, receives **algorithm-specific training tasks**, executes **realistic ML simulations** with convergence patterns, and provides **comprehensive telemetry** back to the monitoring systems for performance optimization and auto-scaling decisions.

## 🏗️ Architecture

```
┌────────────────┐      gRPC       ┌─────────────────┐
│  Orchestrator  │◄────────────────►│     Worker      │
│   (Server)     │   Task Request   │   (Client)      │
└────────────────┘   Task Results   └─────────────────┘
                                            │
                                            ▼ Metrics
                                    ┌───────────────────┐
                                    │   Prometheus      │
                                    │   (Port 2112)     │
                                    └───────────────────┘
```

## 🔌 Key Features

- **Distributed Task Execution**: Processes ML training tasks assigned by orchestrator
- **Realistic ML Simulation**: Simulates neural network training with convergence patterns
- **Real-time Metrics**: Comprehensive Prometheus metrics for monitoring
- **Auto-registration**: Seamlessly connects to orchestrator on startup
- **Health Monitoring**: Regular heartbeats and status reporting
- **Resource Tracking**: CPU/Memory usage monitoring and reporting
- **Fault Tolerance**: Graceful error handling and task retry mechanisms
- **Concurrent Processing**: Handles multiple tasks simultaneously

## 📊 ML Training Simulation

### Realistic Convergence Patterns

The worker simulates different types of ML models with realistic training behavior:

```go
// Training simulation with convergence
func simulateTraining(modelType string, epoch int) (loss, accuracy float64) {
    switch modelType {
    case "neural_network":
        loss = 2.5 * math.Exp(-0.1*float64(epoch)) + rand.Float64()*0.1
        accuracy = 1.0 - loss/2.5
    case "xgboost":
        loss = 1.0 * math.Exp(-0.2*float64(epoch)) + rand.Float64()*0.05
        accuracy = 1.0 - loss/1.5
    case "random_forest":
        loss = 0.8 * math.Exp(-0.15*float64(epoch)) + rand.Float64()*0.03
        accuracy = 0.95 - loss/2.0
    }
    return loss, math.Max(0, math.Min(1, accuracy))
}
```

### Training Metrics

Workers track and report comprehensive training metrics:

- **Loss Values**: Training and validation loss over time
- **Accuracy Metrics**: Model accuracy improvement
- **Training Speed**: Tasks per second, batches processed
- **Resource Usage**: CPU, memory, and GPU utilization
- **Task Progress**: Completion status and timing

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKER_ID` | Unique worker identifier | `auto-generated` |
| `ORCHESTRATOR_ADDRESS` | gRPC orchestrator endpoint | `localhost:50051` |
| `WORKER_PORT` | Worker gRPC server port | `50052` |
| `METRICS_PORT` | Prometheus metrics port | `2112` |
| `HEARTBEAT_INTERVAL` | Heartbeat frequency | `30s` |
| `TASK_TIMEOUT` | Maximum task execution time | `300s` |
| `MAX_CONCURRENT_TASKS` | Concurrent task limit | `3` |
| `LOG_LEVEL` | Logging verbosity | `info` |

### Example Configuration

```bash
export WORKER_ID="worker-$(hostname)-$(date +%s)"
export ORCHESTRATOR_ADDRESS=orchestrator:50051
export WORKER_PORT=50052
export METRICS_PORT=2112
export HEARTBEAT_INTERVAL=30s
export MAX_CONCURRENT_TASKS=3
```

## 🚀 Running the Service

### Using Docker (Recommended)

```bash
# Build and run with docker-compose
docker-compose up --scale worker=3

# Or build separately
docker build -t tensorfleet-worker .
docker run -p 2112:2112 \
  -e ORCHESTRATOR_ADDRESS=orchestrator:50051 \
  -e WORKER_ID=worker-1 \
  tensorfleet-worker
```

### Local Development

```bash
# Install dependencies
go mod tidy

# Generate protobuf files (if needed)
protoc --go_out=. --go-grpc_out=. ../proto/*.proto

# Run the service
go run .
```

## 🔄 Task Execution Flow

### Registration & Task Reception

```go
// Worker registration with orchestrator
func (w *Worker) registerWithOrchestrator() error {
    resp, err := w.orchestratorClient.RegisterWorker(context.Background(), &orchestratorpb.RegisterWorkerRequest{
        WorkerId: w.workerID,
        Capabilities: &orchestratorpb.WorkerCapabilities{
            CpuCores:    4,
            MemoryGb:    8,
            GpuCount:    0,
            MaxTasks:    3,
        },
    })
    return err
}
```

### Task Processing Loop

```go
// Main task processing loop
func (w *Worker) processTasksLoop() {
    for {
        // Request task from orchestrator
        task, err := w.requestTask()
        if err != nil {
            time.Sleep(5 * time.Second)
            continue
        }
        
        // Process task concurrently
        go w.executeTask(task)
    }
}
```

## 📊 ML Training Simulation Engine

### Realistic Convergence Patterns

```go
func simulateModelTraining(modelType string, taskID string, epochs int) {
    for epoch := 0; epoch < epochs; epoch++ {
        // Model-specific convergence behavior
        loss, accuracy := computeTrainingMetrics(modelType, epoch)
        
        // Report progress to orchestrator
        reportProgress(taskID, epoch, loss, accuracy)
        
        // Simulate training time
        time.Sleep(time.Duration(200+rand.Intn(300)) * time.Millisecond)
    }
}

func computeTrainingMetrics(modelType string, epoch int) (float64, float64) {
    switch modelType {
    case "xgboost":
        loss := 2.0 * math.Exp(-0.3*float64(epoch)) + rand.Float64()*0.05
        accuracy := 1.0 - loss/2.5 + rand.Float64()*0.02
    case "neural_network":
        loss := 3.0 * math.Exp(-0.1*float64(epoch)) + rand.Float64()*0.1
        accuracy := 0.9 * (1.0 - math.Exp(-0.2*float64(epoch)))
    case "random_forest":
        loss := 1.5 * math.Exp(-0.2*float64(epoch)) + rand.Float64()*0.03
        accuracy := 0.95 - loss/2.0
    }
    
    return math.Max(0.01, loss), math.Max(0.1, math.Min(0.99, accuracy))
}
```

## 📈 Prometheus Metrics Collection

### Core Training Metrics

```go
var (
    taskDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "worker_task_duration_seconds",
            Help: "Time spent executing tasks",
        },
        []string{"model_type", "job_id"},
    )
    
    tasksCompleted = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "worker_tasks_completed_total",
            Help: "Total completed tasks",
        },
        []string{"status", "model_type"},
    )
    
    currentLoss = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "worker_training_loss",
            Help: "Current training loss value",
        },
        []string{"job_id", "task_id"},
    )
)
```

### Real-time Resource Monitoring

```bash
# Access live metrics
curl http://localhost:2112/metrics | grep worker_

# Example metrics output
worker_task_duration_seconds_bucket{model_type="xgboost",job_id="013dcf7d",le="1"} 12
worker_tasks_completed_total{status="completed",model_type="xgboost"} 15
worker_training_loss{job_id="013dcf7d",task_id="eac041ff"} 0.234
worker_cpu_usage_percent 67.2
worker_memory_usage_bytes 2.1e+09
worker_concurrent_tasks 2
```

## 🏥 Health Monitoring & Diagnostics

### Comprehensive Health Checks

```go
type WorkerHealth struct {
    WorkerID         string    `json:"worker_id"`
    Status           string    `json:"status"`
    Uptime          time.Duration `json:"uptime"`
    ActiveTasks      int       `json:"active_tasks"`
    CompletedTasks   int       `json:"completed_tasks"`
    CPUUsage        float64   `json:"cpu_usage"`
    MemoryUsage     int64     `json:"memory_usage"`
    LastHeartbeat   time.Time `json:"last_heartbeat"`
    OrchestratorConn string   `json:"orchestrator_connection"`
}
```

### Heartbeat & Status Reporting

```bash
# Worker status endpoint
curl http://localhost:2112/health

{
  "worker_id": "587dff17-d318-4109-b103-5de47612ef7a",
  "status": "BUSY",
  "uptime": "2h15m30s",
  "active_tasks": 2,
  "completed_tasks": 47,
  "cpu_usage": 78.5,
  "memory_usage": 2147483648,
  "last_heartbeat": "2025-12-08T21:45:30Z",
  "orchestrator_connection": "connected"
}
```

## 🐛 Debugging & Troubleshooting

### Common Issues & Solutions

1. **Connection to Orchestrator Failed**
   ```bash
   # Check orchestrator connectivity
   telnet orchestrator 50051
   
   # Check environment variables
   echo $ORCHESTRATOR_ADDRESS
   
   # View worker logs
   docker logs tensorfleet-worker-1
   ```

2. **Tasks Not Being Assigned**
   ```bash
   # Check worker registration status
   curl http://localhost:2112/health
   
   # Verify orchestrator sees worker
   curl http://api-gateway:8080/worker-activity
   ```

3. **High Memory Usage**
   ```bash
   # Monitor resource usage
   docker stats tensorfleet-worker-1
   
   # Adjust concurrent task limit
   export MAX_CONCURRENT_TASKS=2
   ```

## 🧪 Testing & Validation

### Unit Testing

```bash
# Run all tests
go test ./...

# Test with coverage
go test -cover ./... -coverprofile=coverage.out
go tool cover -html=coverage.out

# Race condition testing
go test -race ./...
```

### Integration Testing

```bash
# Test worker-orchestrator integration
./scripts/test-worker-integration.sh

# Load testing with multiple workers
docker-compose up --scale worker=5
./scripts/load-test-distributed.sh

# gRPC endpoint testing
grpcurl -plaintext localhost:50052 worker.WorkerService/GetStatus
```

### Performance Benchmarking

```bash
# Benchmark task execution
go test -bench=BenchmarkTaskExecution -benchmem

# Concurrent task performance  
go test -bench=BenchmarkConcurrentTasks -cpu=1,2,4,8
```

## � Dependencies & Requirements

### Core Dependencies

- **Go 1.21+**: Runtime environment
- **gRPC**: Communication protocol
- **Prometheus Client**: Metrics collection
- **UUID**: Unique identifier generation
- **Protobuf**: Message serialization

### Runtime Requirements

- **CPU**: 1+ cores (2+ recommended)
- **Memory**: 512MB minimum (2GB recommended) 
- **Network**: gRPC connectivity to orchestrator
- **Storage**: Minimal (logs and temporary files)

## 🔄 Related Services

- [Orchestrator](../orchestrator/README.md) - Task coordination and scheduling
- [API Gateway](../api-gateway/README.md) - REST API interface
- [Monitoring](../monitoring/README.md) - Metrics aggregation and health
- [Frontend](../frontend/README.md) - Worker visualization dashboard
- **Failure Handling**: Redistributes tasks from failed workers

## 🛠️ Development

### Project Structure

```
worker/
├── main.go              # Main service implementation
├── go.mod               # Go module dependencies  
├── Dockerfile           # Container configuration
└── proto/               # gRPC stubs (generated)
```

### Building

```bash
# Development build
go build -o worker .

# Production build with optimizations
go build -ldflags="-w -s" -o worker .

# Multi-platform builds
GOOS=linux GOARCH=amd64 go build -o worker-linux .
```

## 🐛 Troubleshooting

### Common Issues

1. **Orchestrator Connection Failed**
   ```bash
   # Test connectivity
   telnet orchestrator 50051
   ```

2. **Metrics Not Exposed**
   ```bash
   # Check metrics port
   curl http://localhost:2112/metrics
   ```

3. **Task Execution Timeouts**
   ```bash
   # Check worker logs
   docker logs tensorfleet-worker
   ```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=debug
go run .

# Profile performance
go tool pprof http://localhost:2112/debug/pprof/profile
```

## 🔒 Security Considerations

- **gRPC TLS**: Enable for production
- **Resource Limits**: Prevent resource exhaustion
- **Input Validation**: Sanitize task parameters
- **Sandboxing**: Isolate task execution

## 📄 License

Part of the TensorFleet project - see root LICENSE file.

## 👥 Development Team

**Primary Owner**: Aditya Suryawanshi (25211365) - Backend Infrastructure Lead

This service is part of the TensorFleet distributed ML platform developed by:
- Aditya Suryawanshi (25211365) - Backend Infrastructure Lead (Worker, Orchestrator, API Gateway)
- Rahul Mirashi (25211365) - ML & Data Services Lead
- Soham Maji (25204731) - Frontend & Monitoring Lead

For detailed work distribution, see [docs/TEAM_WORK_DIVISION.md](../docs/TEAM_WORK_DIVISION.md)

---

**Last Updated**: December 21, 2025  
**Version**: 2.0  
**Status**: Research implementation; validated scope and remaining gaps are in `../scale.md`
