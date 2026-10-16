# Distributed Training Platform — Overview & Transcript

> Historical architecture/design discussion. Examples of gradient synchronization,
> checkpoint resume, Redis leases, and elastic GPU training below are not proof
> that those capabilities are implemented. Use [FLOW.md](FLOW.md) for the active
> runtime and [scale.md](../scale.md) for tested claims and remaining work.

**One-liner**

“I built a distributed training platform that coordinates TensorFlow jobs across multiple nodes using gRPC for control, Redis for orchestration/state, and MinIO for model artifacts.”

---

## Architecture Overview

- **Coordinator Node**: The brain that accepts training requests, partitions data and hyperparameters, schedules tasks to workers, keeps global training state (checkpoints, barrier sync), handles retries and node failures, stores metadata/leases in Redis and artifacts in MinIO.

- **Worker Nodes**: Workers pull tasks, run TensorFlow training loops, upload checkpoints to MinIO, stream logs/metrics, and send heartbeats/status back to the Coordinator.

- **gRPC Communication**: Typed, low-latency RPCs for control plane operations (submit job, acquire task, heartbeat, checkpoint notify). Supports streaming for logs/metrics.

- **Redis Orchestration**: Fast ephemeral coordination (task queues, leases, heartbeats, minimal commit-log). Use SetNX, TTL leases, lists/streams for queues.

- **TensorFlow Training**: Workers run TF training; frequent checkpointing to MinIO for fault tolerance. Use parameter-server or all-reduce depending on the workload and network.

- **MinIO Artifact Storage**: S3-compatible object storage for datasets, checkpoints, and final models. Durable and decouples large binaries from ephemeral coordination.

---

## Project-specific Implementation (TensorFleet repo)

- **Docker services & wiring**: This repo runs Redis, MinIO, Orchestrator, Worker, Storage, and Worker-ML via `docker-compose.yml` (service names and ports): see [docker-compose.yml](docker-compose.yml). Key services:
  - `redis` (container: `tensorfleet-redis`) — default port `6379` used by the Orchestrator and other services.
  - `minio` (container: `tensorfleet-minio`) — MinIO server on ports `9000` (API) and `9001` (console).
  - `orchestrator` (gRPC control plane) — listens on `50051`; configured with `REDIS_HOST`/`REDIS_PORT` environment variables.
  - `worker-ml` (ML training HTTP API) — runs `api_server.py` on port `8000` and exposes training endpoints.
  - `storage` (file/API gateway to MinIO) — provides dataset/model upload endpoints; environment variables point it to MinIO.

- **Orchestrator (gRPC, Redis)**: The orchestrator is implemented in Go (see [orchestrator/README.md](orchestrator/README.md)) and exposes protobuf/gRPC endpoints for job lifecycle and worker coordination. The service expects Redis for job queues and metadata (env vars: `REDIS_HOST`, `REDIS_PORT`, `TASK_TIMEOUT`, `WORKER_TIMEOUT`). See the orchestrator README for the exact proto/gRPC surface and configuration examples.

- **Storage & Artifacts (MinIO + MongoDB)**: The repository contains a `StorageManager` that talks to MinIO and MongoDB. Buckets used by the project include `models`, `datasets`, `checkpoints`, `artifacts`, and `jobs`. See [storage/storage_manager.py](storage/storage_manager.py) for bucket names and MinIO usage.

- **ML Worker (TensorFlow + scikit-learn)**: The ML worker service is in `worker-ml/` and exposes an HTTP API (`worker-ml/api_server.py`). Training code lives in `worker-ml/model_trainer.py` which implements TensorFlow DNN/CNN training (`train_tensorflow_model`) and scikit-learn training for classical models.

- **Storage client integration**: Workers call the `storage` service (HTTP) to download datasets and upload trained models/checkpoints (see [worker-ml/storage_client.py](worker-ml/storage_client.py)). Stored artifacts use S3-style paths (e.g., `s3://checkpoints/...`) and are saved to MinIO by the Storage service.

- **Where to find proto definitions**: gRPC/proto files live under `proto/` (e.g., [proto/orchestrator.proto](proto/orchestrator.proto)). Generated stubs appear in service-specific folders (for example `orchestrator/pb/` and `worker/` generated outputs).

These concrete locations map the earlier high-level concepts to actual files and services in the repository — useful when you want to produce a runnable prototype or point to code during a demo.


## Founder-Focused Engineering Decisions

- **Why gRPC over REST**: lower latency and HTTP/2 multiplexing; protobuf schemas for safe evolution; streaming RPCs for logs and checkpoint streaming.

- **Synchronization Challenges**: Synchronous SGD (deterministic but sensitive to stragglers) vs asynchronous (higher throughput, risk of stale gradients). Use hybrid modes: local sync groups + async global updates.

- **Fault Tolerance & Node Failures**: Short TTL leases in Redis; workers renew leases via heartbeat. Expired leases => coordinator reassigns tasks; resume from latest MinIO checkpoint.

- **Consistency vs Throughput**: Offer `sync` and `async` modes. Hybrid: small synchronized groups for consistency-critical parts, async globally for throughput.

- **Distributed Coordination**: Redis for ephemeral state, MinIO for durable artifacts, Coordinator mediates job-level decisions while heavy data sync happens peer-to-peer (NCCL/all-reduce) when possible.

- **Retry Logic**: Idempotent RPCs, exponential backoff with jitter, capped retries, and escalation to coordinator when necessary.

- **Training State Management**: Two-tiered state: ephemeral in Redis (heartbeats, lease pointers), durable in MinIO (checkpoints) plus checkpoint pointers in Redis for quick resume.

---

## Compact Code Examples

### 1) Protobuf (control API)

```proto
syntax = "proto3";

service Orchestrator {
  rpc SubmitJob(SubmitJobReq) returns (JobRef);
  rpc GetTask(WorkerInfo) returns (Task);
  rpc Heartbeat(HeartbeatMsg) returns (HeartbeatResp);
  rpc ReportCheckpoint(CheckpointMsg) returns (Ack);
  rpc StreamLogs(stream LogLine) returns (Ack);
}

message SubmitJobReq { string job_id = 1; string dataset_uri = 2; int32 workers = 3; }
message JobRef { string job_id = 1; }
message WorkerInfo { string worker_id = 1; string capabilities = 2; }
message Task { string task_id = 1; string checkpoint_uri = 2; string cmd = 3; }
message HeartbeatMsg { string worker_id = 1; string status = 2; int64 ts = 3; }
message CheckpointMsg { string task_id = 1; string minio_uri = 2; int64 step = 3; }
message Ack {}
message LogLine { string task_id = 1; string line = 2; int64 ts = 3; }
```

### 2) Coordinator (Go) — assign task & lease in Redis

```go
func assignTask(redisClient *redis.Client, taskID, workerID string, leaseTTL time.Duration) error {
  leaseKey := fmt.Sprintf("lease:%s", taskID)
  ok, err := redisClient.SetNX(ctx, leaseKey, workerID, leaseTTL).Result()
  if err != nil { return err }
  if !ok { return fmt.Errorf("task leased") }
  // record assignment
  redisClient.HSet(ctx, "task:"+taskID, "worker", workerID, "status", "assigned")
  return nil
}
```

### 3) Worker (Python) — get task via gRPC, run TF, upload checkpoint to MinIO

```python
stub = orchestrator_pb2_grpc.OrchestratorStub(channel)
task = stub.GetTask(orchestrator_pb2.WorkerInfo(worker_id=WORKER_ID))

for step, batch in enumerate(dataset):
    loss = model.train_on_batch(batch)
    if step % CHECKPOINT_STEPS == 0:
        ckpt_path = f"/tmp/{task.task_id}/ckpt-{step}.ckpt"
        model.save_weights(ckpt_path)
        minio_client.fput_object("checkpoints", f"{task.task_id}/{step}.ckpt", ckpt_path)
        stub.ReportCheckpoint(orchestrator_pb2.CheckpointMsg(task_id=task.task_id, minio_uri=f"s3://checkpoints/{task.task_id}/{step}.ckpt", step=step))
```

### 4) Redis queue pattern (Python) — atomic pop for work

```python
# RPOPLPUSH: move from pending -> processing atomically
task = redis.rpoplpush("queue:pending", "queue:processing")
```

### 5) Exponential backoff with jitter

```python
def backoff(attempt, base=0.5, cap=30.0):
    sleep = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, sleep * 0.1)
    time.sleep(sleep + jitter)
```

### 6) Checkpoint pointer in Redis

```python
redis.hset(f"job:{job_id}", mapping={"latest_checkpoint": minio_uri, "step": step})
uri = redis.hget(f"job:{job_id}", "latest_checkpoint")
```

---

## Fault Scenarios & Handling

- **Worker crash mid-step**: Lease TTL expiry -> coordinator reassigns; resume from `job:<id>.latest_checkpoint` in Redis that points to MinIO.
- **Network partition**: Workers retry and keep local progress; coordinator decides reassign after lease TTL; idempotent tasks prevent double-effects.
- **Slow/straggler worker**: Speculative re-execution or backup worker; tune timeouts or use smaller sync groups.
- **Bad/corrupt checkpoint**: Keep multiple recent checkpoints with checksums; fall back to previous valid checkpoint.

---

## Operational Notes

- Observability: stream metrics & logs via gRPC to monitoring backend; track queue length and heartbeat lag.
- Cost tradeoffs: checkpoint frequency vs storage/I/O cost — more frequent checkpoints reduce recovery time.
- Security: use mTLS for gRPC and pre-signed MinIO URLs, enforce ACLs for Redis.

---

## Speaker-Friendly Transcript

Hello — I built a distributed training platform to coordinate TensorFlow jobs across many nodes. At a high level: the Coordinator schedules work, Workers run training and report progress, Redis holds ephemeral coordination data and leases, and MinIO stores durable artifacts like checkpoints and final models. We use gRPC for control plane RPCs and streaming.

Why gRPC? gRPC gives us typed contracts and low-latency streaming. Control messages like heartbeats and checkpoint notifications are frequent — HTTP/1 REST would be inefficient and brittle here. Protobuf lets us evolve schemas safely across releases.

Coordination design: the Coordinator doesn't micro-manage every training step. It assigns tasks, issues leases stored in Redis, and accepts checkpoint notifications. Workers renew leases via heartbeats. If a lease expires, the Coordinator treats the task as orphaned and reassigns it — but we avoid losing progress because workers write incremental checkpoints to MinIO frequently.

Synchronization is the hard part. Synchronous training gives deterministic convergence but is vulnerable to the slowest worker — so for production we support two modes: strict synchronous for critical runs, and async/hybrid for throughput-sensitive jobs. For heavy gradient exchange we prefer peer-to-peer all-reduce (NCCL) among a small synchronized group rather than routing everything through the Coordinator.

Fault tolerance is handled by short leases, checkpointing, and idempotent task semantics. If a worker crashes, the Coordinator reassigns the task and the new worker resumes from the latest checkpoint stored in MinIO. We keep checkpoint pointers in Redis for quick lookups.

For retry logic we implement exponential backoff with jitter for RPCs and requeueing, and we cap retries before escalating to manual or automated remediation.

If you want, I can expand this into slides or produce a runnable minimal prototype (coordinator + worker + proto + docker-compose).

---
