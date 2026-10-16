# Real independent training on the TensorFleet fleet

This is the first implementation step for gap 1 in `scale.md`. A fleet job
trains one complete scikit-learn classifier on one worker. Submit multiple
jobs to distribute independent fits across the existing Go worker fleet.
The original path fits independent models. The newer distributed-forest path
constructs one model by training different trees across workers; it does not
synchronize gradients. See the additional modes below.

## Execution path

1. Submit a `fleet_*` job through the gateway.
2. For basic estimators, the Go orchestrator enqueues one task for that job.
3. An available Go worker pulls it and starts its own Python subprocess.
4. The trainer downloads the specified CSV or NPZ from the storage service. Missing
   data fails the job; there is no synthetic-data or MongoDB fallback.
5. It makes a seeded, stratified train/test split and fits preprocessing only
   on the training partition. The saved pipeline includes preprocessing.
6. It fits the classifier, evaluates the held-out partition, uploads a real
   pickle artifact, downloads it, checks its SHA-256, and verifies that the
   reloaded pipeline produces the same held-out predictions.
7. It uploads and reads back a result manifest. Only then does the worker
   report successful completion and real log loss/accuracy to the orchestrator.

Supported model types: `fleet_random_forest`, `fleet_logistic_regression`,
and `fleet_decision_tree`. Additional modes are `fleet_distributed_random_forest`
and `fleet_accelerated_logistic`. Existing model types still use the legacy simulated
Go-worker path; use the explicit `fleet_` names for real execution.

## Run against Docker Compose

Start the existing stack using its configured infrastructure credentials:

```bash
MAX_CONCURRENT_TASKS=1 FETCH_INTERVAL_MS=100 docker compose up -d --build
```

Each rebuilt Go worker image now contains Python and the pinned fleet trainer
dependencies. It executes training locally inside that worker container;
the separate `worker-ml` HTTP service is not involved. The worker's storage
URL is `http://storage:8081`, configurable through `STORAGE_SERVICE_URL`.

Upload a dataset, then submit a real job:

```bash
curl --fail -X POST http://localhost:8081/api/v1/upload/datasets/iris.csv \
  -F 'file=@data/Iris.csv'

curl --fail -X POST http://localhost:8080/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "fleet_random_forest",
    "dataset_path": "datasets/iris.csv",
    "epochs": 1,
    "hyperparameters": {
      "target_column": "Species",
      "random_state": "42",
      "test_size": "0.2",
      "estimator_params": "{\"n_estimators\":100,\"max_depth\":5}"
    }
  }'
```

`target_column` must match the CSV exactly. Exclude identifier or leakage
columns from research datasets before uploading: the trainer uses every
non-target column as a feature. The checked-in Iris example includes an ID
column and is useful for a functional demo, not a clean research evaluation.
All values in `hyperparameters` are strings because the existing API uses a
string-valued map. `estimator_params` is a JSON object encoded as one string.
Use `random_state` at the top level; `n_jobs` is controlled by the trainer.

`epochs` defaults to 1 for these fleet model types; other values are rejected.
A scikit-learn fit is not partitioned into independent epoch jobs. `num_workers`
does not reserve workers for basic estimator jobs. For distributed forests it
specifies shard count; manifests record actual participating workers. Fleet
size is controlled by deployment replicas; result manifests identify workers
that actually performed the work.

Poll `GET /api/v1/jobs/<job_id>`. On success, `message` contains
`s3://artifacts/fleet/<job_id>/result.json`. Retrieve it through:

```text
GET http://localhost:8081/api/v1/download/artifacts/fleet/<job_id>/result.json
```

The manifest contains the unique model artifact URI, checksum, actual worker
and attempt IDs, dataset checksum, split seed and test indices, fit duration,
quality metrics, package versions, and raw-input/prediction verification rows.
Retrieve `artifact_uri` using the same `/api/v1/download/<bucket>/<object>` API.
These objects are stored directly in MinIO through the existing object API;
they are not registered in the MongoDB model-registry UI in this first step.
The legacy metadata-only auto-save endpoint is skipped for real fleet jobs.

## Native development and correctness checks

```bash
python3 -m pip install -r worker/fleet-requirements.txt
python3 -m unittest discover -s worker/tests -p 'test_*.py' -v
(cd orchestrator && go test -race ./...)
(cd worker && go test ./...)
(cd api-gateway && go test ./...)
python3 benchmarks/smoke_real_fleet.py
```

For a manually started native worker, run `go run .` from `worker/`, set
`TRAINING_PYTHON` to the intended interpreter, and set `STORAGE_SERVICE_URL`
to the reachable storage endpoint. `FLEET_TRAINER_PATH` defaults to
`fleet_trainer.py` relative to the worker's working directory.

The smoke check builds the real Go services into a temporary directory, starts
two worker processes, and submits four real training jobs through the gateway.
It independently verifies model checksums, reload predictions, held-out accuracy,
two actual participating workers, terminal dataset failures, and invalid-request
handling. It uses an in-memory HTTP fixture for the existing storage object API
and Redis-unavailable operation. It does not validate the deployed storage
service, MinIO durability, container builds, or performance. It cleans up its
service processes and temporary files on exit.

## Limits and next experiments

A real Docker/MinIO MNIST experiment now passes: three fixed-seed random-forest
fits use 50,000 training images, 10,000 validation images, and the separate official
10,000-image test set. Models were independently reloaded and all partition metrics
reproduced. See [saved evidence and reproduction](../result/README.md).

For explicit partitions, supply both `validation_dataset_path` and
`test_dataset_path` in the string-valued hyperparameters map. The primary
`dataset_path` is then used entirely for fitting, with no internal split. All
partitions must have matching schemas. For `.npz` inputs, use non-object arrays
`X` with shape `(rows, features)` and `y` with shape `(rows,)`; target-column
names are supplied through the existing required `target_column` parameter.
CSV continues to work. Explicit validation and test data are evaluated without
fitting preprocessing or selecting model parameters on either partition.

- Each trainer uses one numerical-library thread and random forest `n_jobs=1`.
  Set worker concurrency and CPU limits explicitly for comparable trials.
- Real tasks have a 240-second subprocess timeout, below the default five-minute
  lease. Keep the orchestrator lease at least at that default for this path.
  Lease renewal and long-running/checkpointed training are not implemented.
- The worker polls cancellation while training. A training/storage error becomes
  terminal `FAILED`, with the error in the status message, rather than an
  unbounded retry of a permanently invalid dataset or configuration.
- A worker crash can still trigger the existing lease retry. Unique per-attempt
  model paths avoid overwriting model bytes, but manifest publication is not an
  exactly-once transaction. Redis snapshots now restore unfinished work after a coordinator restart.
- CPU/memory telemetry is unavailable: gateway fields are null and the dashboard
  displays “Not measured”; unavailable values must not become resource evidence.
- Datasets and artifacts are buffered in memory. This first path targets small or
  moderate tabular classification workloads, not large streaming datasets.

The repeated 1/2/4-worker experiment and matched process-pool baselines are now
saved in `result/research/classical-scaling.json`. See `result/research/README.md`
for measured results and limits; these are single-host observations.


## One random forest across workers

Submit `model_type=fleet_distributed_random_forest`, `epochs=1`, and
`num_workers=N`. The orchestrator creates N tree-partition tasks plus one
aggregation task. The latter waits until every partition completes. Every
partition uses the same training/preprocessing data, while an advanced seeded
random generator assigns the same global tree seeds regardless of N. The
aggregation worker verifies shard provenance, checksums, preprocessing, and
class compatibility, concatenates the trained trees, and saves one standard
sklearn pipeline.

N is the partition count, not a reservation of physical machines. Deploy enough
worker replicas for concurrent execution and check actual worker IDs in the
manifest. Data is replicated in each worker; more replicas do not make an
otherwise too-large dataset fit in memory. In tests, an uneven 13-tree/3-partition
forest matches sequential probabilities, and the 40-tree MNIST two-worker forest
matches all 10,000 test predictions from the original sequential reference.
Out-of-bag scoring and warm-start construction are rejected for this backend.

## Apple MPS, CUDA selection, and native workers

`fleet_accelerated_logistic` trains a linear softmax classifier with PyTorch and
exports a normal sklearn logistic-regression estimator with learned coefficients.
The saved model supports CPU inference without requiring PyTorch. Use
`device=mps`, `cuda`, `cpu`, or `auto` in hyperparameters; `auto` chooses the best
available device on the assigned worker. Explicit unavailable devices never
silently fall back to CPU. Configure training epochs/batch size/learning rate
inside `estimator_params`; the top-level job `epochs` remains one fit.

Workers advertise detected capabilities through `AssignTask`. Incompatible jobs
wait without blocking other CPU work. Rebuild gateway, orchestrator, and worker
images together after protocol changes. The current Linux Docker workers do not
advertise MPS. On this Mac, launch a native worker to use its one Apple GPU:

```bash
python3 -m pip install -r worker/accelerator-requirements.txt
python3 worker/launch_fleet.py --algorithm fleet_accelerated_logistic \
  --rows 50000 --features 784 --plan-only
python3 worker/launch_fleet.py --algorithm fleet_accelerated_logistic \
  --rows 50000 --features 784
```

The launcher also supports `fleet_distributed_random_forest`, choosing a bounded
number of native CPU worker processes from declared dataset dimensions, available
CPUs, tree count, and a conservative memory estimate. Small datasets are kept to
one worker to avoid startup overhead. Use `--max-workers` and `--memory-gib` to
set budgets. It provisions local processes, not cloud/Kubernetes nodes, and its
memory estimate is not a peak-memory guarantee or a prediction of optimal speed.
MPS jobs use one worker/device; replicas do not create additional GPUs. Native
CUDA selection is implemented for compatible installations but not tested on
this hardware. macOS is runtime-tested; other OS support requires validation.

## Durable state and retries

Durable jobs are persisted before tasks become visible. Redis snapshots use
`state:job:<id>` separately from gateway history. At coordinator startup,
unfinished tasks are requeued with their original identities and payloads;
completed-task deduplication is reconstructed. Every assignment receives a fresh
lease token; stale or wrong-worker completion reports cannot remove a newer lease.
Queue-full retries are retained in a deferred queue rather than discarded.

Redis-unavailable admission fails by default. `ALLOW_EPHEMERAL_JOBS=true` is an
explicit development/test opt-in that cannot promise restart recovery. Durable
records currently expire after 24 hours. Recovery is at least once; unique model
paths preserve attempt artifacts, but final-manifest publication and external
side effects are not an exactly-once transaction. There is no neural optimizer
checkpoint-resume or multi-GPU training implementation in this classical path.

Startup now fails if durable recovery cannot complete, unless ephemeral mode is
explicitly enabled. Durable admission uses a bounded persistence call and returns
503 on Redis unavailability; the post-refactor Docker regression verifies this.
