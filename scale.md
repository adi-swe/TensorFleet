# TensorFleet: research gaps and experimental evidence plan

Repository audit: 16 October 2026. Scope: source code, deployment configuration, benchmark scripts, and Markdown reports. The initial audit ran no services; the implementation and experimental follow-up below now include real Docker and native MPS evidence. PDF/DOCX/video report contents were not inspected. This is a repository evidence audit, not a literature review or a claim of research novelty.

## Main finding

TensorFleet has a simulated distributed task execution path and a separate real ML training path. The checked-in benchmarks measure these separately. They do not establish that TensorFleet accelerates real end-to-end distributed model training.

The most useful next contribution is a reproducible experiment connecting real training execution, fleet scheduling, and verified model artifacts. More benchmark charts over simulated tasks would not close this gap.

## Follow-up status on `feat/real-fleet-training`

The numbered findings below preserve the original audit. Current evidence and
limitations are maintained in [research results](result/research/README.md),
with raw artifacts under `result/` and application flow in `docs/FLOW.md`.

1. **Real execution implemented and tested for classical ML.** Independent
   sklearn fits and one tree-parallel distributed forest run through the actual
   gateway, scheduler, workers and storage. The two-worker forest matches all
   10,000 sequential MNIST test probabilities. This does not implement
   distributed neural-network or LLM training.
2. **Reproducible evidence added; unsupported claims withdrawn.** Runnable
   experiment entry points, raw requests/states/manifests, source hashes,
   official MNIST provenance and negative results replace historical synthetic
   speedup/uptime claims in current reports.
3. **Accounting corrected.** Failed/cancelled/unfinished jobs cannot count as
   successful throughput. Actual task completions and participating workers
   are checked; invalid reports and zero/negative benchmark inputs fail.
4. **Test leakage corrected.** Preprocessing fits training data; torch uses
   internal validation for checkpoint selection, then evaluates held-out test
   once. Regression checks change test labels without changing fitted weights.
   MNIST uses fixed 50k/10k/10k train/validation/official-test partitions.
5. **Real scaling evidence added, scope limited.** 90 valid randomized trials
   span three datasets, five seeds and 1/2/4 workers against direct process-pool
   execution. Two-worker MNIST was modestly faster than one; four was slower.
   Cross-host scaling and larger-than-memory data remain untested.
6. **Recovery/fencing implemented and fault-tested.** Redis snapshots restore
   unfinished tasks, original payloads survive retry, fresh tokens fence stale
   completion, and completed IDs prevent duplicate accounting. Worker and
   coordinator kill trials verify saved models. Redis loss, long-run checkpoint
   resume and exactly-once publication remain open.
7. **Baseline and configuration ablation added.** Matched forest budgets and
   per-tree seeds support direct comparisons. A separate repeated polling/slot
   grid tests configuration sensitivity. Direct outer timing includes container
   startup; the separately recorded inner time must be considered. No native
   distributed-DL baseline has been measured.
8. **Repeated, multi-dataset evidence added.** Five randomized trials per
   configuration, raw outliers, sample deviations and bootstrap intervals are
   retained. Evidence remains one machine; independent replication is open.
9. **Metric boundaries corrected.** End-to-end verified-artifact duration,
   fit duration, attempted/successful request rates, and actual neural examples
   processed have distinct names/units. Unknown resource telemetry is shown as
   unavailable. Queue/I/O decomposition and energy measurement remain open.
10. **Operational checks expanded.** Sustained gateway reads, concurrent
    checksummed artifact downloads, admission backpressure, storage failures,
    MinIO restart and worker/coordinator interruption have runnable checks.
    These do not establish uptime, saturation limits, constant server memory,
    or reliability across repeated multi-node faults.

The feasible classical-ML implementation and evidence gaps have been addressed;
hardware-dependent and broader research questions remain explicitly open.
“Fixed” means the listed behavior has evidence, not that all future research
claims or production reliability guarantees have been established.

## 1. Critical: real training is not demonstrated through the distributed fleet

Evidence:

- `worker/main.go`, `simulateTraining`: sleeps for a random 1–4 seconds and generates loss/accuracy from formulas. `GetWorkerStatus` returns random CPU and memory usage. Successful task responses contain empty model weights.
- `orchestrator/cmd/orchestrator/main.go`, `CreateTrainingJob`: creates ten synthetic batches per epoch. `ReportTaskCompletion` records the latest reported loss/accuracy; this is not model aggregation.
- `benchmarks/benchmark_scaling.py`: submits `model_type=benchmark` and a dataset path, then measures completion of those tasks.
- `benchmarks/benchmark_training.py`: directly imports local trainer functions. It bypasses the gateway, orchestrator, worker fleet, dataset transfer, and model persistence.
- `docker-compose.yml`: deploys the Go worker fleet alongside a separate `worker-ml` HTTP service. The latter uses a local thread pool in `worker-ml/api_server.py`.

Research question: Does TensorFleet improve the completion time of real ML workloads, and at what orchestration and storage cost?

Required experiment:

1. First distinguish independent training jobs (for example a hyperparameter sweep) from multiple workers jointly training one model.
2. For the smaller implementation step, schedule a fixed set of real, independent training jobs through TensorFleet. Compare sequential execution, a direct local process-pool baseline, and TensorFleet with 1/2/4/8 workers under matched CPU/thread budgets.
3. Keep models, datasets, seeds, training budgets, and completed outputs identical. Verify saved models by loading them and running held-out predictions.
4. Measure batch makespan, successful jobs/hour, per-job queue delay, training duration, dataset/artifact I/O, and total submission-to-verified-artifact duration.

Claim enabled: distributed ML job orchestration throughput. This does not establish acceleration of a single model.

To support the stronger single-model training claim, implement and evaluate shared training semantics: data partitioning, synchronized gradients or a documented alternative, optimizer/checkpoint state, and convergence. Compare against direct single-worker training and the same framework's native distributed execution. Report time to a predefined held-out quality target, not just time per epoch.

## 2. Critical: published numerical results lack the advertised evidence bundle

Evidence:

- `docs/PERFORMANCE_EVALUATION.md` reports 6.84× scaling speedup, gateway throughput, and model accuracy, and states that raw JSON is committed under `benchmarks/results/`.
- Neither that result directory nor its JSON files is present in the scanned checkout/tracked benchmark inventory.
- `benchmarks/README.md`, the evaluation document, and the root README reference `benchmarks/run_benchmarks.sh`; that runner is absent.
- `docs/PROJECT_REPORT.md`, section 8, additionally claims linear improvement to 50 workers, 1,000+ concurrent jobs, 3–10× faster training, 85% utilization, sub-30-second fault recovery, and 99.5% availability. The inspected benchmark scripts do not substantiate those claims.

These are unverified claims in this checkout; missing evidence alone does not prove the numbers were fabricated.

Required evidence: a runnable versioned harness, original per-run measurements, exact commands, commit SHA, dependency versions, dataset hashes, seeds, topology, resource limits, worker settings, timestamps, and artifact identifiers/checksums. Label historical numbers as unverified until backed by recoverable raw results or reruns. Separate observed measurements from estimates and future design goals.

## 3. Critical: benchmark success accounting can report misleading throughput

Evidence:

- `benchmarks/benchmark_scaling.py` removes `COMPLETED`, `FAILED`, and `CANCELLED` jobs from the pending set, then counts all removed jobs as `completed_jobs`.
- It divides the planned task count by elapsed time even after timeout or unsuccessful terminal states.
- `--workers` is a result label/request parameter; the script does not provision or verify that number of active workers.
- `benchmarks/report_scaling.py` accepts input runs without checking timeouts, successful completion, or matching workloads/configurations.
- Completion polling every two seconds introduces timing uncertainty, especially for short runs.

Required correction before drawing conclusions: separately count successful, failed, cancelled, and unfinished jobs; measure actually successful tasks; fail or clearly exclude invalid runs; verify participating workers and concurrency; validate matched workloads and resource budgets. Retain unsuccessful trials in the evidence bundle and report their failure rate rather than silently discarding them.

Validation experiment: deliberately fail/cancel jobs, leave a run unfinished, and supply a worker-count mismatch. Confirm each is classified correctly and cannot enter a successful scaling summary.

## 4. Critical: model selection reuses the test set

Evidence: `worker-ml/pytorch_trainer.py`, `train_pytorch_model`, evaluates `test_loader` every epoch, chooses `best_state` by that loss, restores it, and then reports accuracy on the same loader. This selection happens even with early stopping disabled.

Implication: the reported test accuracy is not an untouched estimate of generalization. The documented 92.1% MLP result cannot serve as strong comparative quality evidence without a clean evaluation protocol and raw runs.

Required experiment: separate train/validation/test partitions; select checkpoints and hyperparameters using validation only; evaluate the final selected model on test once. Use fixed stratified partitions for classification, multiple training seeds, and report uncertainty. Fit learned preprocessing only on training data. Include an appropriate simple baseline and matched tuning budgets.

## 5. High: scaling results isolate sleeping tasks, not compute scalability

Evidence: the evaluation's scaling workload consists of random sleeps on a single host. Sleeping tasks release CPU capacity; adding workers can improve concurrency without demonstrating proportional compute capacity. The documented configuration also allows two tasks per worker.

Research question: How does scaling change when work is CPU-bound, accelerator-bound, I/O-bound, or too short to amortize scheduling?

Required experiment: fixed-work strong scaling with real training at 1/2/4/8 workers, then a separate weak-scaling experiment with work proportional to resources. Record physical cores/GPUs, process counts, threads per trainer, concurrency per worker, peak memory, actual CPU/GPU utilization, and oversubscription. Repeat on genuinely separate nodes before making cross-node claims. Keep synthetic sleeps as a clearly labelled control-plane experiment.

Report speedup T1/Tn and efficiency (T1/Tn)/n alongside resource consumption. For a fixed single-host CPU budget, interpret worker-count changes as concurrency tuning rather than adding compute resources.

## 6. High: failure recovery and durable execution are unmeasured

Evidence:

- `orchestrator/cmd/orchestrator/main.go` holds the queue, leases, and completion deduplication map in memory. Redis stores serialized job records; loading a job record does not reconstruct the pending queue or in-flight leases.
- Lease timeout defaults to five minutes, with a reaper every 30 seconds. These defaults do not support a general sub-30-second worker recovery claim.
- The reaper logs a dropped requeue when the task queue is full. The failed-attempt requeue reconstructs a task without copying its original epoch/batch fields.
- Completion deduplication is by task ID in memory. It does not itself prove exactly-once artifact effects or persistence across restart.
- `docs/DISTRIBUTED_TRAINING_OVERVIEW.md` describes heartbeat renewal, checkpoint resume, synchronization, and Redis lease patterns. Those descriptions should be treated as design intent unless traced to the active execution path.

Research question: Under which failures does TensorFleet finish correct work without losing tasks or publishing conflicting artifacts?

Required experiments: kill an active worker; allow a slow task to exceed its lease; send late/duplicate completion reports; saturate the queue during retry; restart the orchestrator with Redis available; interrupt Redis or artifact storage. Compare each with a no-fault run.

Measure detection delay, reassignment delay, recovery-to-useful-work, total makespan penalty, lost/repeated computation, missing tasks, duplicate artifacts, and final prediction equivalence. Verify retry payloads preserve the original task. Separate worker retry, coordinator restart recovery, and checkpoint resume; these are different guarantees. Checkpoint recovery requires a real checkpoint/restore path before it can be measured.

## 7. High: no controlled baseline or ablation establishes the source of benefit

Evidence: the scaling reporter compares fleet sizes, but no checked-in experiment isolates scheduler policy, polling interval, serialization, or persistence overhead. The evaluation attributes scaling losses to pickup latency, tail effects, and serialized assignment without controlled measurements of their relative contributions.

Required experiment: compare TensorFleet with direct execution under the same resources; sweep `FETCH_INTERVAL_MS` and `MAX_CONCURRENT_TASKS`; compare Redis healthy versus unavailable; vary task duration and duration skew. Measure scheduler RPC time, idle-slot time, queue wait, tail completion time, throughput, and errors. Test heterogeneous worker speeds and competing jobs to expose stragglers and fairness.

Treat gRPC-versus-REST superiority, or superiority to another platform, as an untested hypothesis unless an equivalent controlled baseline is implemented. A technology choice is not evidence of a research contribution.

## 8. High: statistical and workload coverage is too narrow

Evidence: scripts execute one trial per configuration/algorithm, largely with seed 42; the training benchmark uses one synthetic 5,000 × 16 three-class dataset. There is no repeated-trial aggregation or confidence interval machinery.

Required protocol: start with at least five independent trials per configuration, randomize configuration order, document warm-up, and increase trials when variability remains material. Report trial count, individual values, mean/median, dispersion, and a stated confidence-interval method. Preserve failures and outliers with explanations.

Include real datasets of materially different sizes and characteristics, plus synthetic controls. Use representative input modalities when evaluating CNN/LSTM/Transformer architectures; reshaping tabular features does not establish performance on image, sequence, or language workloads. Report quality and runtime separately; heterogeneous algorithms have different optimization budgets.

## 9. Medium: training throughput definitions are not comparable

Evidence: `benchmarks/benchmark_training.py` reports sklearn throughput as training rows/elapsed time, but PyTorch throughput as training rows × requested epochs/elapsed time. The timed calls include evaluation and other work, and requested epochs may exceed executed epochs if early stopping is enabled.

Required experiment: record actual examples processed and epochs completed, distinguish fit-only time from complete trainer-call time, and separately report end-to-end service time. Document synchronization and timing boundaries for accelerator work. Do not interpret the current cross-framework samples/s column as equivalent work per second.

## 10. Medium: gateway and artifact experiments omit operational conditions

Evidence:

- `benchmarks/benchmark_gateway.py` uses short request-count bursts, one reachability warm-up, and aggregate latency summaries. Errors contribute to attempted request throughput. This is insufficient to establish sustained throughput or long-run availability.
- The evaluation measures a Redis-unavailable configuration and predicts improved performance with Redis available; that direction and magnitude need measurement.
- No checked-in benchmark measures concurrent artifact transfer, checkpoint overhead, or successful reload of a model saved by a fleet job.
- The Go orchestrator's auto-save request contains job metadata; its success alone is not proof of a valid trained model.

Required experiments: sustained load with concurrency/arrival-rate sweeps and warm-up; Redis healthy/unavailable comparisons; queue saturation and recovery; concurrent dataset reads/model uploads over realistic file sizes. Record successful requests/s separately from attempts/s, latency by endpoint/status, errors/timeouts, memory, bandwidth, and artifact checksums. Do not extrapolate a short zero-error burst into uptime or production reliability.

## Recommended order of work

1. **Evidence integrity:** recover or regenerate raw results; restore the runner; correct success accounting and train/validation/test separation before publishing further numbers.
2. **Real workload proof:** connect a fixed independent-training workload to the fleet and verify final artifacts. Establish sequential and direct process-pool baselines.
3. **Scaling and overhead:** repeat matched-resource 1/2/4/8-worker runs, then multi-node runs; decompose compute, queue, network, and storage time.
4. **Failure correctness:** execute worker, lease, saturation, coordinator, Redis, and storage fault cases with output verification.
5. **Mechanism and generality:** polling/concurrency ablations, heterogeneous workers, mixed workloads, and additional real datasets.
6. **Stronger training claim, if required:** implement genuine joint training and compare time-to-quality with native framework baselines. This is additional implementation work, not a missing chart.

## Minimum evidence package for a defensible TensorFleet result

- A precise hypothesis and claim boundary: task scheduling, independent training throughput, or joint model training.
- A versioned command that reproduces the complete workload and summary from raw data.
- Run manifests containing code/dependency versions, resource allocation, configuration, seeds, and dataset hashes.
- Per-job and per-attempt timestamps/statuses; actual participating worker IDs; resource observations from real telemetry.
- Verified artifacts, checksums, held-out predictions, and the documented quality evaluation protocol.
- Repeated baseline/treatment trials with uncertainty and all unsuccessful runs retained.
- Plots of makespan/throughput versus resources, end-to-end time decomposition, and fault recovery behavior, each derived from saved measurements.

Current defensible conclusion: the repository contains a prototype orchestration system and real local ML training implementations. Its documentation reports performance measurements, but the inspected checkout lacks the raw evidence needed to validate them. Genuine TensorFleet experimental evidence must connect correct real work to the deployed platform and reproducible measurements.
