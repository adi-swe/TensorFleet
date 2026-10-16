# Real TensorFleet research evidence

These are actual local experiments on the 16 GB M4 MacBook Air, not simulated
training estimates. Docker CPU workers share one host and an approximately 8 GB
Docker VM. Native PyTorch uses the host MPS GPU. No cross-node, NVIDIA CUDA,
distributed DL, or LLM training experiment was performed.

## Repeated model training

`classical-scaling.json` contains 90 valid trials: MNIST, Iris and Wine; 1/2/4
workers; five seeds; TensorFleet and a direct process-pool baseline. Work order
was randomized with seed 2026. Both modes train the same 24-tree, depth-16
forest with identical per-tree seeds and one numerical thread per trainer.
Each trial checks completion, downloads the saved pipeline, verifies SHA-256,
and reloads it to reproduce prediction probes. TensorFleet trials also check
actual participating worker IDs. Forest shards replicate the complete dataset.

MNIST TensorFleet mean end-to-end times were 9.305, 8.430 and 23.789 seconds for
1, 2 and 4 workers respectively. Corresponding direct baseline means were
10.049, 9.227 and 19.521 seconds. Four workers did not improve this workload.
Do not infer the cause from timing alone: shared memory, CPU contention,
serialization, transport and scheduling can all contribute.

Timing starts at submission/direct process launch and ends after downloading,
checking and reloading the artifact. Fleet startup and data upload are excluded.
Direct outer time includes Docker container startup; its inner process-pool time
is recorded separately. This is not an equal-startup-cost orchestration comparison.
A 60.385-second first direct Iris trial is retained, including its 0.836-second
inner time. Raw outliers are never discarded. Group summaries provide medians,
sample standard deviations and seeded 5,000-resample bootstrap mean intervals;
five trials on one machine cannot establish universal scaling behavior.

`distributed-equivalence.json` separately compares a 40-tree, two-worker forest
against the original sequential seed-42 MNIST model: all 10,000 test predictions,
probabilities and global tree seeds match. Test accuracy is 96.51%. The original
three independent 40-tree runs remain in [the initial results](../README.md).
The 24-tree scaling sweep is a distinct experiment and budget.

`accelerated-logistic.json` contains five paired CPU/MPS seeds, with randomized
case order, the same 30 epochs and batch size 2048. Mean native fit time was
3.804 seconds on CPU and 4.731 seconds on MPS. Mean held-out accuracy was 92.108%
and 92.094% respectively. MPS execution works, but was slower for this workload.
These are fit times, not the forest end-to-end timing boundary. The saved model
is portable sklearn coefficients; inference does not require a GPU.

## Recovery and operations

`fault-trials.json` records individual no-fault, worker-kill and coordinator-kill
experiments, including raw job states, requests, fault offsets and verified model
manifests. Fault trials use a 20-second lease and 30-second reaper; production
defaults remain a five-minute lease. They test retry/reconstruction with Redis
surviving, not checkpoint continuation, Redis loss, or measured uptime.

`ablation.json` records a randomized 5-repeat grid of 100/500/1000 ms fetch
intervals and 1/2 concurrent slots per worker. Two physical CPU workers process
four Iris shards plus aggregation. Startup is excluded. This distinguishes
configuration sensitivity from scaling physical worker count. With one slot,
mean verified-artifact times were 1.807/2.359/2.440 seconds for
100/500/1000 ms polling. At 100 ms, two slots averaged 1.851 seconds,
so additional slot concurrency did not help this small workload.

`gateway-c*.json` preserve sustained ten-second reads per endpoint at client
concurrency 1, 4 and 16. They report successful and attempted rates, raw latency
samples, and error counts. These are closed-loop client tests, not proof of
server saturation or long-term availability. `storage-concurrency.json` checks
20 streamed model downloads at concurrency four using the saved SHA-256; it
does not establish a constant-memory server or large-data streaming training.

## Reproduce and inspect

Use the isolated Docker stack and pinned data described in `../README.md`:

```sh
python3 benchmarks/research_suite.py run --trials 5 --workers 1 2 4 --datasets mnist iris wine
python3 benchmarks/validate_mps.py --trials 5
python3 benchmarks/fault_trials.py
python3 benchmarks/operational_trials.py
```

Run timing experiments sequentially with unrelated training idle. Scripts retain
failures and restore the two-worker deployment in `finally` blocks. Operations
and fault tools intentionally interrupt only this test Compose project.

`source-bundle/manifest.json` identifies the exact pre-refactor implementation
used by the scaling experiment. Every recorded scaling source hash matches its
archived file. The current reusable package retains the CLI entry points; see
[code organization](../../docs/CODE_ORGANIZATION.md) and the final regression
check record. Large `.pkl`, `.npz` and service logs remain locally under `result/`
and are ignored by Git; readable JSON evidence is trackable. Dataset provenance,
SHA-256 and exact partitions are in `../mnist-data.json`.

## Research still requiring other resources

Cross-host network scaling, CUDA hardware, framework-native distributed DL/LLM
baselines, long-running checkpoint/lease renewal, out-of-core partitioned data,
full resource/queue/I/O telemetry, repeated fault statistics, and power/energy
measurements remain open. Classical ML on one lightweight machine is the
validated scope. There is no claim that all algorithms use GPUs or that more
workers always improve performance.

## Post-refactor verification

`regression.json` records nine passing checks: 17 Python regressions, Go race
checks for orchestrator/worker/gateway, frontend production build, a race-enabled
native two-worker integration smoke test, native capability discovery, and
Linux/Windows ARM64 Go cross-compilation. Compilation does not validate runtime
behavior on those platforms. CPU and MPS capabilities were detected on this Mac.

The rebuilt Docker stack is checked separately in `post-refactor-docker.json`:
a fresh two-shard 40-tree MNIST forest must reproduce every original held-out
probability, and pausing Redis must reject new durable work with HTTP 503.
`post-refactor-mps/accelerated-logistic.json` keeps separate accelerator regression
runs so they cannot overwrite the original five-trial performance evidence.

All 20 concurrent storage downloads matched the 48,447,835-byte model checksum
in 3.826 seconds total wall time. All nine sustained API scenarios reported zero
errors. These short checks establish the observed behavior only, not an uptime
or storage-memory guarantee.
