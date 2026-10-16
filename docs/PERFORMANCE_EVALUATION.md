# TensorFleet performance evaluation

The earlier numerical tables in this document lacked the advertised raw results
and runner in the audited checkout. They are **unverified historical claims**,
not an experimental baseline. This evaluation now uses saved, reproducible runs.

## Verified evidence

- [Initial Docker MNIST experiment](../result/README.md): three independent
  random-forest fits, separate 50,000/10,000/10,000 train/validation/official-test
  partitions, saved models, independent reload evaluations, and storage checks.
- [Current research evidence](../result/research/): a single forest trained as
  deterministic tree partitions across workers, native MPS/CPU logistic training,
  and repeated classical-ML scaling with direct process-pool controls.
- [Gap-by-gap implementation and limitations](../scale.md).

A run is evidence only when its saved status is `passed`; incomplete or failed
trials must remain visible. The initial MNIST result supports real independent-job
execution, not a distributed-training speedup. The newer distributed forest is
joint model construction through tree parallelism; each worker loads the full
training dataset. It is not an out-of-core, gradient-synchronized training system.

## Measurement rules

Use actual participating worker IDs and verified models. Report successful,
failed, cancelled, and unfinished jobs separately. Reject invalid scaling runs.
Keep source/data hashes, exact partitions, worker allocations, versions, commands,
raw per-trial values, and confidence-interval methods with every reported result.

The real workload suite varies 1/2/4 CPU workers on one Docker host and compares
TensorFleet with resource-budgeted direct process pools. Configuration order is
randomized; model seeds and tree identities are matched across configurations.
Read individual timings, medians, dispersion, and uncertainty together. Container
startup and model verification are part of the stated end-to-end timing; image
builds, dataset upload, and fleet startup are excluded. First-use filesystem and
container startup outliers are retained, not silently removed.

`benchmark_scaling.py` measures **simulated orchestration tasks**. Its throughput
must never be labelled model-training throughput. `benchmark_training.py` times
complete local trainer calls; its sklearn input-rows/s and neural actual-fit-
examples/s have explicitly different units and are not equivalent amounts of work.

## Limits

The current evidence is from one Apple Silicon host, an approximately 8 GB Docker
VM, native MPS where requested, and the documented workloads. It does not establish
cross-machine scaling, NVIDIA CUDA performance, LLM training, long-run availability,
or superiority to other training platforms. Controlled polling/concurrency and
failure experiments must be interpreted according to their actual saved settings.
No sub-30-second recovery, 50-worker scaling, 99.5% uptime, or constant-memory
artifact-I/O guarantee is established by the original documentation.
