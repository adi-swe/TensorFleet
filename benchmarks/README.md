# TensorFleet experiment tools

Use the real Docker deployment described in [result/README.md](../result/README.md).
Raw results live under `result/`; only a saved `passed` status establishes a
completed experiment. The earlier advertised historical JSON bundle was absent
from the audited checkout and is not a validated baseline.

## Real training and baselines

- `run_mnist_fleet.py`: independent random-forest jobs on 50,000 training,
  10,000 validation, and the separate 10,000-image official test partition.
  Downloads, hashes, reloads, and independently evaluates every model.
- `research_suite.py run`: repeated, randomized tree-parallel forest trials
  across MNIST, Iris, and Wine at matched 1/2/4-worker configurations. Includes
  direct process-pool baselines with the same aggregate CPU/memory budgets,
  exact tree seeds, data partitions, and model parameters. Retains all failures,
  startup outliers, per-trial manifests, dispersion, and bootstrap intervals.
- `validate_mps.py --trials 5`: native Mac worker running actual MPS and CPU
  logistic regression through the gateway, with randomized matched cases.
  Requires native PyTorch and the already uploaded MNIST partitions.
- `fault_trials.py`: individual real worker/coordinator kill-and-restart
  experiments and API backpressure checks. Run after performance experiments;
  it temporarily changes the lease timeout and restores the deployment.
- `smoke_real_fleet.py`: native correctness check with a storage HTTP fixture
  and explicitly enabled ephemeral admission. Not a deployment benchmark.

From the repository root:

```bash
python3 benchmarks/run_mnist_fleet.py --output result
python3 benchmarks/research_suite.py run --trials 5 --workers 1 2 4
python3 benchmarks/validate_mps.py --trials 5
python3 benchmarks/fault_trials.py
```

`benchmarks/run_benchmarks.sh` is a POSIX wrapper for `research_suite.py run`;
its arguments are the same as the Python command. Keep other training and
load-test workloads idle during timing experiments. Cross-node, CUDA, DL/LLM,
and long-run reliability results require separate experiments; they are not
implied by these tools.

## Control-plane and local trainer tools

`benchmark_scaling.py` measures synthetic sleep-task orchestration only.
It requires positive inputs, checks actual participating workers, separates
successful/failed/cancelled/unfinished jobs, counts actual completed tasks, and
exits unsuccessfully for invalid runs. Run it against an isolated fleet.

`report_scaling.py` rejects old schemas, invalid trials, unfinished workloads,
mismatched workload configurations, missing one-worker baselines, and worker
count mismatches. Its per-trial ratios do not replace repeated-trial statistics.

`benchmark_gateway.py` records attempted and successful request rates separately,
raw latency samples, success-only percentiles, and error counts. For sustained
read tests, use `--duration-seconds 10 --skip-submissions --job-id <existing-id>`.
The load model is closed-loop fixed client concurrency; achieved throughput is
not proof of server saturation. Default POST measurements remain short bursts.

`benchmark_training.py` measures local trainer calls outside the fleet. sklearn
input-training-rows/s and neural actual-fit-examples-processed/s use different
units. Neural metrics record actual epochs/examples and keep test data out of
validation/checkpoint selection. Do not relabel local trainer performance as
end-to-end distributed throughput.

`operational_trials.py` runs the repeated polling/slot grid, sustained read
scenarios and concurrent checksummed downloads against the isolated test stack.
`verify_research.py` runs post-refactor model/validity regressions, Go race checks,
a frontend production build, native integration and Go cross-compilation; it
writes `result/research/regression.json` and local per-check logs. Shared Compose
and bootstrap helpers live in `experiment_utils.py`.
