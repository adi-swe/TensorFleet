# Code organization and reuse

The training implementation is grouped by responsibility under `worker/training/`:

- `engine.py`: algorithm configuration, train-only preprocessing, forest shard
  and aggregation execution, metrics and verified model publication.
- `data.py`: CSV/NPZ decoding and shape validation.
- `storage.py`: object API paths, upload/download transport and SHA-256 identity.
- `accelerated.py`: optional PyTorch device discovery and portable logistic fit.
- `resources.py`: conservative native worker-count estimates.

`worker/fleet_trainer.py` is the stable JSON stdin/stdout CLI and compatibility
import facade. Go workers, native launchers, benchmark clients and tests reuse
the same engine. `accelerated_logistic.py` and `resource_plan.py` preserve previous
import names without duplicating implementation. Optional torch imports stay
inside accelerator functions so CPU Docker workers need only fleet requirements.
The Docker image copies the complete training package.

`worker/tests/` contains model-integrity and research-validity regressions.
Go scheduler tests remain beside the package they test. Orchestrator helpers
separate distributed scheduling and recovery from its service entry point.

`benchmarks/experiment_utils.py` shares seeded bootstrap intervals and isolated
Compose commands. Experiment CLIs remain separate because scaling, fault
injection, load generation and native GPU training have different lifecycles.
`run_benchmarks.sh` remains a thin convenience entry point; Python CLIs work
without a POSIX shell.

`result/research/` is evidence, not application source. Its `source-bundle/`
contains an intentionally immutable pre-refactor snapshot with SHA-256 manifest;
that snapshot exactly matches source hashes in the 90-trial scaling record.
Module moves therefore do not erase the tested implementation. Historical
40-tree MNIST results stay in `result/`; extended experiments have their own
budgets and manifests under `result/research/`.

Validate changes with:

```sh
python3 -m unittest discover -s worker/tests -p 'test_*.py' -v
(cd orchestrator && go test -race ./...)
(cd worker && go test -race ./...)
(cd api-gateway && go test -race ./...)
python3 benchmarks/smoke_real_fleet.py
(cd frontend && npm run build)
```

Native MPS requires a macOS Python environment with accelerator requirements.
Docker runs CPU workers. Cross-compilation checks portability of Go code only;
it does not demonstrate runtime support on untested hardware or operating systems.

The machine-readable final check record is `result/research/regression.json`.
Run it again with `python3 benchmarks/verify_research.py`. Rebuilt-container
model equivalence and Redis admission checks use
`python3 benchmarks/verify_docker_training.py`. Preserve the original performance
runs when repeating accelerator checks by passing a separate `--output` directory
to `validate_mps.py`.
