# MNIST Docker experiment results

Completed 16 October 2026 on `feat/real-fleet-training`.

**PASS:** Three real MNIST random-forest jobs ran through the gateway, orchestrator, two Docker workers, storage service, and MinIO. Independent local reloads reproduced training, validation, and official-test metrics for every saved model.

## Dataset and protocol

Used 50,000 images for fitting and 10,000 for validation, stratified from the official 60,000-image training partition with seed 2026. The official 10,000-image test partition was kept separate. Each 28 × 28 image became 784 pixel features. Preprocessing was fitted on training data only.

Fixed random forest: 40 trees, maximum depth 20, `max_features=sqrt`; model seeds 42, 43, 44. No model or parameter was selected using the test set. Each worker had one CPU, 2 GiB memory, one task slot, and one numerical-library thread.

The source archive SHA-256 was checked against the [Keras MNIST loader](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py). Source URL, full checksum, exact train/validation indices, uploaded dataset hashes, and class counts are in [mnist-data.json](mnist-data.json). The plan was written before observing the MNIST job results: [PLAN.md](PLAN.md).

## Measured quality

- Seed 42: train **99.942%**, validation **96.720%**, test **96.510%**; fit 11.058 s.
- Seed 43: train **99.930%**, validation **96.610%**, test **96.330%**; fit 10.852 s.
- Seed 44: train **99.928%**, validation **96.460%**, test **96.260%**; fit 8.173 s.

Training: mean **99.933%**, sample standard deviation **0.008 percentage points** across the three seeds.

Validation: mean **96.597%**, sample standard deviation **0.131 percentage points** across the three seeds.

Official test: mean **96.367%**, sample standard deviation **0.129 percentage points** across the three seeds.

## Timing and participation

Two distinct worker IDs performed the three fits. From immediately before first submission to observing all three terminal job statuses: **25.184 s**. From that same start to completing the client-side download and independent evaluation of all models: **29.034 s**. These durations exclude source download, split preparation, dataset upload, image builds, and service startup. Status was polled every 0.5 s.

Mean trainer fit duration was **10.028 s**. This fit-only value excludes training/validation/test evaluation, artifact storage, reload verification, Python startup, and queue delay.

Raw requests, job IDs, terminal statuses, worker IDs, per-run metrics, exact Docker image IDs, worker allocations, source-file hashes, and timestamps are in [mnist-run.json](mnist-run.json). [resource-snapshot.json](resource-snapshot.json) is one real instantaneous Docker stats observation; it is not an average or a peak measurement.

## Verification and failure checks

Saved artifacts were downloaded through the real storage API, checked by SHA-256, reloaded as complete preprocessing/classifier pipelines, and independently evaluated on all 50,000 training, 10,000 validation, and 10,000 test examples. Accuracy matched each worker manifest exactly, and validation/test log loss also matched. The gateway-reported test accuracy matched the independent test evaluation.

[deployment-checks.json](deployment-checks.json) records passing checks for unsupported algorithms, invalid epoch counts, missing target columns, missing datasets, and a deliberately stopped storage service. Failed jobs reported zero completed tasks and did not publish success manifests. Storage was restored afterwards.

After restarting the real MinIO container, all three dataset objects and all three model objects retained their expected checksums; all three result manifests were unchanged. This establishes persistence across that container restart, not arbitrary crash recovery or exactly-once execution.

## Saved artifacts

- Seed 42: [manifest](20d5a59d-9236-4293-bd13-d469df855062-manifest.json), [independent evaluation](20d5a59d-9236-4293-bd13-d469df855062-evaluation.json), model `20d5a59d-9236-4293-bd13-d469df855062-model.pkl`, predictions `20d5a59d-9236-4293-bd13-d469df855062-predictions.npz`.
- Seed 43: [manifest](747602c3-41b7-4f48-b0f4-fa6d831ae33c-manifest.json), [independent evaluation](747602c3-41b7-4f48-b0f4-fa6d831ae33c-evaluation.json), model `747602c3-41b7-4f48-b0f4-fa6d831ae33c-model.pkl`, predictions `747602c3-41b7-4f48-b0f4-fa6d831ae33c-predictions.npz`.
- Seed 44: [manifest](d68bb3c0-8815-44a2-a220-84960650f092-manifest.json), [independent evaluation](d68bb3c0-8815-44a2-a220-84960650f092-evaluation.json), model `d68bb3c0-8815-44a2-a220-84960650f092-model.pkl`, predictions `d68bb3c0-8815-44a2-a220-84960650f092-predictions.npz`.

Each evaluation contains train/validation/test confusion matrices and per-class precision, recall, and F1. Prediction archives contain labels and predictions for all three partitions. Model pickles, prediction archives, split-index archives, and logs are retained locally under this directory; `.gitignore` excludes those large/binary or operational files. JSON evidence and this report are trackable. Only load pickle artifacts from a trusted run.

## Reproduce

Install the worker Python requirements in the interpreter used to run the client-side verification. Use the Compose override below to obtain local MongoDB and build the official-source MinIO image rather than the unavailable original image. The Dockerfile checks a pinned upstream commit.

```bash
python3 -m pip install -r worker/fleet-requirements.txt
MONGODB_URL=mongodb://mongodb:27017/tensorfleet_gap1 MONGODB_DB=tensorfleet_gap1 \
  docker compose --env-file /dev/null -p tensorfleet-gap1 \
  -f docker-compose.yml -f result/compose.override.yml \
  up -d --build --scale worker=2 mongodb redis minio storage orchestrator api-gateway worker
# Wait for the gateway and storage health endpoints to return HTTP 200.
python3 benchmarks/run_mnist_fleet.py --output result
python3 result/verify_deployment.py
```

The MNIST runner downloads the source archive to `/tmp/tensorfleet-mnist.npz` if absent, verifies its checksum, uploads the exact partitions, submits the fixed-seed jobs, and saves evidence here. Rerunning in the same output directory replaces the summary files and adds new job artifacts; copy this directory first if preserving a particular run.

## Scope of the conclusion

This is genuine end-to-end training and artifact-correctness evidence for independent MNIST classification jobs on one Docker host. It does not demonstrate collaborative training of one model across workers or a speedup over sequential/native execution. Three seeds on a single fixed split are a small initial experiment, not a broad statistical evaluation. Controlled baselines and repeated scaling sweeps remain necessary.

The Docker test stack is left running, with local test data in its volumes. No frontend, separate worker-ml service, or model-registry UI was started for this experiment.

## Extended research evidence

See [repeated scaling, MPS, recovery and operations](research/README.md) for the
subsequent experiments. The results above preserve the original 40-tree budget.
