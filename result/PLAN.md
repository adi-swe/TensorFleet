# MNIST Docker validation plan

Predeclared before running the MNIST jobs, 16 October 2026.

- Use the real gateway, Go orchestrator, two Docker workers, storage service,
  local Redis, local MongoDB, and real MinIO. No storage or training fixture.
- Obtain `mnist.npz` from the TensorFlow/Keras dataset endpoint. Check SHA-256
  `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
- Split only the official 60,000 training images: 50,000 fit / 10,000 validation,
  stratified with seed 2026. Preserve the official 10,000 test images intact.
- Flatten each 28 × 28 image into 784 pixel features. Fit imputation/scaling and
  the classifier on the training partition only.
- Run `fleet_random_forest` with 40 trees, max depth 20, and `max_features=sqrt`.
  Model seeds: 42, 43, 44. One numerical thread and one task per worker, with
  one CPU and 2 GiB memory allocated to each worker. Do not tune on test results.
- Verify all three jobs finish, both workers participate, and the saved/reloaded
  models reproduce train, validation, and test metrics independently.
- Save raw manifests, model files, predictions, per-class reports, confusion
  matrices, data hashes, split indices, image IDs, source hashes, and timings
  beneath `result/`. Classifier metrics are measurements, not acceptance targets.
- Test missing-data and invalid-request behavior after the real fits. Verify
  saved objects survive a MinIO container restart.

This establishes real training and output correctness on one local Docker host.
It does not establish distributed single-model training, scaling speedup, long-run
reliability, or broad generalization across datasets. Three seeds provide only a
small initial quality sample; report all three rather than selecting the best.

Reproduction uses the repo's Compose file plus `result/compose.override.yml`.
The MinIO test image builds official upstream release `RELEASE.2025-10-15T17-29-55Z`
from the pinned commit in `result/minio-build/Dockerfile`, because the original
Docker Hub image was inaccessible and the official Quay image returned 401.
This is a test-only deployment override; the base MinIO image remains unchanged.
