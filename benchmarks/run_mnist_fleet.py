#!/usr/bin/env python3
"""Train/validate/test real MNIST jobs on an already running Docker fleet.

No storage fixture. Preserves source data provenance, partitions, API states,
model artifacts, predictions, independent evaluations, and service image IDs.
"""
import argparse
import hashlib
import io
import json
import pickle
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

MNIST_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
MNIST_SHA256 = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
ROOT = Path(__file__).resolve().parents[1]


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def get(url):
    response = requests.get(url, timeout=(5, 120))
    response.raise_for_status()
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gateway', default='http://localhost:8080')
    parser.add_argument('--storage', default='http://localhost:8081')
    parser.add_argument('--output', type=Path, default=ROOT / 'result')
    parser.add_argument('--mnist', type=Path, default=Path('/tmp/tensorfleet-mnist.npz'))
    parser.add_argument('--expected-workers', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    if not args.mnist.exists():
        args.mnist.write_bytes(get(MNIST_URL).content)
    source = args.mnist.read_bytes()
    if sha256(source) != MNIST_SHA256:
        raise ValueError('MNIST source checksum mismatch')
    with np.load(io.BytesIO(source), allow_pickle=False) as archive:
        X = archive['x_train'].reshape(60000, 784)
        y = archive['y_train']
        X_test = archive['x_test'].reshape(10000, 784)
        y_test = archive['y_test']
    train_index, validation_index = train_test_split(
        np.arange(60000), test_size=10000, stratify=y, random_state=2026)
    assert not set(train_index).intersection(validation_index)
    np.savez_compressed(args.output / 'mnist-split-indices.npz',
                        train_indices=train_index, validation_indices=validation_index)
    partitions = {'train': (X[train_index], y[train_index]),
                  'validation': (X[validation_index], y[validation_index]),
                  'test': (X_test, y_test)}
    data_provenance = {'source_url': MNIST_URL, 'source_sha256': MNIST_SHA256,
                       'original_train_rows': 60000, 'official_test_rows': 10000,
                       'split_seed': 2026, 'feature_count': 784,
                       'train_indices': train_index.tolist(), 'validation_indices': validation_index.tolist(),
                       'partitions': {}}
    for name, (features, labels) in partitions.items():
        buffer = io.BytesIO()
        np.savez_compressed(buffer, X=features, y=labels)
        payload = buffer.getvalue()
        path = f'datasets/mnist/{MNIST_SHA256[:12]}/{name}-split2026.npz'
        response = requests.post(f'{args.storage}/api/v1/upload/{path}',
                                 files={'file': (f'{name}.npz', payload)}, timeout=(5, 120))
        response.raise_for_status()
        assert sha256(get(f'{args.storage}/api/v1/download/{path}').content) == sha256(payload)
        data_provenance['partitions'][name] = {'path': path, 'rows': len(labels),
            'sha256': sha256(payload), 'class_counts': np.bincount(labels, minlength=10).tolist()}
        print(f'Uploaded and verified MNIST {name}: {len(labels)} rows', flush=True)
    write_json(args.output / 'mnist-data.json', data_provenance)

    # Fixed before observing validation/test results; no test-based selection.
    settings = {'n_estimators': 40, 'max_depth': 20, 'max_features': 'sqrt'}
    run = {'schema_version': 1, 'started_at': started_at, 'status': 'running',
           'scope': 'Single-host Docker independent-job correctness and quality; not a scaling comparison',
           'model_type': 'fleet_random_forest', 'estimator_params': settings,
           'model_seeds': [42, 43, 44], 'split_seed': 2026,
           'gateway': args.gateway, 'storage': args.storage,
           'host_platform': platform.platform(),
           'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'source_sha256': {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in (
               ROOT / 'worker/fleet_trainer.py', ROOT / 'worker/real_training.go', ROOT / 'worker/main.go',
               ROOT / 'orchestrator/cmd/orchestrator/main.go', ROOT / 'api-gateway/main.go',
               Path(__file__).resolve())},
           'jobs': []}
    container_ids = subprocess.check_output(['docker', 'ps', '-q', '--filter',
                                           'label=com.docker.compose.project=tensorfleet-gap1'], text=True).split()
    if container_ids:
        inspected = json.loads(subprocess.check_output(['docker', 'inspect', *container_ids], text=True))
        # Save reproducibility fields, never environment variables/credentials.
        run['containers'] = [{'name': item['Name'], 'image_id': item['Image'],
                               'image': item['Config']['Image'],
                               'cpu_quota_nanocpus': item['HostConfig']['NanoCpus'],
                               'memory_limit_bytes': item['HostConfig']['Memory']}
                              for item in inspected]
    write_json(args.output / 'mnist-run.json', run)
    batch_start = time.perf_counter()
    try:
        for seed in run['model_seeds']:
            payload = {'model_type': run['model_type'],
                       'dataset_path': data_provenance['partitions']['train']['path'], 'epochs': 1,
                       'hyperparameters': {'target_column': 'label', 'random_state': str(seed),
                           'validation_dataset_path': data_provenance['partitions']['validation']['path'],
                           'test_dataset_path': data_provenance['partitions']['test']['path'],
                           'estimator_params': json.dumps(settings)}}
            before = time.perf_counter()
            response = requests.post(f'{args.gateway}/api/v1/jobs', json=payload, timeout=(5, 30))
            response.raise_for_status()
            created = response.json()
            job_id = str(UUID(created['job_id']))
            assert created['num_tasks'] == 1, created
            run['jobs'].append({'job_id': job_id, 'seed': seed, 'request': payload,
                                'submission': created, 'submitted_offset_s': before - batch_start,
                                'submit_latency_s': time.perf_counter() - before})
            write_json(args.output / 'mnist-run.json', run)
            print(f'Submitted MNIST seed {seed}: {job_id}', flush=True)
        pending = {job['job_id']: job for job in run['jobs']}
        deadline = time.monotonic() + 600
        while pending and time.monotonic() < deadline:
            for job_id, job in list(pending.items()):
                state = get(f'{args.gateway}/api/v1/jobs/{job_id}').json()
                job['latest_state'] = state
                if state['status'] in ('COMPLETED', 'FAILED', 'CANCELLED'):
                    job['terminal_observed_offset_s'] = time.perf_counter() - batch_start
                    pending.pop(job_id)
                    print(f'MNIST seed {job["seed"]}: {state["status"]}', flush=True)
            write_json(args.output / 'mnist-run.json', run)
            if pending:
                time.sleep(.5)
        run['batch_terminal_observed_s'] = time.perf_counter() - batch_start
        assert not pending, f'Unfinished jobs: {list(pending)}'
        workers = set()
        for job in run['jobs']:
            job_id = job['job_id']
            state = job['latest_state']
            assert state['status'] == 'COMPLETED' and state['completed_tasks'] == 1, state
            manifest_path = state['message'].removeprefix('s3://')
            manifest = get(f'{args.storage}/api/v1/download/{manifest_path}').json()
            write_json(args.output / f'{job_id}-manifest.json', manifest)
            artifact = get(f'{args.storage}/api/v1/download/{manifest["artifact_uri"].removeprefix("s3://")}').content
            assert sha256(artifact) == manifest['artifact_sha256']
            assert manifest['dataset_sha256'] == data_provenance['partitions']['train']['sha256']
            for partition in ('validation', 'test'):
                assert manifest['external_datasets'][partition]['sha256'] == data_provenance['partitions'][partition]['sha256']
            assert manifest['split']['train_rows'] == 50000
            assert manifest['split']['validation_rows'] == manifest['split']['test_rows'] == 10000
            (args.output / f'{job_id}-model.pkl').write_bytes(artifact)
            # Only load artifacts just generated by this run on our local fleet.
            model = pickle.loads(artifact)
            evaluations = {}
            predictions = {}
            for name, (features, labels) in partitions.items():
                frame = pd.DataFrame(features, columns=manifest['features'])
                predicted = model.predict(frame)
                accuracy = float(accuracy_score(labels, predicted))
                assert abs(accuracy - manifest['metrics'][f'{name}_accuracy']) < 1e-12
                evaluations[name] = {'rows': len(labels), 'accuracy': accuracy,
                    'confusion_matrix': confusion_matrix(labels, predicted, labels=np.arange(10)).tolist(),
                    'classification_report': classification_report(labels, predicted, labels=np.arange(10),
                                                                    output_dict=True, zero_division=0)}
                if name != 'train':
                    loss = float(log_loss(labels, model.predict_proba(frame), labels=model.classes_))
                    assert abs(loss - manifest['metrics'][f'{name}_log_loss']) < 1e-12
                    evaluations[name]['log_loss'] = loss
                predictions[f'{name}_labels'] = labels
                predictions[f'{name}_predictions'] = predicted
            np.savez_compressed(args.output / f'{job_id}-predictions.npz', **predictions)
            write_json(args.output / f'{job_id}-evaluation.json', evaluations)
            assert state['current_accuracy'] == evaluations['test']['accuracy']
            workers.add(manifest['worker_id'])
            job.update({'worker_id': manifest['worker_id'], 'artifact_sha256': manifest['artifact_sha256'],
                        'artifact_uri': manifest['artifact_uri'], 'metrics': manifest['metrics'],
                        'independently_verified': True})
        assert len(workers) == args.expected_workers, f'Expected {args.expected_workers} participating workers; saw {workers}'
        run['participating_workers'] = sorted(workers)
        run['independent_verification_finished_s'] = time.perf_counter() - batch_start
        run['summary'] = {metric: {'mean': statistics.mean(job['metrics'][metric] for job in run['jobs']),
                                  'sample_stdev': statistics.stdev(job['metrics'][metric] for job in run['jobs'])}
                          for metric in ('train_accuracy', 'validation_accuracy', 'test_accuracy', 'fit_time_s')}
        run['status'] = 'passed'
        print(json.dumps(run['summary'], indent=2), flush=True)
    except BaseException as error:
        run['status'] = 'failed'
        run['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        run['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'mnist-run.json', run)
    print(f'PASS: MNIST train/validation/test and artifacts verified; saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
