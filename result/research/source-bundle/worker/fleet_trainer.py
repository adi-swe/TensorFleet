#!/usr/bin/env python3
"""One real, independent training job. JSON request on stdin, result on stdout.

Artifacts use the storage service's existing object API, without MongoDB fallback
or synthetic data. Only load pickle bytes produced by this invocation.
"""
import hashlib
import io
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

ALGORITHMS = {
    'fleet_random_forest': RandomForestClassifier,
    'fleet_logistic_regression': LogisticRegression,
    'fleet_decision_tree': DecisionTreeClassifier,
    'fleet_distributed_random_forest': RandomForestClassifier,
    'fleet_accelerated_logistic': LogisticRegression,
}


def capabilities():
    from accelerated_logistic import available_devices
    role = os.environ.get('TENSORFLEET_WORKER_ROLE', 'all')
    basic = [] if role == 'accelerator' else [name for name in ALGORITHMS if name != 'fleet_accelerated_logistic']
    return basic + [f'fleet_accelerated_logistic:{device}' for device in available_devices()
                    if role != 'cpu' or device == 'cpu']


class ObjectStore:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')

    def url(self, action, path):
        path = path.removeprefix('s3://')
        bucket, separator, name = path.partition('/')
        if not separator or not name or bucket not in ('datasets', 'models', 'artifacts'):
            raise ValueError('Expected datasets/<object>, models/<object>, or artifacts/<object>')
        return f'{self.base_url}/api/v1/{action}/{quote(bucket)}/{quote(name, safe="/")}'

    def get(self, path):
        response = requests.get(self.url('download', path), timeout=(5, 60))
        response.raise_for_status()
        return response.content

    def put(self, path, content):
        response = requests.post(
            self.url('upload', path),
            files={'file': ('artifact', content, 'application/octet-stream')},
            timeout=(5, 60),
        )
        response.raise_for_status()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def load_dataset(content, path, target):
    if path.endswith('.npz'):
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            features, labels = archive['X'], archive['y']
            if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
                raise ValueError('NPZ requires X (rows, features) and y (rows,)')
            frame = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
            frame[target] = labels
            return frame
    return pd.read_csv(io.BytesIO(content))


def train_job(job, store):
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    algorithm = job['model_type']
    if algorithm not in ALGORITHMS:
        raise ValueError(f'Unsupported fleet algorithm: {algorithm}')
    params = dict(job.get('hyperparameters') or {})
    stage = params.pop('_fleet_stage', 'fit')
    shard_index = int(params.pop('_fleet_shard_index', '0'))
    shard_count = int(params.pop('_fleet_shards', '1'))
    device = params.pop('device', 'auto')
    if device not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError('device must be auto, cpu, mps, or cuda')
    if algorithm != 'fleet_accelerated_logistic' and device not in ('auto', 'cpu'):
        raise ValueError('This tree/sklearn backend supports CPU only')
    target = params.pop('target_column', None)
    if not target:
        raise ValueError('hyperparameters.target_column is required')
    seed = int(params.pop('random_state', '42'))
    test_size = float(params.pop('test_size', '0.2'))
    validation_path = params.pop('validation_dataset_path', None)
    test_path = params.pop('test_dataset_path', None)
    if bool(validation_path) != bool(test_path):
        raise ValueError('validation_dataset_path and test_dataset_path must be supplied together')
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between zero and one')
    # Model parameters are a JSON object inside the existing string-valued map.
    estimator_params = json.loads(params.pop('estimator_params', '{}'))
    if params:
        raise ValueError(f'Unknown fleet hyperparameters: {sorted(params)}')
    if not isinstance(estimator_params, dict):
        raise ValueError('estimator_params must be a JSON object')
    if 'random_state' in estimator_params or 'n_jobs' in estimator_params:
        raise ValueError('Use top-level random_state; trainer controls n_jobs=1')

    dataset = store.get(job['dataset_path'])
    data = load_dataset(dataset, job['dataset_path'], target)
    if target not in data or data[target].isna().any():
        raise ValueError('Target column must exist and contain no missing labels')
    X, y = data.drop(columns=[target]), data[target]
    if X.shape[1] == 0 or y.nunique() < 2:
        raise ValueError('Classification needs features and at least two classes')
    external_datasets = {}
    X_validation = y_validation = None
    if validation_path:
        partitions = []
        for name, path in [('validation', validation_path), ('test', test_path)]:
            content = store.get(path)
            frame = load_dataset(content, path, target)
            if frame.columns.tolist() != data.columns.tolist() or frame[target].isna().any():
                raise ValueError(f'{name} dataset schema must match training data')
            if not set(frame[target].unique()).issubset(set(y.unique())):
                raise ValueError(f'{name} dataset contains unseen labels')
            external_datasets[name] = {'path': path, 'sha256': digest(content), 'rows': len(frame)}
            partitions.append((frame.drop(columns=[target]), frame[target]))
        X_train, y_train = X, y
        (X_validation, y_validation), (X_test, y_test) = partitions
        train_index, test_index = np.arange(len(X_train)), np.arange(len(X_test))
    else:
        train_index, test_index = train_test_split(
            np.arange(len(data)), test_size=test_size, random_state=seed, stratify=y,
        )
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical),
    ])
    options = {'random_state': seed, **estimator_params}
    distribution = {}
    if algorithm in ('fleet_random_forest', 'fleet_distributed_random_forest'):
        options['n_jobs'] = 1
    if algorithm == 'fleet_distributed_random_forest':
        total_trees = int(options.get('n_estimators', 100))
        if not 1 <= shard_count <= total_trees or not 0 <= shard_index < shard_count:
            raise ValueError('Invalid forest tree partition')
        if options.get('oob_score') or options.get('warm_start'):
            raise ValueError('Distributed forest does not support oob_score or warm_start')
        start_tree = shard_index * total_trees // shard_count
        end_tree = (shard_index + 1) * total_trees // shard_count
        distribution = {'stage': stage, 'shards': shard_count, 'total_trees': total_trees,
                        'shard_index': shard_index, 'tree_start': start_tree, 'tree_count': end_tree - start_tree}
        if stage not in ('shard', 'aggregate'):
            raise ValueError('Distributed forest requires orchestrator-assigned stages')
        if stage == 'shard':
            # Advance the public RandomState by the number of earlier trees.
            # Every tree gets the same seed regardless of partition count.
            random = np.random.RandomState(seed)
            random.randint(np.iinfo(np.int32).max, size=start_tree)
            options.update(n_estimators=end_tree - start_tree, random_state=random)
    if algorithm == 'fleet_logistic_regression':
        options.setdefault('max_iter', 1000)
    constructor_options = {'random_state': seed} if algorithm == 'fleet_accelerated_logistic' else options
    model = Pipeline([('preprocess', preprocessing), ('model', ALGORITHMS[algorithm](**constructor_options))])
    fit_started = time.perf_counter()
    device_info = {'requested': device, 'actual': 'cpu'}
    if algorithm == 'fleet_distributed_random_forest' and stage == 'aggregate':
        partials, manifests = [], []
        for index in range(shard_count):
            manifest = json.loads(store.get(f'artifacts/fleet/{job["job_id"]}/shards/{index}.json'))
            expected_start = index * total_trees // shard_count
            expected_count = (index + 1) * total_trees // shard_count - expected_start
            if (manifest['dataset_sha256'] != digest(dataset) or manifest['external_datasets'] != external_datasets
                    or manifest['distribution']['tree_start'] != expected_start
                    or manifest['distribution']['tree_count'] != expected_count
                    or manifest['versions']['sklearn'] != sklearn.__version__):
                raise ValueError('Forest shard provenance or tree partition mismatch')
            raw = store.get(manifest['artifact_uri'])
            if digest(raw) != manifest['artifact_sha256']:
                raise ValueError('Forest shard checksum mismatch')
            partial = pickle.loads(raw)
            if len(partial['model'].estimators_) != expected_count:
                raise ValueError('Forest shard tree count mismatch')
            if partials and (not np.array_equal(partial.classes_, partials[0].classes_) or
                            pickle.dumps(partial['preprocess']) != pickle.dumps(partials[0]['preprocess'])):
                raise ValueError('Forest shards have incompatible preprocessing or classes')
            partials.append(partial)
            manifests.append(manifest)
        model = partials[0]
        trees = [tree for partial in partials for tree in partial['model'].estimators_]
        model['model'].estimators_ = trees
        model['model'].n_estimators = len(trees)
        model['model'].random_state = seed
        distribution.update(tree_start=0, tree_count=len(trees),
                            worker_ids=sorted({manifest['worker_id'] for manifest in manifests}),
                            sum_shard_fit_time_s=sum(manifest['metrics']['fit_time_s'] for manifest in manifests),
                            shard_artifacts=[manifest['artifact_uri'] for manifest in manifests])
    elif algorithm == 'fleet_accelerated_logistic':
        from accelerated_logistic import train_logistic
        transformed = preprocessing.fit_transform(X_train)
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        estimator, device_info = train_logistic(transformed, y_train, estimator_params, seed, device)
        model = Pipeline([('preprocess', preprocessing), ('model', estimator)])
    else:
        with threadpool_limits(limits=1):
            model.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_started
    train_accuracy = float(accuracy_score(y_train, model.predict(X_train)))
    prediction = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, prediction))
    loss = float(log_loss(y_test, model.predict_proba(X_test), labels=model.classes_))
    validation_metrics = {}
    if X_validation is not None:
        validation_metrics = {
            'validation_accuracy': float(accuracy_score(y_validation, model.predict(X_validation))),
            'validation_log_loss': float(log_loss(y_validation, model.predict_proba(X_validation), labels=model.classes_)),
        }

    attempt = str(uuid4())
    artifact_path = f'models/fleet/{job["job_id"]}/{job["task_id"]}/{attempt}/model.pkl'
    artifact = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    store.put(artifact_path, artifact)
    downloaded = store.get(artifact_path)
    if digest(downloaded) != digest(artifact):
        raise ValueError('Stored model checksum mismatch')
    restored = pickle.loads(downloaded)
    if not np.array_equal(restored.predict(X_test), prediction):
        raise ValueError('Stored model predictions differ from trained model')

    result = {
        'schema_version': 1, 'status': 'completed',
        'job_id': job['job_id'], 'task_id': job['task_id'],
        'worker_id': job['worker_id'], 'attempt_id': attempt,
        'algorithm': algorithm, 'hyperparameters': job.get('hyperparameters', {}),
        'device': device_info, 'distribution': distribution,
        'dataset_path': job['dataset_path'], 'dataset_sha256': digest(dataset),
        'external_datasets': external_datasets,
        'target_column': target, 'features': X.columns.tolist(),
        'split': {'mode': 'explicit_train_validation_test' if validation_path else 'seeded_train_test',
                  'seed': seed, 'train_rows': len(train_index), 'test_rows': len(test_index),
                  'validation_rows': 0 if X_validation is None else len(X_validation),
                  'test_indices': test_index.tolist()},
        'metrics': {'train_accuracy': train_accuracy,
                    'test_accuracy': accuracy, 'test_log_loss': loss, **validation_metrics,
                    'fit_time_s': fit_time, 'elapsed_before_manifest_s': time.perf_counter() - started},
        'artifact_uri': f's3://{artifact_path}', 'artifact_sha256': digest(artifact),
        'artifact_verified': True,
        'verification': {
            'rows': json.loads(X_test.head(8).to_json(orient='records', double_precision=15)),
            'predictions': prediction[:8].tolist(),
        },
        'started_at': started_at, 'completed_at': datetime.now(timezone.utc).isoformat(),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__},
        'threads_per_trainer': 1,
    }
    result_path = f'artifacts/fleet/{job["job_id"]}/result.json'
    if distribution.get('stage') == 'shard':
        result_path = f'artifacts/fleet/{job["job_id"]}/shards/{shard_index}.json'
    result['result_uri'] = f's3://{result_path}'
    manifest = json.dumps(result, allow_nan=False).encode()
    store.put(result_path, manifest)
    if store.get(result_path) != manifest:
        raise ValueError('Stored result manifest differs from uploaded manifest')
    return result


if __name__ == '__main__':
    try:
        if '--capabilities' in sys.argv:
            print(json.dumps(capabilities()))
            sys.exit(0)
        request = json.load(sys.stdin)
        result = train_job(request, ObjectStore(os.environ.get('STORAGE_SERVICE_URL', 'http://storage:8081')))
        print(json.dumps(result, allow_nan=False))
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
