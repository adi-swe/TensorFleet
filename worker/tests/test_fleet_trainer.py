"""Real fits and artifact integrity checks; no infrastructure required."""
import copy
import sys
import importlib.util
import json
import io
import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('fleet_trainer', Path(__file__).resolve().parents[1] / 'fleet_trainer.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class MemoryStore:
    def __init__(self):
        data = load_iris(as_frame=True).frame
        data['category'] = np.where(data.index % 2, 'odd', 'even')
        self.objects = {'datasets/iris.csv': data.to_csv(index=False).encode()}

    def get(self, path):
        return self.objects[path.removeprefix('s3://')]

    def put(self, path, content):
        self.objects[path] = content


class FleetTrainerTests(unittest.TestCase):
    def job(self, algorithm='fleet_random_forest'):
        return {'job_id': 'job', 'task_id': 'task', 'worker_id': 'worker',
                'model_type': algorithm, 'dataset_path': 'datasets/iris.csv',
                'hyperparameters': {'target_column': 'target', 'random_state': '7'}}

    def test_real_models_survive_storage_roundtrip_with_preprocessing(self):
        for algorithm in ('fleet_random_forest', 'fleet_logistic_regression', 'fleet_decision_tree'):
            with self.subTest(algorithm=algorithm):
                store = MemoryStore()
                result = trainer.train_job(self.job(algorithm), store)
                saved = json.loads(store.get(result['result_uri']))
                artifact = store.get(saved['artifact_uri'])
                self.assertEqual(saved['artifact_sha256'], trainer.digest(artifact))
                self.assertTrue(saved['artifact_verified'])
                model = pickle.loads(artifact)
                rows = pd.DataFrame(saved['verification']['rows'])[saved['features']]
                self.assertEqual(model.predict(rows).tolist(), saved['verification']['predictions'])
                self.assertGreater(saved['metrics']['test_accuracy'], 0.7)
                self.assertEqual(saved['split']['train_rows'], 120)

    def test_corrupt_artifact_never_publishes_success(self):
        store = MemoryStore()
        original_get = store.get
        store.get = lambda path: b'corrupted' if path.startswith('models/') else original_get(path)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            trainer.train_job(self.job(), store)
        self.assertNotIn('artifacts/fleet/job/result.json', store.objects)

    def test_upload_failure_never_publishes_success(self):
        store = MemoryStore()
        def fail(path, content):
            raise OSError('storage unavailable')
        store.put = fail
        with self.assertRaisesRegex(OSError, 'storage unavailable'):
            trainer.train_job(self.job(), store)

    def test_missing_dataset_does_not_fall_back_to_sample_data(self):
        job = self.job()
        job['dataset_path'] = 'datasets/missing.csv'
        with self.assertRaises(KeyError):
            trainer.train_job(job, MemoryStore())

    def test_rejects_invalid_or_oversubscribed_configuration(self):
        for changes in ({'target_column': ''}, {'test_size': '1'},
                        {'estimator_params': '{"n_jobs": 8}'}, {'unknown': 'value'}):
            job = copy.deepcopy(self.job())
            job['hyperparameters'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                trainer.train_job(job, MemoryStore())

    def test_explicit_train_validation_test_npz(self):
        data = load_iris(as_frame=True).frame
        # Each split contains all three classes, without row overlap.
        store = MemoryStore()
        for offset, name in enumerate(('train', 'validation', 'test')):
            partition = data.iloc[offset::3]
            buffer = io.BytesIO()
            np.savez_compressed(buffer, X=partition.drop(columns='target').to_numpy(), y=partition['target'].to_numpy())
            store.objects[f'datasets/{name}.npz'] = buffer.getvalue()
        job = self.job()
        job['dataset_path'] = 'datasets/train.npz'
        job['hyperparameters'].update({'validation_dataset_path': 'datasets/validation.npz',
                                       'test_dataset_path': 'datasets/test.npz'})
        result = trainer.train_job(job, store)
        self.assertEqual(result['split']['mode'], 'explicit_train_validation_test')
        self.assertEqual(result['split']['train_rows'], 50)
        self.assertEqual(result['split']['validation_rows'], 50)
        self.assertEqual(result['split']['test_rows'], 50)
        self.assertIn('validation_accuracy', result['metrics'])
        self.assertEqual(result['external_datasets']['test']['sha256'], trainer.digest(store.get('datasets/test.npz')))
        self.assertEqual(len(result['features']), 4)

    def test_external_partitions_must_be_supplied_together(self):
        job = self.job()
        job['hyperparameters']['test_dataset_path'] = 'datasets/test.npz'
        with self.assertRaisesRegex(ValueError, 'supplied together'):
            trainer.train_job(job, MemoryStore())

    def test_distributed_forest_matches_single_forest_with_uneven_shards(self):
        store = MemoryStore()
        for index in range(3):
            job = self.job('fleet_distributed_random_forest')
            job.update(task_id=f'shard-{index}', worker_id=f'worker-{index % 2}')
            job['hyperparameters'].update({'estimator_params': '{"n_estimators":13}',
                '_fleet_stage': 'shard', '_fleet_shards': '3', '_fleet_shard_index': str(index)})
            trainer.train_job(job, store)
        job['task_id'] = 'aggregate'
        job['hyperparameters'].update(_fleet_stage='aggregate', _fleet_shard_index='0')
        result = trainer.train_job(job, store)
        combined = pickle.loads(store.get(result['artifact_uri']))
        baseline = self.job()
        baseline['job_id'] = 'reference'
        baseline['hyperparameters']['estimator_params'] = '{"n_estimators":13}'
        reference = trainer.train_job(baseline, store)
        reference_model = pickle.loads(store.get(reference['artifact_uri']))
        data = pd.read_csv(io.BytesIO(store.get('datasets/iris.csv'))).drop(columns='target')
        np.testing.assert_allclose(combined.predict_proba(data), reference_model.predict_proba(data), atol=1e-14)
        self.assertEqual(len(combined['model'].estimators_), 13)
        self.assertEqual(len(result['distribution']['worker_ids']), 2)
        self.assertEqual([tree.random_state for tree in combined['model'].estimators_],
                         [tree.random_state for tree in reference_model['model'].estimators_])

    def test_accelerated_logistic_cpu_artifact_is_portable(self):
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            self.skipTest('optional torch not installed')
        job = self.job('fleet_accelerated_logistic')
        job['hyperparameters'].update(device='cpu', estimator_params='{"epochs":80,"batch_size":64,"learning_rate":0.03}')
        result = trainer.train_job(job, MemoryStore())
        self.assertEqual(result['device']['actual'], 'cpu')
        self.assertGreater(result['metrics']['test_accuracy'], .8)

    def test_trees_reject_explicit_gpu(self):
        job = self.job()
        job['hyperparameters']['device'] = 'mps'
        with self.assertRaisesRegex(ValueError, 'CPU only'):
            trainer.train_job(job, MemoryStore())


if __name__ == '__main__':
    unittest.main()
