"""Regression tests for research validity, not just happy-path training."""
import importlib.util
import sys
import unittest
import tempfile
import json
import io
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class ResearchValidityTests(unittest.TestCase):
    def test_failed_scaling_job_is_not_successful_throughput(self):
        sys.path.insert(0, str(ROOT / 'benchmarks'))
        import benchmark_scaling
        from unittest.mock import patch
        def api(method, url, body=None):
            if url.endswith('/worker-activity'): return {'workers': []}
            if method == 'POST': return {'job_id': 'failed'}
            return {'status': 'FAILED', 'completed_tasks': 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'failed.json'
            with patch.object(sys, 'argv', ['benchmark_scaling', '--workers', '1', '--jobs', '1', '--epochs', '1', '--output', str(path)]), \
                 patch.object(benchmark_scaling, 'api', side_effect=api), patch.object(benchmark_scaling.time, 'sleep'), \
                 patch('sys.stdout', new_callable=io.StringIO), self.assertRaises(SystemExit):
                benchmark_scaling.main()
            result = json.loads(path.read_text())
            self.assertEqual(result['completed_jobs'], 0)
            self.assertEqual(result['failed_jobs'], 1)
            self.assertEqual(result['tasks_per_second'], 0)
            self.assertFalse(result['valid'])

    def test_submission_errors_remain_in_invalid_evidence(self):
        sys.path.insert(0, str(ROOT / 'benchmarks'))
        import benchmark_scaling
        from unittest.mock import patch
        def api(method, url, body=None):
            if method == 'POST': raise ConnectionError('injected outage')
            return {'workers': []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'failed.json'
            with patch.object(sys, 'argv', ['benchmark_scaling', '--workers', '1', '--jobs', '1', '--epochs', '1', '--output', str(path)]), patch.object(benchmark_scaling, 'api', side_effect=api), patch('sys.stdout', new_callable=io.StringIO), self.assertRaises(SystemExit):
                benchmark_scaling.main()
            result = json.loads(path.read_text())
            self.assertFalse(result['valid'])
            self.assertEqual(result['submission_failures'], 1)
            self.assertEqual(result['unfinished_jobs'], 1)
            self.assertEqual(result['errors'][0]['phase'], 'submission')

    def test_scaling_report_rejects_invalid_trials(self):
        sys.path.insert(0, str(ROOT / 'benchmarks'))
        import report_scaling
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.json'
            path.write_text(json.dumps({'workers': 1, 'jobs': 1, 'epochs_per_job': 1, 'total_tasks': 10,
                'schema_version': 2, 'valid': False, 'completed_jobs': 0, 'makespan_s': 1}))
            with patch.object(sys, 'argv', ['report', str(path)]), self.assertRaises(ValueError):
                report_scaling.main()

    def test_test_labels_cannot_change_checkpoint_selection(self):
        import torch
        sys.path.insert(0, str(ROOT / 'worker-ml'))
        import pytorch_trainer
        from unittest.mock import patch
        random = np.random.RandomState(7)
        X = random.normal(size=(150, 4))
        y = (X[:, 0] > 0).astype(np.int64)
        params = {'epochs': 3, 'batch_size': 16, 'random_state': 7}
        with patch.object(pytorch_trainer, 'get_device', return_value=torch.device('cpu')):
            model, metrics = pytorch_trainer.train_pytorch_model('pytorch_logistic', X[:120], X[120:], y[:120], y[120:], params)
            changed, changed_metrics = pytorch_trainer.train_pytorch_model('pytorch_logistic', X[:120], X[120:], y[:120], 1-y[120:], params)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, changed.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(metrics['history'], changed_metrics['history'])
        self.assertEqual(metrics['examples_processed'], metrics['fit_rows'] * metrics['epochs_completed'])
        self.assertEqual(metrics['validation_rows'] + metrics['fit_rows'], 120)


if __name__ == '__main__':
    unittest.main()
