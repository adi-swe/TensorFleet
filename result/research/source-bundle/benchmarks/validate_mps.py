"""Exercise native MPS and CPU logistic training through the real gateway.

Uses existing uploaded MNIST partitions. Keeps the Docker stack intact and
terminates its native worker on exit. macOS GPU access requires native execution.
"""
import argparse
import random
import statistics
import hashlib
import json
import os
import pickle
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'result/research'


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return str(sock.getsockname()[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5)
    args = parser.parse_args()
    if args.trials < 2: parser.error('at least two trials are required')
    OUTPUT.mkdir(exist_ok=True)
    data = json.loads((ROOT / 'result/mnist-data.json').read_text())
    subprocess.run(['go', 'build', '-o', '/tmp/tensorfleet-native-worker', '.'], cwd=ROOT / 'worker', check=True)
    report = {'status': 'running', 'jobs': [], 'scope': 'Native single-device MPS versus CPU; no multi-GPU claim'}
    log = (OUTPUT / 'native-mps.log').open('w')
    process = subprocess.Popen(['/tmp/tensorfleet-native-worker'], cwd=ROOT / 'worker', stdout=log, stderr=log, env={
        **os.environ, 'PORT': free_port(), 'METRICS_PORT': '0', 'ORCHESTRATOR_ADDR': 'localhost:50051',
        'STORAGE_SERVICE_URL': 'http://localhost:8081', 'TRAINING_PYTHON': sys.executable,
        'MAX_CONCURRENT_TASKS': '1', 'FETCH_INTERVAL_MS': '100',
    })
    try:
        cases = [(trial, device) for trial in range(args.trials) for device in ('mps', 'cpu')]
        random.Random(2026).shuffle(cases)
        report['case_order'] = cases
        for trial, device in cases:
            request = {'model_type': 'fleet_accelerated_logistic', 'dataset_path': data['partitions']['train']['path'],
                'epochs': 1, 'hyperparameters': {'target_column': 'label', 'random_state': str(42 + trial), 'device': device,
                    'validation_dataset_path': data['partitions']['validation']['path'],
                    'test_dataset_path': data['partitions']['test']['path'],
                    'estimator_params': json.dumps({'epochs': 30, 'batch_size': 2048, 'learning_rate': .01})}}
            started = time.perf_counter()
            response = requests.post('http://localhost:8080/api/v1/jobs', json=request, timeout=15)
            response.raise_for_status()
            job_id = response.json()['job_id']
            record = {'trial': trial, 'request': request, 'submission': response.json()}
            report['jobs'].append(record)
            (OUTPUT / 'accelerated-logistic.json').write_text(json.dumps(report, indent=2))
            while time.perf_counter() - started < 300:
                state = requests.get(f'http://localhost:8080/api/v1/jobs/{job_id}', timeout=10).json()
                if state['status'] in ('COMPLETED', 'FAILED', 'CANCELLED'):
                    break
                if process.poll() is not None:
                    raise RuntimeError('Native worker exited; inspect native-mps.log')
                time.sleep(.5)
            record.update(state=state, submission_to_terminal_observed_s=time.perf_counter()-started)
            assert state['status'] == 'COMPLETED', state
            response = requests.get('http://localhost:8081/api/v1/download/'+state['message'].removeprefix('s3://'), timeout=30)
            response.raise_for_status()
            manifest = response.json()
            assert manifest['device']['actual'] == device, manifest['device']
            response = requests.get('http://localhost:8081/api/v1/download/'+manifest['artifact_uri'].removeprefix('s3://'), timeout=30)
            response.raise_for_status()
            artifact = response.content
            assert hashlib.sha256(artifact).hexdigest() == manifest['artifact_sha256']
            model = pickle.loads(artifact)
            rows = pd.DataFrame(manifest['verification']['rows'])[manifest['features']]
            assert model.predict(rows).tolist() == manifest['verification']['predictions']
            (OUTPUT / f'logistic-{device}-{trial}-model.pkl').write_bytes(artifact)
            (OUTPUT / f'logistic-{device}-{trial}-manifest.json').write_text(json.dumps(manifest, indent=2))
            record.update(metrics=manifest['metrics'], device=manifest['device'], artifact_verified=True,
                          worker_id=manifest['worker_id'])
            print(device, manifest['metrics'], flush=True)
        report['summary'] = {device: {
            'trials': args.trials,
            'mean_fit_time_s': statistics.mean(job['metrics']['fit_time_s'] for job in report['jobs'] if job['device']['actual'] == device),
            'sample_stdev_fit_time_s': statistics.stdev(job['metrics']['fit_time_s'] for job in report['jobs'] if job['device']['actual'] == device),
            'mean_test_accuracy': statistics.mean(job['metrics']['test_accuracy'] for job in report['jobs'] if job['device']['actual'] == device),
        } for device in ('mps', 'cpu')}
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        process.terminate()
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        (OUTPUT / 'accelerated-logistic.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
