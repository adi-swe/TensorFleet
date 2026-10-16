#!/usr/bin/env python3
"""Native integration smoke check with a storage HTTP fixture, NOT a benchmark.

Builds and starts the actual gateway, orchestrator, and two Go workers. Runs
real sklearn fits and checks stored models. Does not validate MinIO/MongoDB,
Docker, cross-node operation, reliability, or scaling performance.
"""
import email.policy
import hashlib
import io
import json
import os
import pickle
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def api(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


class StorageFixture(BaseHTTPRequestHandler):
    objects = {}
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_GET(self):
        prefix = '/api/v1/download/'
        key = unquote(self.path.removeprefix(prefix))
        with self.lock:
            body = self.objects.get(key)
        if not self.path.startswith(prefix) or body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        prefix = '/api/v1/upload/'
        if not self.path.startswith(prefix):
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers['Content-Length']))
        message = BytesParser(policy=email.policy.default).parsebytes(
            f'Content-Type: {self.headers["Content-Type"]}\r\nMIME-Version: 1.0\r\n\r\n'.encode() + body)
        parts = [part for part in message.iter_parts()
                 if part.get_param('name', header='Content-Disposition') == 'file']
        if len(parts) != 1:
            self.send_error(400)
            return
        with self.lock:
            self.objects[unquote(self.path.removeprefix(prefix))] = parts[0].get_payload(decode=True)
        self.send_response(201)
        self.end_headers()
        self.wfile.write(b'{}')


def wait_for(check, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.1)
    raise TimeoutError('Native service did not reach expected state')


def main():
    objects = StorageFixture.objects
    objects.clear()
    objects['datasets/iris.csv'] = load_iris(as_frame=True).frame.to_csv(index=False).encode()
    storage = ThreadingHTTPServer(('127.0.0.1', 0), StorageFixture)
    threading.Thread(target=storage.serve_forever, daemon=True).start()
    storage_url = f'http://127.0.0.1:{storage.server_port}'
    processes, logs = [], []
    with tempfile.TemporaryDirectory(prefix='tensorfleet-smoke-') as directory:
        temporary = Path(directory)
        try:
            builds = [('orchestrator', './cmd/orchestrator'), ('api-gateway', '.'), ('worker', '.')]
            for service, package in builds:
                subprocess.run(['go', 'build', '-race', '-o', str(temporary / service), package], cwd=ROOT / service, check=True)
            orchestrator_port, gateway_port = free_port(), free_port()
            base = {**os.environ, 'REDIS_ADDR': f'127.0.0.1:{free_port()}',
                    'ALLOW_EPHEMERAL_JOBS': 'true',
                    'ORCHESTRATOR_ADDR': f'127.0.0.1:{orchestrator_port}',
                    'STORAGE_SERVICE_URL': storage_url}

            def start(service, label, **extra):
                log_path = temporary / f'{label}.log'
                log_file = log_path.open('w')
                logs.append((log_path, log_file))
                processes.append(subprocess.Popen(
                    [str(temporary / service)], env={**base, **extra},
                    cwd=ROOT / service, stdout=log_file, stderr=subprocess.STDOUT))

            start('orchestrator', 'orchestrator', PORT=str(orchestrator_port))
            def orchestrator_ready():
                with socket.create_connection(('127.0.0.1', orchestrator_port), timeout=.2):
                    return True
            wait_for(orchestrator_ready)
            start('api-gateway', 'gateway', PORT=str(gateway_port))
            gateway = f'http://127.0.0.1:{gateway_port}'
            wait_for(lambda: api(f'{gateway}/health'))
            for number in range(2):
                start('worker', f'worker-{number}', PORT=str(free_port()), METRICS_PORT='0',
                      FETCH_INTERVAL_MS='100', MAX_CONCURRENT_TASKS='1',
                      TRAINING_PYTHON=sys.executable, FLEET_TRAINER_PATH=str(ROOT / 'worker/fleet_trainer.py'))

            job_ids = []
            for algorithm in ('fleet_random_forest', 'fleet_logistic_regression',
                              'fleet_decision_tree', 'fleet_random_forest'):
                created = api(f'{gateway}/api/v1/jobs', {
                    'model_type': algorithm, 'dataset_path': 'datasets/iris.csv',
                    'hyperparameters': {'target_column': 'target'},
                })
                assert created['num_tasks'] == 1, created
                job_ids.append(created['job_id'])
            workers = set()
            for job_id in job_ids:
                def terminal():
                    state = api(f'{gateway}/api/v1/jobs/{job_id}')
                    return state if state['status'] in ('COMPLETED', 'FAILED') else None
                state = wait_for(terminal, timeout=90)
                assert state['status'] == 'COMPLETED', state
                assert state['completed_tasks'] == 1, state
                manifest = json.loads(objects[state['message'].removeprefix('s3://')])
                artifact = objects[manifest['artifact_uri'].removeprefix('s3://')]
                assert hashlib.sha256(artifact).hexdigest() == manifest['artifact_sha256']
                model = pickle.loads(artifact)  # Only our own just-produced test artifacts.
                rows = pd.DataFrame(manifest['verification']['rows'])[manifest['features']]
                assert model.predict(rows).tolist() == manifest['verification']['predictions']
                # Re-evaluate the entire held-out partition, not just the fixture rows.
                dataset = pd.read_csv(io.BytesIO(objects['datasets/iris.csv']))
                held_out = dataset.iloc[manifest['split']['test_indices']]
                actual_accuracy = float((model.predict(held_out[manifest['features']]) == held_out['target']).mean())
                assert actual_accuracy == manifest['metrics']['test_accuracy']
                assert actual_accuracy == state['current_accuracy']
                workers.add(manifest['worker_id'])
            assert len(workers) == 2, f'Expected two participating workers, observed {workers}'

            failed = api(f'{gateway}/api/v1/jobs', {
                'model_type': 'fleet_random_forest', 'dataset_path': 'datasets/missing.csv',
                'hyperparameters': {'target_column': 'target'},
            })
            def failed_state():
                state = api(f'{gateway}/api/v1/jobs/{failed["job_id"]}')
                return state if state['status'] == 'FAILED' else None
            failure = wait_for(failed_state, timeout=45)
            assert failure['completed_tasks'] == 0
            assert f'artifacts/fleet/{failed["job_id"]}/result.json' not in objects

            try:
                api(f'{gateway}/api/v1/jobs', {
                    'model_type': 'fleet_random_forest', 'dataset_path': 'datasets/iris.csv',
                    'epochs': 10, 'hyperparameters': {'target_column': 'target'},
                })
                raise AssertionError('Multi-epoch independent fit was accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 400
            for path, handle in logs:
                handle.flush()
                assert 'DATA RACE' not in path.read_text(), f'Race detector reported an error in {path.name}'
            print('PASS: four real jobs on two workers; model checksums, reload predictions, '
                  'held-out accuracy, terminal failure, and invalid request handling verified.')
            print('Storage was an HTTP test fixture; these are correctness checks, not scaling measurements.')
        except BaseException:
            for path, handle in logs:
                handle.flush()
                print(f'--- {path.name} ---\n{path.read_text()[-6000:]}', file=sys.stderr)
            raise
        finally:
            for process in reversed(processes):
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for _, handle in logs:
                handle.close()
            storage.shutdown()
            storage.server_close()


if __name__ == '__main__':
    main()
