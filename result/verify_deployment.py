"""Check failure behavior and restart persistence after the MNIST run.

Run from repo root: python3 result/verify_deployment.py
Only targets the local tensorfleet-gap1 containers used by this experiment.
"""
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OUTPUT = Path(__file__).resolve().parent
GATEWAY = 'http://localhost:8080'
STORAGE = 'http://localhost:8081'


def wait_for(check, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except requests.RequestException:
            pass
        time.sleep(.5)
    raise TimeoutError('Expected deployment state not reached')


def main():
    run = json.loads((OUTPUT / 'mnist-run.json').read_text())
    assert run['status'] == 'passed', 'MNIST verification must pass before restart checks'
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'checks': [], 'status': 'running'}
    data = json.loads((OUTPUT / 'mnist-data.json').read_text())

    def failed_job(dataset_path):
        response = requests.post(f'{GATEWAY}/api/v1/jobs', json={
            'model_type': 'fleet_random_forest', 'dataset_path': dataset_path,
            'hyperparameters': {'target_column': 'label'}}, timeout=15)
        response.raise_for_status()
        job_id = response.json()['job_id']
        def terminal():
            state = requests.get(f'{GATEWAY}/api/v1/jobs/{job_id}', timeout=10).json()
            return state if state['status'] in ('FAILED', 'COMPLETED', 'CANCELLED') else None
        state = wait_for(terminal)
        assert state['status'] == 'FAILED' and state['completed_tasks'] == 0, state
        response = requests.get(f'{STORAGE}/api/v1/download/artifacts/fleet/{job_id}/result.json', timeout=10)
        # Storage can be deliberately unavailable in the second failure check.
        return state, response.status_code

    try:
        for name, patch in [
            ('unsupported_algorithm', {'model_type': 'fleet_unknown'}),
            ('invalid_epochs', {'epochs': 10}),
            ('missing_target', {'hyperparameters': {}}),
        ]:
            body = {'model_type': 'fleet_random_forest',
                    'dataset_path': data['partitions']['train']['path'],
                    'hyperparameters': {'target_column': 'label'}, **patch}
            response = requests.post(f'{GATEWAY}/api/v1/jobs', json=body, timeout=15)
            assert response.status_code == 400, response.text
            report['checks'].append({'name': name, 'passed': True, 'response': response.json()})
        state, status = failed_job('datasets/mnist/does-not-exist.npz')
        assert status == 404
        report['checks'].append({'name': 'missing_dataset_fails_without_success_manifest', 'passed': True, 'job_state': state})

        # Stop the actual storage service, submit a real job, and restore it
        # regardless of the outcome. Do not interfere with model evaluation.
        subprocess.run(['docker', 'stop', 'tensorfleet-storage'], check=True, capture_output=True)
        try:
            response = requests.post(f'{GATEWAY}/api/v1/jobs', json={
                'model_type': 'fleet_random_forest', 'dataset_path': data['partitions']['train']['path'],
                'hyperparameters': {'target_column': 'label'}}, timeout=15)
            response.raise_for_status()
            job_id = response.json()['job_id']
            def unavailable_failure():
                state = requests.get(f'{GATEWAY}/api/v1/jobs/{job_id}', timeout=10).json()
                return state if state['status'] in ('FAILED', 'COMPLETED') else None
            state = wait_for(unavailable_failure)
            assert state['status'] == 'FAILED' and state['completed_tasks'] == 0, state
        finally:
            subprocess.run(['docker', 'start', 'tensorfleet-storage'], check=True, capture_output=True)
            wait_for(lambda: requests.get(f'{STORAGE}/health', timeout=5).status_code == 200)
        response = requests.get(f'{STORAGE}/api/v1/download/artifacts/fleet/{job_id}/result.json', timeout=10)
        assert response.status_code == 404
        report['checks'].append({'name': 'storage_outage_fails_without_success_manifest', 'passed': True, 'job_state': state})

        subprocess.run(['docker', 'restart', 'tensorfleet-minio'], check=True, capture_output=True)
        wait_for(lambda: requests.get(f'{STORAGE}/health', timeout=5).status_code == 200)
        objects = []
        for partition in data['partitions'].values():
            objects.append((partition['path'], partition['sha256']))
        for job in run['jobs']:
            manifest_file = OUTPUT / f'{job["job_id"]}-manifest.json'
            manifest = json.loads(manifest_file.read_text())
            objects.append((manifest['artifact_uri'].removeprefix('s3://'), manifest['artifact_sha256']))
            # Compare the manifest parsed semantically: the saved local copy
            # is pretty-printed, while the uploaded manifest is compact JSON.
            response = requests.get(f'{STORAGE}/api/v1/download/artifacts/fleet/{job["job_id"]}/result.json', timeout=15)
            response.raise_for_status()
            assert response.json() == manifest
        for path, checksum in objects:
            response = requests.get(f'{STORAGE}/api/v1/download/{path}', timeout=120)
            response.raise_for_status()
            assert hashlib.sha256(response.content).hexdigest() == checksum, path
        report['checks'].append({'name': 'minio_restart_preserves_datasets_models_and_manifests',
                                 'passed': True, 'checksummed_objects': len(objects), 'manifests': len(run['jobs'])})
        report['status'] = 'passed'
        print('PASS: invalid inputs, missing dataset, storage outage, and MinIO restart persistence.')
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        (OUTPUT / 'deployment-checks.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
