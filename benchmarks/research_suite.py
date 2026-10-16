"""Repeated real-ML scaling and direct process-pool baselines on local Docker.

Run with the result/ Compose test stack running. Raw failed trials are retained.
All forests use the same per-tree seeds across partition counts. Datasets are
replicated per worker: this measures tree parallelism, not out-of-core training.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np
import requests

from experiment_utils import ROOT, bootstrap, compose
sys.path.insert(0, str(ROOT / 'worker'))
from fleet_trainer import ObjectStore, train_job


def direct_shard(payload):
    return train_job(payload, ObjectStore(os.environ.get('STORAGE_SERVICE_URL', 'http://storage:8081')))


def direct(args):
    request = json.loads(args.request)
    base = {'job_id': 'direct-'+str(uuid.uuid4()), 'model_type': 'fleet_distributed_random_forest',
            'dataset_path': request['dataset_path'], 'worker_id': 'direct-parent'}
    started = time.perf_counter()
    payloads = []
    for index in range(args.workers):
        payloads.append({**base, 'task_id': f'shard-{index}', 'worker_id': f'direct-process-{index}',
                        'hyperparameters': {**request['hyperparameters'], '_fleet_stage': 'shard',
                                           '_fleet_shards': str(args.workers), '_fleet_shard_index': str(index)}})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(direct_shard, payloads))
    manifest = direct_shard({**base, 'task_id': 'aggregate', 'hyperparameters': {
        **request['hyperparameters'], '_fleet_stage': 'aggregate', '_fleet_shards': str(args.workers)}})
    print(json.dumps({'duration_s': time.perf_counter()-started, 'manifest': manifest}))


def run(args):
    from sklearn.datasets import load_iris, load_wine
    output = ROOT / 'result/research'
    output.mkdir(exist_ok=True)
    data = json.loads((ROOT/'result/mnist-data.json').read_text())
    datasets = {'mnist': {name: partition['path'] for name, partition in data['partitions'].items()}}
    for name, loader in [('iris', load_iris), ('wine', load_wine)]:
        frame = loader(as_frame=True).frame
        path = f'datasets/research/{name}.csv'
        content = frame.to_csv(index=False).encode()
        response = requests.post(f'http://localhost:8081/api/v1/upload/{path}', files={'file': (name+'.csv', content)}, timeout=30)
        response.raise_for_status()
        datasets[name] = {'train': path}
    record = {'schema_version': 1, 'status': 'running', 'trials_per_configuration': args.trials,
              'workers': args.workers, 'datasets': datasets, 'tree_count': args.trees,
              'max_depth': 16, 'case_order_seed': 2026, 'host': platform.platform(),
              'scope': 'Single-host CPU tree-parallel training; shared hardware, replicated data, no cross-node claim',
              'timing_scope': 'Submission to downloaded/checksummed model and reload prediction probe; excludes fleet startup and dataset upload',
              'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'worker/fleet_trainer.py', *sorted((ROOT/'worker/training').glob('*.py')), ROOT/'orchestrator/cmd/orchestrator/main.go', ROOT/'orchestrator/cmd/orchestrator/distributed_training.go', ROOT/'orchestrator/cmd/orchestrator/recovery.go', Path(__file__).resolve()]},
              'trials': []}
    path = output/'classical-scaling.json'
    cases = [(dataset, workers, trial, mode) for dataset in args.datasets for workers in args.workers
             for trial in range(args.trials) for mode in ('tensorfleet', 'direct_process_pool')]
    random.Random(2026).shuffle(cases)
    record['case_order'] = cases
    path.write_text(json.dumps(record, indent=2))
    try:
        for dataset, workers, trial, mode in cases:
            compose('up', '-d', '--no-build', '--scale', f'worker={workers}', 'worker')
            ids = subprocess.check_output(['docker','ps','-q','--filter','label=com.docker.compose.project=tensorfleet-gap1',
                                           '--filter','label=com.docker.compose.service=worker'], text=True).split()
            for container in ids:
                for _ in range(100):
                    logs = subprocess.check_output(['docker','logs',container], stderr=subprocess.STDOUT, text=True)
                    if 'Trainer capabilities:' in logs: break
                    time.sleep(.1)
                else: raise RuntimeError('Worker trainer did not become ready')
            if mode == 'direct_process_pool':
                # The direct baseline gets the same aggregate CPU/memory budget;
                # idle fleet workers remain running but execute no jobs.
                pass
            parameters = {'target_column': 'label' if dataset == 'mnist' else 'target', 'random_state': str(42+trial),
                          'estimator_params': json.dumps({'n_estimators': args.trees, 'max_depth': 16, 'max_features': 'sqrt'})}
            if dataset == 'mnist':
                parameters.update(validation_dataset_path=datasets[dataset]['validation'], test_dataset_path=datasets[dataset]['test'])
            request = {'model_type': 'fleet_distributed_random_forest', 'epochs': 1, 'num_workers': workers,
                       'dataset_path': datasets[dataset]['train'], 'hyperparameters': parameters}
            row = {'dataset': dataset, 'workers': workers, 'trial': trial, 'mode': mode, 'request': request, 'valid': False}
            record['trials'].append(row)
            started = time.perf_counter()
            try:
                if mode == 'tensorfleet':
                    response = requests.post('http://localhost:8080/api/v1/jobs', json=request, timeout=30)
                    response.raise_for_status()
                    row['submission'] = response.json()
                    job_id = row['submission']['job_id']
                    while time.perf_counter()-started < 240:
                        state = requests.get(f'http://localhost:8080/api/v1/jobs/{job_id}', timeout=15).json()
                        if state['status'] in ('COMPLETED','FAILED','CANCELLED'): break
                        time.sleep(.1)
                    row['state'] = state
                    if state['status'] != 'COMPLETED' or state['completed_tasks'] != workers+1: raise RuntimeError(f'Incomplete workload: {state}')
                    response = requests.get('http://localhost:8081/api/v1/download/'+state['message'].removeprefix('s3://'), timeout=30)
                    response.raise_for_status()
                    manifest = response.json()
                    if len(manifest['distribution']['worker_ids']) != workers: raise RuntimeError('Participating worker mismatch')
                else:
                    completed = subprocess.run(['docker','run','--rm','--network','tensorfleet-gap1_tensorfleet-net',
                        '--cpus',str(workers),'--memory',f'{2*workers}g','-v',f'{ROOT}:/source:ro',
                        'tensorfleet-gap1-worker','python3','/source/benchmarks/research_suite.py','direct',
                        '--workers',str(workers),'--request',json.dumps(request)], check=True, capture_output=True, text=True)
                    direct_result = json.loads(completed.stdout)
                    manifest = direct_result['manifest']
                    row['direct_internal_duration_s'] = direct_result['duration_s']
                response = requests.get('http://localhost:8081/api/v1/download/'+manifest['artifact_uri'].removeprefix('s3://'),timeout=60)
                response.raise_for_status()
                raw = response.content
                if hashlib.sha256(raw).hexdigest() != manifest['artifact_sha256']: raise ValueError('Model checksum mismatch')
                import pickle
                import pandas as pd
                model = pickle.loads(raw)
                probe = pd.DataFrame(manifest['verification']['rows'])[manifest['features']]
                if model.predict(probe).tolist() != manifest['verification']['predictions']: raise ValueError('Reload predictions differ')
                row.update(valid=True, duration_s=time.perf_counter()-started, manifest=manifest,
                           tree_seed_fingerprint=hashlib.sha256(json.dumps([tree.random_state for tree in model['model'].estimators_]).encode()).hexdigest())
                print(dataset, workers, trial, mode, round(row['duration_s'],3), flush=True)
            except Exception as error:
                row.update(error=str(error), duration_s=time.perf_counter()-started)
                raise
            finally:
                path.write_text(json.dumps(record, indent=2))
        # Every matched workload must build exactly the same global tree seeds.
        for dataset in args.datasets:
            for trial in range(args.trials):
                fingerprints = {row['tree_seed_fingerprint'] for row in record['trials'] if row['dataset']==dataset and row['trial']==trial}
                if len(fingerprints) != 1: raise ValueError('Mismatched forests across resource configurations')
        summaries = []
        for dataset in args.datasets:
            for workers in args.workers:
                for mode in ('tensorfleet','direct_process_pool'):
                    rows = [row for row in record['trials'] if row['dataset']==dataset and row['workers']==workers and row['mode']==mode]
                    values = [row['duration_s'] for row in rows]
                    summaries.append({'dataset':dataset,'workers':workers,'mode':mode,'trials':len(values),
                        'mean_duration_s':statistics.mean(values),'median_duration_s':statistics.median(values),
                        'sample_stdev_s':statistics.stdev(values), 'bootstrap_95pct_mean_duration_s':bootstrap(values)})
        record.update(status='passed',summary=summaries)
    except BaseException as error:
        record.update(status='failed',error=str(error))
        raise
    finally:
        path.write_text(json.dumps(record, indent=2))
        compose('up','-d','--no-build','--scale','worker=2','worker')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    baseline = sub.add_parser('direct')
    baseline.add_argument('--workers',type=int,required=True)
    baseline.add_argument('--request',required=True)
    sweep = sub.add_parser('run')
    sweep.add_argument('--workers',nargs='+',type=int,default=[1,2,4])
    sweep.add_argument('--trials',type=int,default=5)
    sweep.add_argument('--trees',type=int,default=24)
    sweep.add_argument('--datasets',nargs='+',choices=('mnist','iris','wine'),default=['mnist','iris','wine'])
    arguments = parser.parse_args()
    if arguments.command == 'direct': direct(arguments)
    else:
        if arguments.trials < 2 or min(arguments.workers)<1 or max(arguments.workers)>arguments.trees:
            parser.error('Need at least two trials and positive worker counts no larger than tree count')
        run(arguments)
