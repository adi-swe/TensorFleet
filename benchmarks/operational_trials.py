"""Local polling/slot ablation, sustained reads, and storage integrity checks.

Runs sequentially against the isolated result/ deployment; preserves raw trials.
"""
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from experiment_utils import ROOT, bootstrap, compose

OUT = ROOT / 'result/research'


def get_json(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def ablation():
    override = OUT / 'ablation.override.yml'
    record = {'status': 'running', 'scope': 'Two CPU workers, four Iris shards, 24 trees; one factor grid; startup excluded', 'trials': []}
    cases = [(poll, slots, trial) for poll in (100, 500, 1000) for slots in (1, 2) for trial in range(5)]
    random.Random(2026).shuffle(cases)
    try:
        for poll, slots, trial in cases:
            override.write_text(f'services:\n  worker:\n    environment:\n      FETCH_INTERVAL_MS: "{poll}"\n      MAX_CONCURRENT_TASKS: "{slots}"\n')
            compose('-f', str(override), 'up', '-d', '--no-build', '--scale', 'worker=2', 'worker')
            # Wait for both new worker registrations, then let their first fetch finish.
            ids = subprocess.check_output(['docker', 'ps', '-q', '--filter', 'label=com.docker.compose.project=tensorfleet-gap1', '--filter', 'label=com.docker.compose.service=worker'], text=True).split()
            for container in ids:
                deadline = time.monotonic()+30
                while 'Trainer capabilities:' not in subprocess.check_output(['docker', 'logs', container], stderr=subprocess.STDOUT, text=True):
                    if time.monotonic()>deadline: raise TimeoutError('Worker readiness')
                    time.sleep(.1)
            time.sleep(1.1)
            request = {'model_type': 'fleet_distributed_random_forest', 'epochs': 1, 'num_workers': 4,
                       'dataset_path': 'datasets/research/iris.csv', 'hyperparameters': {'target_column': 'target', 'random_state': str(42+trial), 'estimator_params': json.dumps({'n_estimators':24, 'max_depth':16, 'max_features':'sqrt'})}}
            row = {'poll_ms':poll, 'slots':slots, 'trial':trial, 'request':request, 'valid':False}
            record['trials'].append(row)
            start=time.perf_counter()
            response=requests.post('http://localhost:8080/api/v1/jobs',json=request,timeout=30);response.raise_for_status()
            row['submission']=response.json();job_id=row['submission']['job_id']
            while time.perf_counter()-start<120:
                state=get_json(f'http://localhost:8080/api/v1/jobs/{job_id}')
                if state['status'] in ('COMPLETED','FAILED','CANCELLED'):break
                time.sleep(.05)
            row['state']=state
            if state['status']!='COMPLETED' or state['completed_tasks']!=5:raise RuntimeError(state)
            manifest=get_json('http://localhost:8081/api/v1/download/'+state['message'].removeprefix('s3://'))
            response=requests.get('http://localhost:8081/api/v1/download/'+manifest['artifact_uri'].removeprefix('s3://'),timeout=60);response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest()!=manifest['artifact_sha256']:raise ValueError('Checksum mismatch')
            row.update(valid=True,duration_s=time.perf_counter()-start,manifest=manifest)
            (OUT/'ablation.json').write_text(json.dumps(record,indent=2))
            print('ablation',poll,slots,trial,row['duration_s'],flush=True)
        record['summary']=[]
        for poll in (100,500,1000):
            for slots in (1,2):
                values=[r['duration_s'] for r in record['trials'] if r['poll_ms']==poll and r['slots']==slots]
                record['summary'].append({'poll_ms':poll,'slots':slots,'mean_s':statistics.mean(values),'sample_stdev_s':statistics.stdev(values),'bootstrap_95pct_mean_s':bootstrap(values)})
        record['status']='passed'
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:
        compose('up','-d','--no-build','--scale','worker=2','worker')
        (OUT/'ablation.json').write_text(json.dumps(record,indent=2))


def operations():
    known=json.loads((OUT/'distributed-job.json').read_text())['submission']['job_id']
    for concurrency in (1,4,16):
        subprocess.run([sys.executable,str(ROOT/'benchmarks/benchmark_gateway.py'),'--duration-seconds','10','--concurrency',str(concurrency),'--skip-submissions','--job-id',known,'--output',str(OUT/f'gateway-c{concurrency}.json')],check=True)
    manifest=json.loads((OUT/'distributed-manifest.json').read_text())
    def download(_):
        start=time.perf_counter();digest=hashlib.sha256();size=0
        with requests.get('http://localhost:8081/api/v1/download/'+manifest['artifact_uri'].removeprefix('s3://'),stream=True,timeout=60) as response:
            response.raise_for_status()
            for chunk in response.iter_content(65536):digest.update(chunk);size+=len(chunk)
        return {'duration_s':time.perf_counter()-start,'bytes':size,'sha256':digest.hexdigest(),'valid':digest.hexdigest()==manifest['artifact_sha256']}
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(download,range(20)))
    result={'status':'passed' if all(r['valid'] for r in rows) else 'failed','concurrency':4,'downloads':rows,'wall_s':time.perf_counter()-start,'scope':'Model download integrity/throughput; client streaming, no server RSS bound claim'}
    (OUT/'storage-concurrency.json').write_text(json.dumps(result,indent=2))
    if result['status']!='passed':raise ValueError('Storage integrity')


if __name__=='__main__':
    ablation()
    operations()
