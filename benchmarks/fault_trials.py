"""Real worker/coordinator failure checks; run after the scaling sweep.

Uses the local test deployment only, restores containers/settings in finally,
and keeps every state/error. These are individual fault injections, not uptime.
"""
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import requests

from experiment_utils import compose, ROOT


def wait(check, timeout=120):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            value = check()
            if value: return value
        except requests.RequestException: pass
        time.sleep(.2)
    raise TimeoutError('Fault experiment did not reach the expected state')


def main():
    output = ROOT/'result/research'
    output.mkdir(exist_ok=True)
    data = json.loads((ROOT/'result/mnist-data.json').read_text())
    override = output/'fault.override.yml'
    override.write_text('services:\n  orchestrator:\n    environment:\n      TASK_LEASE_TIMEOUT_SECONDS: "20"\n')
    report = {'status':'running','lease_timeout_s':20,'reaper_interval_s':30,'trials':[]}
    def configure():
        return compose('-f',str(override),'up','-d','--no-build','orchestrator')
    def state(job_id):
        response=requests.get(f'http://localhost:8080/api/v1/jobs/{job_id}',timeout=10)
        response.raise_for_status()
        return response.json()
    request={'model_type':'fleet_distributed_random_forest','epochs':1,'num_workers':2,
             'dataset_path':data['partitions']['train']['path'],'hyperparameters':{
                 'target_column':'label','random_state':'42',
                 'validation_dataset_path':data['partitions']['validation']['path'],
                 'test_dataset_path':data['partitions']['test']['path'],
                 'estimator_params':json.dumps({'n_estimators':40,'max_depth':20,'max_features':'sqrt'})}}
    try:
        configure()
        known=json.loads((output/'distributed-job.json').read_text())['submission']['job_id']
        wait(lambda: state(known).get('status'))
        for mode in ('no_fault','worker_kill','coordinator_kill'):
            started=time.perf_counter()
            response=requests.post('http://localhost:8080/api/v1/jobs',json=request,timeout=30)
            response.raise_for_status()
            created=response.json();job_id=created['job_id']
            row={'fault':mode,'submission':created,'request':request}
            report['trials'].append(row)
            def assigned():
                response=requests.get('http://localhost:8080/worker-activity',timeout=10)
                response.raise_for_status()
                workers=[w for w in response.json()['workers'] if w['current_job_id']==job_id and w['status']=='BUSY']
                return workers if len(workers)==2 else None
            row['assigned_workers']=wait(assigned)
            if mode != 'no_fault':
                row['fault_injected_offset_s']=time.perf_counter()-started
                container='tensorfleet-gap1-worker-1' if mode=='worker_kill' else 'tensorfleet-orchestrator'
                subprocess.run(['docker','kill',container],check=True,capture_output=True)
                subprocess.run(['docker','start',container],check=True,capture_output=True)
            terminal=wait(lambda: (s if (s:=state(job_id))['status'] in ('COMPLETED','FAILED','CANCELLED') else None),timeout=150)
            row.update(final_state=terminal,duration_s=time.perf_counter()-started)
            assert terminal['status']=='COMPLETED' and terminal['completed_tasks']==3, terminal
            response=requests.get('http://localhost:8081/api/v1/download/'+terminal['message'].removeprefix('s3://'),timeout=30)
            response.raise_for_status();manifest=response.json()
            response=requests.get('http://localhost:8081/api/v1/download/'+manifest['artifact_uri'].removeprefix('s3://'),timeout=60)
            response.raise_for_status();raw=response.content
            assert hashlib.sha256(raw).hexdigest()==manifest['artifact_sha256']
            import pickle
            import numpy as np
            import pandas as pd
            model=pickle.loads(raw)
            reference=pickle.loads((ROOT/'result/20d5a59d-9236-4293-bd13-d469df855062-model.pkl').read_bytes())
            probe=pd.DataFrame(manifest['verification']['rows'])[manifest['features']]
            np.testing.assert_allclose(model.predict_proba(probe),reference.predict_proba(probe),rtol=0,atol=1e-14)
            row.update(artifact_verified=True,manifest=manifest)
            if mode!='no_fault': row['makespan_penalty_s']=row['duration_s']-report['trials'][0]['duration_s']
            print(mode,round(row['duration_s'],3),flush=True)
            (output/'fault-trials.json').write_text(json.dumps(report,indent=2))
        # Admission backpressure must be explicit and must not enqueue a partial job.
        response=requests.post('http://localhost:8080/api/v1/jobs',json={
            'model_type':'benchmark','dataset_path':'datasets/control.csv','epochs':101},timeout=15)
        assert response.status_code==429,response.text
        report['backpressure']={'status_code':response.status_code,'retry_after':response.headers.get('Retry-After'),
                                'response':response.json()}
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        subprocess.run(['docker','start','tensorfleet-gap1-worker-1'],capture_output=True)
        subprocess.run(['docker','start','tensorfleet-orchestrator'],capture_output=True)
        compose('up','-d','--no-build','--scale','worker=2','orchestrator','worker')
        (output/'fault-trials.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
