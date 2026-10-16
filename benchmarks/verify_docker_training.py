"""Post-refactor Docker model equivalence and durable-admission regression."""
import hashlib
import json
import pickle
import subprocess
import time

import numpy as np
import pandas as pd
import requests
from experiment_utils import ROOT


def main():
    output=ROOT/'result/research/post-refactor-docker.json'
    report={'status':'running','checks':[]}
    try:
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            try:
                response=requests.get('http://localhost:8080/health',timeout=2)
                if response.ok:break
            except requests.RequestException:pass
            time.sleep(.2)
        data=json.loads((ROOT/'result/mnist-data.json').read_text())
        request={'model_type':'fleet_distributed_random_forest','epochs':1,'num_workers':2,'dataset_path':data['partitions']['train']['path'],
                 'hyperparameters':{'target_column':'label','random_state':'42','validation_dataset_path':data['partitions']['validation']['path'],'test_dataset_path':data['partitions']['test']['path'],'estimator_params':json.dumps({'n_estimators':40,'max_depth':20,'max_features':'sqrt'})}}
        response=requests.post('http://localhost:8080/api/v1/jobs',json=request,timeout=30);response.raise_for_status()
        report.update(request=request,submission=response.json());job_id=response.json()['job_id']
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            response=requests.get(f'http://localhost:8080/api/v1/jobs/{job_id}',timeout=15);response.raise_for_status();state=response.json()
            if state['status'] in ('COMPLETED','FAILED','CANCELLED'):break
            time.sleep(.2)
        report['state']=state
        assert state['status']=='COMPLETED' and state['completed_tasks']==3,state
        response=requests.get('http://localhost:8081/api/v1/download/'+state['message'].removeprefix('s3://'),timeout=30);response.raise_for_status();manifest=response.json()
        response=requests.get('http://localhost:8081/api/v1/download/'+manifest['artifact_uri'].removeprefix('s3://'),timeout=60);response.raise_for_status();raw=response.content
        assert hashlib.sha256(raw).hexdigest()==manifest['artifact_sha256']
        model=pickle.loads(raw)
        reference=pickle.loads((ROOT/'result/20d5a59d-9236-4293-bd13-d469df855062-model.pkl').read_bytes())
        response=requests.get('http://localhost:8081/api/v1/download/'+data['partitions']['test']['path'],timeout=60);response.raise_for_status()
        import io
        with np.load(io.BytesIO(response.content),allow_pickle=False) as partition:
            X=pd.DataFrame(partition['X'],columns=manifest['features'])
        np.testing.assert_allclose(model.predict_proba(X),reference.predict_proba(X),rtol=0,atol=1e-14)
        report['manifest']=manifest
        report['checks'].append({'name':'all_10000_test_probabilities_match_original','passed':True})
        try:
            subprocess.run(['docker','pause','tensorfleet-redis'],check=True,capture_output=True)
            response=requests.post('http://localhost:8080/api/v1/jobs',json={'model_type':'fleet_decision_tree','dataset_path':'datasets/research/iris.csv','hyperparameters':{'target_column':'target'}},timeout=15)
            report['checks'].append({'name':'redis_unavailable_durable_admission','status_code':response.status_code,'response':response.json(),'passed':response.status_code==503})
            assert response.status_code==503,response.text
        finally:
            subprocess.run(['docker','unpause','tensorfleet-redis'],check=True,capture_output=True)
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=str(error));raise
    finally:
        output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
