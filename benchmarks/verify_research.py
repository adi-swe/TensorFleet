"""Repeatable post-refactor checks; records failures and command logs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'result/research'


def main():
    report = {'status': 'running', 'checks': [], 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'worker/training').glob('*.py'))}}
    OUTPUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tensorfleet-portability-') as temporary:
        checks = [
            ('python-regressions', ROOT, [sys.executable,'-m','unittest','discover','-s','worker/tests','-p','test_*.py','-v'], {}),
            *[(f'{name}-race',ROOT/name,['go','test','-race','./...'],{}) for name in ('orchestrator','worker','api-gateway')],
            ('frontend-build',ROOT/'frontend',['npm','run','build'],{}),
            ('native-integration',ROOT,[sys.executable,'benchmarks/smoke_real_fleet.py'],{}),
            ('native-capabilities',ROOT,[sys.executable,'worker/fleet_trainer.py','--capabilities'],{}),
            *[(f'worker-{system}-compile',ROOT/'worker',['go','build','-o',str(Path(temporary)/f'worker-{system}'),'.'],{'GOOS':system,'GOARCH':'arm64','CGO_ENABLED':'0'}) for system in ('linux','windows')],
        ]
        try:
            for name,cwd,command,extra in checks:
                start=time.perf_counter()
                row={'name':name,'command':command,'cwd':str(cwd.relative_to(ROOT)),'environment_overrides':extra}
                report['checks'].append(row)
                with (OUTPUT/f'{name}.log').open('w') as log:
                    result=subprocess.run(command,cwd=cwd,env={**os.environ,**extra},stdout=log,stderr=subprocess.STDOUT)
                row.update(exit_code=result.returncode,duration_s=time.perf_counter()-start,passed=result.returncode==0)
                print(name,result.returncode,flush=True)
                (OUTPUT/'regression.json').write_text(json.dumps(report,indent=2))
                if result.returncode:raise RuntimeError(f'{name} failed; inspect {OUTPUT/name}.log')
            report['status']='passed'
        except BaseException as error:
            report.update(status='failed',error=str(error));raise
        finally:
            (OUTPUT/'regression.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
