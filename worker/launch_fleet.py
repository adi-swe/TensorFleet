"""Launch a resource-budgeted native fleet, including native Apple MPS access.

Runs on macOS, Linux, and Windows with Go and the appropriate Python deps.
Worker count is computed from declared dataset dimensions and a memory budget.
This provisions local processes, not cloud machines or Kubernetes nodes.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

from resource_plan import plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=('fleet_distributed_random_forest', 'fleet_accelerated_logistic'), required=True)
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--features', type=int, required=True)
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--memory-gib', type=float)
    parser.add_argument('--orchestrator', default='localhost:50051')
    parser.add_argument('--storage', default='http://localhost:8081')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    allocation = plan(args.algorithm, args.rows, args.features, args.trees, args.max_workers,
                      None if args.memory_gib is None else int(args.memory_gib * 1024**3))
    print(json.dumps(allocation, indent=2), flush=True)
    if args.plan_only: return
    root = Path(__file__).resolve().parent
    processes = []
    with tempfile.TemporaryDirectory(prefix='tensorfleet-native-') as directory:
        binary = str(Path(directory) / ('worker.exe' if os.name == 'nt' else 'worker'))
        subprocess.run(['go', 'build', '-o', binary, '.'], cwd=root, check=True)
        try:
            for _ in range(allocation['workers']):
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = str(sock.getsockname()[1])
                processes.append(subprocess.Popen([binary], cwd=root, env={
                    **os.environ, 'PORT': port, 'METRICS_PORT': '0', 'MAX_CONCURRENT_TASKS': '1',
                    'TRAINING_PYTHON': sys.executable, 'FLEET_TRAINER_PATH': str(root/'fleet_trainer.py'),
                    'ORCHESTRATOR_ADDR': args.orchestrator, 'STORAGE_SERVICE_URL': args.storage,
                    'FETCH_INTERVAL_MS': '100',
                    'TENSORFLEET_WORKER_ROLE': 'cpu' if allocation['device'] == 'cpu' else 'accelerator',
                }))
            print('Fleet running; Ctrl+C stops the local workers. Submit num_workers equal to the planned forest shard count.', flush=True)
            for process in processes:
                if process.wait() != 0: raise RuntimeError('A native worker exited unexpectedly')
        except KeyboardInterrupt:
            pass
        finally:
            for process in processes:
                if process.poll() is None: process.terminate()
            for process in processes:
                try: process.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    main()
