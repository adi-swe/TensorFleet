#!/usr/bin/env python3
"""
Worker-fleet scaling benchmark.

Submits a fixed batch of training jobs through the API gateway, waits for
all of them to complete, and measures the makespan and aggregate task
throughput. Run it once per worker-fleet size, then feed the JSON outputs
to report_scaling.py to compute speedup and parallel efficiency.

Usage:
    python3 benchmark_scaling.py --gateway http://localhost:8080 \
        --workers 4 --jobs 6 --epochs 2 --output results/scaling_w4.json
"""
import argparse
import json
import sys
import time
import urllib.request


def api(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-User-ID', 'benchmark')
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gateway', default='http://localhost:8080')
    parser.add_argument('--workers', type=int, required=True,
                        help='Required number of workers that actually complete tasks')
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=2,
                        help='Epochs per job (10 tasks per epoch)')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Max seconds to wait for completion')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    if min(args.jobs, args.epochs, args.workers, args.timeout) < 1:
        parser.error('jobs, epochs, workers, and timeout must be positive')
    errors = []
    try:
        before = api('GET', f'{args.gateway}/worker-activity')
    except Exception as error:
        errors.append({'phase': 'initial_worker_activity', 'error': str(error)})
        before = {'workers': []}
    counts_before = {worker['worker_id']: worker['tasks_completed'] for worker in before['workers']}

    total_tasks = args.jobs * args.epochs * 10  # orchestrator: 10 batches/epoch

    print(f'Submitting {args.jobs} jobs x {args.epochs} epochs '
          f'({total_tasks} tasks) against a {args.workers}-worker fleet...')

    start = time.perf_counter()
    job_ids = []
    for i in range(args.jobs):
        try:
            resp = api('POST', f'{args.gateway}/api/v1/jobs', {
                'model_type': 'benchmark',
                'dataset_path': 'datasets/benchmark.csv',
                'num_workers': args.workers,
                'epochs': args.epochs,
                'hyperparameters': {'job_name': f'scaling-w{args.workers}-job{i}'},
            })
            job_ids.append(resp['job_id'])
        except Exception as error:
            errors.append({'phase': 'submission', 'index': i, 'error': str(error)})


    submit_time = time.perf_counter() - start
    print(f'  Submitted in {submit_time:.2f}s. Waiting for completion...')

    pending = set(job_ids)
    final_states = {}
    first_completion = None
    while pending:
        if time.perf_counter() - start > args.timeout:
            print(f'TIMEOUT: {len(pending)} jobs still pending', file=sys.stderr)
            break
        time.sleep(2)
        for job_id in list(pending):
            try:
                status = api('GET', f'{args.gateway}/api/v1/jobs/{job_id}')
            except Exception as error:
                errors.append({'phase': 'poll', 'job_id': job_id, 'error': str(error)})
                continue
            final_states[job_id] = status
            if status['status'] in ('COMPLETED', 'FAILED', 'CANCELLED'):
                pending.discard(job_id)
                if first_completion is None and status['status'] == 'COMPLETED':
                    first_completion = time.perf_counter() - start
        done = len(job_ids) - len(pending)
        print(f'  {done}/{len(job_ids)} jobs complete '
              f'({time.perf_counter() - start:.0f}s elapsed)', flush=True)

    makespan = time.perf_counter() - start
    try:
        after = api('GET', f'{args.gateway}/worker-activity')
    except Exception as error:
        errors.append({'phase': 'final_worker_activity', 'error': str(error)})
        after = {'workers': []}
    observed = sorted(worker['worker_id'] for worker in after['workers']
                      if worker['tasks_completed'] > counts_before.get(worker['worker_id'], 0))
    counts = {state: sum(job['status'] == state for job in final_states.values())
              for state in ('COMPLETED', 'FAILED', 'CANCELLED')}
    actual_completed_tasks = sum(job['completed_tasks'] for job in final_states.values())
    valid = counts['COMPLETED'] == args.jobs and len(observed) == args.workers and not pending and not errors
    result = {
        'schema_version': 2,
        'workload_kind': 'simulated_orchestration',
        'valid': valid,
        'observed_worker_ids': observed,
        'job_states': final_states,
        'errors': errors,
        'submission_failures': args.jobs - len(job_ids),
        'failed_jobs': counts['FAILED'],
        'cancelled_jobs': counts['CANCELLED'],
        'unfinished_jobs': args.jobs - sum(counts.values()),
        'actual_completed_tasks': actual_completed_tasks,
        'workers': args.workers,
        'jobs': args.jobs,
        'epochs_per_job': args.epochs,
        'total_tasks': total_tasks,
        'submit_time_s': round(submit_time, 2),
        'first_job_completed_s': round(first_completion or 0, 2),
        'makespan_s': round(makespan, 2),
        'tasks_per_second': round(actual_completed_tasks / makespan, 3),
        'completed_jobs': counts['COMPLETED'],
        'timed_out': bool(pending),
    }
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    if not valid:
        sys.exit('Invalid scaling trial: incomplete workload or participating worker mismatch')


if __name__ == '__main__':
    main()
