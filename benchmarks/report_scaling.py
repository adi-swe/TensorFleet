#!/usr/bin/env python3
"""
Aggregate benchmark_scaling.py outputs into a speedup/efficiency table.

Usage:
    python3 report_scaling.py results/scaling_w*.json
"""
import json
import sys


def main():
    if len(sys.argv) < 2:
        print('Usage: report_scaling.py <scaling_w*.json> ...', file=sys.stderr)
        sys.exit(1)

    def read_run(path):
        with open(path) as source: return json.load(source)

    runs = sorted(
        (read_run(path) for path in sys.argv[1:]),
        key=lambda r: r['workers'],
    )
    if not runs or runs[0]['workers'] != 1:
        raise ValueError('A valid one-worker baseline is required')
    signature = lambda r: (r['jobs'], r['epochs_per_job'], r['total_tasks'], r.get('workload_kind'))
    for run in runs:
        if (run.get('schema_version') != 2 or run.get('valid') is not True
                or run.get('timed_out') or run.get('failed_jobs') or run.get('cancelled_jobs')
                or run.get('unfinished_jobs') or run['completed_jobs'] != run['jobs']
                or run.get('actual_completed_tasks') != run['total_tasks']
                or len(run.get('observed_worker_ids', [])) != run['workers']
                or run['makespan_s'] <= 0 or signature(run) != signature(runs[0])):
            raise ValueError('Cannot compare invalid or mismatched scaling trials')
    baseline = runs[0]

    print(f"\nWorkload: {baseline['jobs']} jobs x {baseline['epochs_per_job']} "
          f"epochs = {baseline['total_tasks']} tasks "
          f"(baseline: {baseline['workers']} worker(s))\n")
    header = (f"{'Workers':>8} | {'Makespan (s)':>13} | {'Tasks/s':>8} | "
              f"{'Speedup':>8} | {'Efficiency':>10}")
    print(header)
    print('-' * len(header))
    for run in runs:
        speedup = baseline['makespan_s'] / run['makespan_s']
        efficiency = speedup / (run['workers'] / baseline['workers'])
        print(f"{run['workers']:>8} | {run['makespan_s']:>13.1f} | "
              f"{run['tasks_per_second']:>8.2f} | {speedup:>7.2f}x | "
              f"{efficiency * 100:>9.1f}%")
    print()


if __name__ == '__main__':
    main()
