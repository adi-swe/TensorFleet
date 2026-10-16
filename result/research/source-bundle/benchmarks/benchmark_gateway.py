#!/usr/bin/env python3
"""
API Gateway latency/throughput benchmark.

Fires concurrent requests at the gateway's hot endpoints and reports
throughput (req/s) plus latency percentiles (p50/p90/p99). Uses only the
Python standard library so it runs anywhere.

Usage:
    python3 benchmark_gateway.py --gateway http://localhost:8080 \
        --concurrency 16 --requests 400 --output results/gateway.json
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def do_request(method, url, body=None):
    """Returns (latency_seconds, ok)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-User-ID', 'benchmark')
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
            ok = 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        ok = False
    return time.perf_counter() - start, ok


def run_scenario(name, method, url, body, concurrency, total_requests, duration_s=0):
    print(f'  {name}: {total_requests} requests @ concurrency {concurrency} ...',
          flush=True)
    latencies, successful_latencies, errors = [], [], 0
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while True:
            futures = [pool.submit(do_request, method, url, body) for _ in range(total_requests)]
            for future in futures:
                latency, ok = future.result()
                latencies.append(latency)
                if ok:
                    successful_latencies.append(latency)
                else:
                    errors += 1
            if not duration_s or time.perf_counter() - wall_start >= duration_s:
                break
    total_requests = len(latencies)
    wall = time.perf_counter() - wall_start

    latencies.sort()
    result = {
        'scenario': name,
        'load_model': 'closed_loop_fixed_client_concurrency',
        'latency_samples_ms': [round(value * 1000, 4) for value in latencies],
        'successful_latency_ms': {
            'p50': round(percentile(sorted(successful_latencies), 50)*1000, 2) if successful_latencies else None,
            'p99': round(percentile(sorted(successful_latencies), 99)*1000, 2) if successful_latencies else None,
        },
        'method': method,
        'url': url,
        'concurrency': concurrency,
        'requests': total_requests,
        'errors': errors,
        'error_rate': errors / total_requests,
        'throughput_rps': round(total_requests / wall, 1),
        'attempted_requests_per_second': round(total_requests / wall, 1),
        'successful_requests_per_second': round((total_requests - errors) / wall, 1),
        'duration_s': wall,
        'latency_scope': 'all completed client calls, including errors/timeouts',
        'latency_ms': {
            'mean': round(statistics.mean(latencies) * 1000, 2),
            'p50': round(percentile(latencies, 50) * 1000, 2),
            'p90': round(percentile(latencies, 90) * 1000, 2),
            'p99': round(percentile(latencies, 99) * 1000, 2),
            'max': round(latencies[-1] * 1000, 2),
        },
    }
    print(f"    -> {result['throughput_rps']} req/s, "
          f"p50 {result['latency_ms']['p50']}ms, "
          f"p99 {result['latency_ms']['p99']}ms, "
          f"errors {errors}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gateway', default='http://localhost:8080')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--requests', type=int, default=400)
    parser.add_argument('--output', default=None)
    parser.add_argument('--duration-seconds', type=float, default=0, help='Minimum duration per read endpoint; zero uses fixed request count')
    parser.add_argument('--skip-submissions', action='store_true')
    parser.add_argument('--job-id', help='Reuse an existing job; avoids seed writes in persistence-outage tests')
    args = parser.parse_args()
    if args.requests < 8 or args.concurrency < 1 or args.duration_seconds < 0:
        parser.error('Need requests >= 8, concurrency >= 1, and nonnegative duration')

    # Warm-up + reachability check
    _, ok = do_request('GET', f'{args.gateway}/health')
    if not ok:
        print(f'Gateway not reachable at {args.gateway}', file=sys.stderr)
        sys.exit(1)

    # Seed one job so the status endpoint has a real job to query
    job_body = {
        'model_type': 'benchmark',
        'dataset_path': 'datasets/benchmark.csv',
        'num_workers': 1,
        'epochs': 1,
        'hyperparameters': {'job_name': 'gateway-benchmark-seed'},
    }
    if args.job_id:
        seed_job_id = args.job_id
    else:
        req = urllib.request.Request(
            f'{args.gateway}/api/v1/jobs',
            data=json.dumps(job_body).encode(), method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('X-User-ID', 'benchmark')
        with urllib.request.urlopen(req, timeout=15) as resp:
            seed_job_id = json.loads(resp.read())['job_id']
    
        # Status reads do not need active simulated training in the background.
        do_request('DELETE', f'{args.gateway}/api/v1/jobs/{seed_job_id}')

    print(f'Benchmarking gateway at {args.gateway}')
    results = [
        run_scenario('GET /health', 'GET', f'{args.gateway}/health',
                     None, args.concurrency, args.requests, args.duration_seconds),
        run_scenario('GET /worker-activity', 'GET',
                     f'{args.gateway}/worker-activity',
                     None, args.concurrency, args.requests, args.duration_seconds),
        run_scenario('GET /api/v1/jobs/:id', 'GET',
                     f'{args.gateway}/api/v1/jobs/{seed_job_id}',
                     None, args.concurrency, args.requests, args.duration_seconds),
    ]
    if not args.skip_submissions:
        results.append(run_scenario('POST /api/v1/jobs', 'POST', f'{args.gateway}/api/v1/jobs',
                                   job_body, args.concurrency, args.requests // 8))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                       'results': results}, f, indent=2)
        print(f'Results written to {args.output}')


if __name__ == '__main__':
    main()
