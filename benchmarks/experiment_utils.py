"""Shared experiment statistics and isolated Compose deployment controls."""
import os
from pathlib import Path
import random
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(values):
    rng = random.Random(2026)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return [means[125], means[4874]]


def compose(*arguments):
    return subprocess.run(['docker', 'compose', '--env-file', '/dev/null', '-p', 'tensorfleet-gap1',
        '-f', 'docker-compose.yml', '-f', 'result/compose.override.yml', *arguments], cwd=ROOT,
        env={**os.environ, 'MONGODB_URL': 'mongodb://mongodb:27017/tensorfleet_gap1', 'MONGODB_DB': 'tensorfleet_gap1'},
        check=True, capture_output=True, text=True)


