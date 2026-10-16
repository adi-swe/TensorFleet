#!/usr/bin/env python3
"""Stable JSON CLI and import facade for the reusable training package."""
import json
import os
import sys

from training.engine import ALGORITHMS, capabilities, train_job
from training.data import load_dataset
from training.storage import ObjectStore, digest


def main():
    try:
        if '--capabilities' in sys.argv:
            print(json.dumps(capabilities()))
            return
        request = json.load(sys.stdin)
        result = train_job(request, ObjectStore(os.environ.get('STORAGE_SERVICE_URL', 'http://storage:8081')))
        print(json.dumps(result, allow_nan=False))
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
