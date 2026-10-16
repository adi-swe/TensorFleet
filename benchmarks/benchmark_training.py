#!/usr/bin/env python3
"""
ML training throughput benchmark for the worker-ml algorithms.

Trains each algorithm on a synthetic classification dataset and reports
training time, samples/second, and test accuracy. Requires the worker-ml
dependencies (scikit-learn, and torch for the pytorch_* algorithms).

Usage:
    python3 benchmark_training.py --samples 5000 --features 16 \
        --epochs 10 --output results/training.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'worker-ml'))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SKLEARN_ALGORITHMS = ['random_forest', 'logistic_regression', 'decision_tree']
PYTORCH_ALGORITHMS = [
    'pytorch_logistic', 'pytorch_mlp', 'pytorch_cnn',
    'pytorch_lstm', 'pytorch_transformer',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--features', type=int, default=16,
                        help='Use a perfect square so pytorch_cnn runs')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    X, y = make_classification(
        n_samples=args.samples, n_features=args.features,
        n_informative=max(2, args.features - 6), n_classes=args.classes,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f'Dataset: {args.samples} samples x {args.features} features, '
          f'{args.classes} classes; epochs={args.epochs}')

    results = []

    from model_trainer import MLModelTrainer, get_default_hyperparameters
    trainer = MLModelTrainer()

    for algo in SKLEARN_ALGORITHMS:
        params = get_default_hyperparameters(algo)
        start = time.perf_counter()
        _, metrics = trainer.train_sklearn_model(
            algo, X_train, X_test, y_train, y_test, params)
        elapsed = time.perf_counter() - start
        results.append({
            'algorithm': algo,
            'framework': 'scikit-learn',
            'training_time_s': round(elapsed, 3),
            'samples_per_second': round(len(X_train) / elapsed),
            'throughput_unit': 'input_training_rows_per_complete_trainer_call_second',
            'input_training_rows': len(X_train),
            'test_accuracy': round(metrics['test_accuracy'], 4),
        })
        print(f"  {algo:22s} {elapsed:7.2f}s "
              f"{results[-1]['samples_per_second']:>8,} samples/s "
              f"acc={metrics['test_accuracy']:.3f}")

    try:
        from pytorch_trainer import train_pytorch_model, get_device
        device = str(get_device())
        for algo in PYTORCH_ALGORITHMS:
            params = get_default_hyperparameters(algo)
            params['epochs'] = args.epochs
            start = time.perf_counter()
            _, metrics = train_pytorch_model(
                algo, X_train, X_test, y_train, y_test, params)
            elapsed = time.perf_counter() - start
            total_samples = metrics['examples_processed']
            results.append({
                'algorithm': algo,
                'framework': f'pytorch ({device})',
                'training_time_s': round(elapsed, 3),
                'samples_per_second': round(total_samples / elapsed),
                'throughput_unit': 'actual_fit_examples_processed_per_complete_trainer_call_second',
                'examples_processed': total_samples,
                'epochs_completed': metrics['epochs_completed'],
                'fit_with_validation_s': metrics['fit_with_validation_s'],
                'test_accuracy': round(metrics['test_accuracy'], 4),
                'parameters': metrics['num_parameters'],
            })
            print(f"  {algo:22s} {elapsed:7.2f}s "
                  f"{results[-1]['samples_per_second']:>8,} samples/s "
                  f"acc={metrics['test_accuracy']:.3f} ({device})")
    except ImportError:
        print('  torch not installed — skipping pytorch_* algorithms')

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'dataset': {'samples': args.samples, 'features': args.features,
                            'classes': args.classes, 'epochs': args.epochs},
                'results': results,
            }, f, indent=2)
        print(f'Results written to {args.output}')


if __name__ == '__main__':
    main()
