"""Strict CSV/NPZ dataset decoding; never substitutes synthetic data."""
import io

import numpy as np
import pandas as pd


def load_dataset(content, path, target):
    if path.endswith('.npz'):
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            features, labels = archive['X'], archive['y']
            if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
                raise ValueError('NPZ requires X (rows, features) and y (rows,)')
            frame = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
            frame[target] = labels
            return frame
    return pd.read_csv(io.BytesIO(content))


