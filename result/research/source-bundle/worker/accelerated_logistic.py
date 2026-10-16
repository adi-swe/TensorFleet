"""Train multinomial logistic regression with CUDA, Apple MPS, or CPU.

The saved estimator is a standard sklearn LogisticRegression with learned
coefficients, so inference does not depend on this module or a GPU.
"""
import platform

import numpy as np
from sklearn.linear_model import LogisticRegression


def available_devices():
    try:
        import torch
    except ImportError:
        return []
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        devices.append('mps')
    return devices


def train_logistic(X, y, parameters, seed, requested_device='auto'):
    import torch

    devices = available_devices()
    if requested_device == 'auto':
        device = next(candidate for candidate in ('cuda', 'mps', 'cpu') if candidate in devices)
    elif requested_device in devices:
        device = requested_device
    else:
        raise ValueError(f'Requested device {requested_device!r} unavailable; available: {devices}')
    allowed = {'epochs', 'batch_size', 'learning_rate', 'weight_decay'}
    if set(parameters) - allowed:
        raise ValueError(f'Unknown accelerated logistic parameters: {sorted(set(parameters) - allowed)}')
    epochs = int(parameters.get('epochs', 30))
    batch_size = int(parameters.get('batch_size', 1024))
    learning_rate = float(parameters.get('learning_rate', .01))
    weight_decay = float(parameters.get('weight_decay', 0))
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError('Invalid accelerated logistic optimization budget')
    torch.manual_seed(seed)
    classes, encoded = np.unique(np.asarray(y), return_inverse=True)
    features = torch.from_numpy(np.asarray(X, dtype=np.float32))
    labels = torch.from_numpy(encoded.astype(np.int64))
    linear = torch.nn.Linear(features.shape[1], len(classes)).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb, yb = features[indices].to(device), labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(linear(xb), yb)
            loss.backward()
            optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    coefficients = linear.weight.detach().cpu().numpy().astype(np.float64)
    intercept = linear.bias.detach().cpu().numpy().astype(np.float64)
    if len(classes) == 2:
        coefficients = (coefficients[1] - coefficients[0])[None, :]
        intercept = np.array([intercept[1] - intercept[0]])
    estimator = LogisticRegression()
    estimator.classes_ = classes
    estimator.coef_ = coefficients
    estimator.intercept_ = intercept
    estimator.n_features_in_ = features.shape[1]
    estimator.n_iter_ = np.array([epochs], dtype=np.int32)
    return estimator, {
        'requested': requested_device, 'actual': device, 'torch': torch.__version__,
        'platform': platform.system(), 'epochs_completed': epochs,
        'batch_size': batch_size, 'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'device_name': torch.cuda.get_device_name(0) if device == 'cuda' else
                       ('Apple GPU via Metal Performance Shaders' if device == 'mps' else 'CPU'),
    }
