"""
PyTorch Model Trainer — neural models trained with torch.

Algorithms:
    pytorch_logistic     Linear (logistic-regression) classifier
    pytorch_mlp          Multi-layer perceptron
    pytorch_cnn          2D CNN (flat features reshaped to a square image)
    pytorch_lstm         LSTM over the feature sequence
    pytorch_transformer  Transformer encoder over feature tokens

All trainers share one loop (Adam/SGD/AdamW/RMSprop, cross-entropy,
optional early stopping) and return the same metrics dict shape as the
sklearn/TensorFlow trainers so the rest of the pipeline is unchanged.
Models are moved to CPU before returning so the existing pickle-based
save path stays portable across machines.
"""
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from config import get_logger

logger = get_logger(__name__)

PYTORCH_ALGORITHMS = (
    'pytorch_logistic',
    'pytorch_mlp',
    'pytorch_cnn',
    'pytorch_lstm',
    'pytorch_transformer',
)


def get_device() -> torch.device:
    """Pick the best available device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class LogisticModel(nn.Module):
    """Multinomial logistic regression: a single linear layer."""

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)


class MLPModel(nn.Module):
    """Fully-connected network with configurable hidden layers and dropout."""

    def __init__(self, num_features: int, num_classes: int,
                 hidden_layers: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = num_features
        for width in hidden_layers:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNModel(nn.Module):
    """
    2D CNN over tabular/flat inputs reshaped to a square "image"
    (mirrors the TensorFlow CNN: the feature count must be a perfect square).
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        side = int(math.isqrt(num_features))
        if side * side != num_features:
            raise ValueError(
                'pytorch_cnn requires a perfect-square feature count to '
                f'reshape into a 2D grid (got {num_features}).'
            )
        self.side = side
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        pooled = side // 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * pooled * pooled, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, self.side, self.side)
        return self.head(self.conv(x))


class LSTMModel(nn.Module):
    """LSTM that reads the feature vector as a length-N sequence of scalars."""

    def __init__(self, num_features: int, num_classes: int,
                 hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)          # (B, N) -> (B, N, 1)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])    # last layer's final hidden state


class TransformerModel(nn.Module):
    """
    Small Transformer encoder over feature "tokens": each scalar feature is
    projected to d_model, given a learned positional embedding, encoded, then
    mean-pooled into a classification head.
    """

    def __init__(self, num_features: int, num_classes: int,
                 d_model: int = 32, nhead: int = 4, num_encoder_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_features, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x.unsqueeze(-1)) + self.pos_embed  # (B, N, d_model)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


def build_model(algorithm: str, num_features: int, num_classes: int,
                hyperparameters: Dict) -> nn.Module:
    """Instantiate the requested architecture from hyperparameters."""
    if algorithm == 'pytorch_logistic':
        return LogisticModel(num_features, num_classes)
    if algorithm == 'pytorch_mlp':
        hidden = hyperparameters.get('hidden_layers', [128, 64])
        if isinstance(hidden, str):
            hidden = [int(h) for h in hidden.split(',') if h.strip()]
        dropout = float(hyperparameters.get('dropout', 0.2))
        return MLPModel(num_features, num_classes, hidden, dropout)
    if algorithm == 'pytorch_cnn':
        return CNNModel(num_features, num_classes)
    if algorithm == 'pytorch_lstm':
        return LSTMModel(
            num_features, num_classes,
            hidden_size=int(hyperparameters.get('hidden_size', 64)),
            num_layers=int(hyperparameters.get('num_layers', 1)),
        )
    if algorithm == 'pytorch_transformer':
        return TransformerModel(
            num_features, num_classes,
            d_model=int(hyperparameters.get('d_model', 32)),
            nhead=int(hyperparameters.get('nhead', 4)),
            num_encoder_layers=int(hyperparameters.get('num_encoder_layers', 2)),
            dropout=float(hyperparameters.get('dropout', 0.1)),
        )
    raise ValueError(f'Unsupported PyTorch algorithm: {algorithm}')


def _make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
    }
    cls = optimizers.get(name.lower())
    if cls is None:
        logger.warning("Unknown optimizer '%s', falling back to Adam", name)
        cls = torch.optim.Adam
    return cls(params, lr=lr)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device) -> Tuple[float, float, np.ndarray]:
    """Returns (mean_loss, accuracy, predictions)."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    predictions = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        count += xb.size(0)
        predictions.append(preds.cpu().numpy())
    return total_loss / count, correct / count, np.concatenate(predictions)


def train_pytorch_model(
    algorithm: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    hyperparameters: Dict,
) -> Tuple[nn.Module, Dict]:
    """
    Train a PyTorch model. Returns (model_on_cpu, metrics) with the same
    metrics keys as the sklearn/TensorFlow trainers.
    """
    device = get_device()
    logger.info('Starting PyTorch training for %s on %s...', algorithm, device)
    start_time = time.time()

    torch.manual_seed(int(hyperparameters.get('random_state', 42)))

    num_features = X_train.shape[1]
    num_classes = int(len(np.unique(y_train)))
    batch_size = int(hyperparameters.get('batch_size', 32))
    epochs = int(hyperparameters.get('epochs', 10))
    learning_rate = float(hyperparameters.get('learning_rate', 0.001))
    early_stopping = str(hyperparameters.get('early_stopping', 'false')).lower() in ('true', '1')
    patience = int(hyperparameters.get('patience', 5))

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        np.asarray(X_train), np.asarray(y_train), test_size=.2, stratify=y_train,
        random_state=int(hyperparameters.get('random_state', 42)))
    X_train_t = torch.as_tensor(X_fit, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_fit, dtype=torch.long)
    X_test_t = torch.as_tensor(np.asarray(X_test), dtype=torch.float32)
    y_test_t = torch.as_tensor(np.asarray(y_test), dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size
    )
    validation_loader = DataLoader(TensorDataset(
        torch.as_tensor(X_validation, dtype=torch.float32),
        torch.as_tensor(y_validation, dtype=torch.long)), batch_size=batch_size)

    model = build_model(algorithm, num_features, num_classes, hyperparameters).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(
        hyperparameters.get('optimizer', 'adam'), model.parameters(), learning_rate
    )

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    examples_processed = 0
    fit_started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss, correct, count = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            count += xb.size(0)
            examples_processed += xb.size(0)

        train_loss = running_loss / count
        train_acc = correct / count
        val_loss, val_acc, _ = _evaluate(model, validation_loader, criterion, device)

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        logger.info(
            'Epoch %d/%d — loss: %.4f, acc: %.4f, val_loss: %.4f, val_acc: %.4f',
            epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping and epochs_without_improvement >= patience:
                logger.info('Early stopping at epoch %d (patience %d)', epoch + 1, patience)
                break

    if device.type == 'mps': torch.mps.synchronize()
    elif device.type == 'cuda': torch.cuda.synchronize()
    fit_with_validation_s = time.perf_counter() - fit_started
    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss, train_accuracy, _ = _evaluate(model, train_loader, criterion, device)
    test_loss, test_accuracy, y_pred_test = _evaluate(model, test_loader, criterion, device)

    average_type = 'weighted' if num_classes > 2 else 'binary'
    precision = precision_score(y_test, y_pred_test, average=average_type, zero_division=0)
    recall = recall_score(y_test, y_pred_test, average=average_type, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average=average_type, zero_division=0)

    training_time = time.time() - start_time
    logger.info(
        'PyTorch training completed in %.2fs. Test accuracy: %.4f, F1: %.4f',
        training_time, test_accuracy, f1,
    )

    # CPU tensors pickle portably; the save path pickles the whole module.
    model = model.to('cpu')
    model.eval()

    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'accuracy': float(test_accuracy),
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'loss': float(test_loss),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'training_time': training_time,
        'fit_with_validation_s': fit_with_validation_s,
        'examples_processed': examples_processed,
        'epochs_completed': len(history['loss']),
        'fit_rows': len(y_fit), 'validation_rows': len(y_validation),
        'device': str(device),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'history': {k: [float(val) for val in v] for k, v in history.items()},
    }
    return model, metrics
