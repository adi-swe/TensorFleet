"""
ML Model Trainer - handles training for different algorithms
"""
from __future__ import annotations  # keeps tf.* annotations lazy when TF is absent

import time
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError:  # TF is optional — only needed for the cnn/dnn algorithms
    tf = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

from config import get_logger

logger = get_logger(__name__)


class MLModelTrainer:
    """Handles ML model training with different algorithms"""

    def __init__(self):
        pass

    def prepare_data(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Prepare data for training
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, features)
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not in dataframe")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        X_train, X_test = X_train.copy(), X_test.copy()
        # Learn categorical vocabularies only from training observations.
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train[col] = encoder.fit_transform(X_train[[col]]).ravel()
            X_test[col] = encoder.transform(X_test[[col]]).ravel()

        # Encode target variable if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        # Scale numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, X.columns.tolist()

    def _build_tf_model(
        self, 
        algorithm: str, 
        input_shape: tuple, 
        num_classes: int, 
        hyperparameters: Dict
    ) -> tf.keras.Model:
        """
        Builds a TensorFlow model.
        
        Args:
            algorithm: 'dnn' or 'cnn'
            input_shape: Shape of input data
            num_classes: Number of output classes
            hyperparameters: Model hyperparameters
            
        Returns:
            Compiled TensorFlow model
        """
        model = tf.keras.Sequential()
        
        if algorithm == 'dnn':
            model.add(tf.keras.layers.Input(shape=input_shape))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
            
        elif algorithm == 'cnn':
            # Reshape for CNN if input is flat
            if len(input_shape) == 1:
                # Attempt to make it square-like for 2D convolution
                side = int(np.sqrt(input_shape[0]))
                if side * side != input_shape[0]:
                    raise ValueError(
                        "Cannot reshape flat input to a square for CNN. "
                        "Input features must be a perfect square."
                    )
                reshape_target = (side, side, 1)
            else:
                reshape_target = (*input_shape, 1)

            model.add(tf.keras.layers.Input(shape=input_shape))
            model.add(tf.keras.layers.Reshape(reshape_target))
            model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(128, activation="relu"))
            model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

        # Configure optimizer
        optimizer_name = hyperparameters.get('optimizer', 'adam').lower()
        learning_rate = float(hyperparameters.get('learning_rate', 0.001))
        optimizer = tf.keras.optimizers.get({
            'class_name': optimizer_name,
            'config': {'learning_rate': learning_rate}
        })

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_tensorflow_model(
        self, 
        algorithm: str, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray, 
        hyperparameters: Dict
    ) -> Tuple[tf.keras.Model, Dict]:
        """
        Trains a TensorFlow model (CNN or DNN).
        
        Args:
            algorithm: 'dnn' or 'cnn'
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            hyperparameters: Model hyperparameters
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        if tf is None:
            raise ImportError(
                "tensorflow is not installed — required for 'cnn'/'dnn' algorithms"
            )
        logger.info(f"Starting TensorFlow training for {algorithm} model...")
        start_time = time.time()

        # Determine input shape and number of classes
        input_shape = (X_train.shape[1],)
        num_classes = len(np.unique(y_train))

        # Build model
        model = self._build_tf_model(algorithm, input_shape, num_classes, hyperparameters)
        model.summary(print_fn=logger.info)

        # Get hyperparameters
        batch_size = int(hyperparameters.get('batch_size', 32))
        epochs = int(hyperparameters.get('epochs', 10))

        # Create tf.data pipeline
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train, y_train, test_size=.2, stratify=y_train,
            random_state=int(hyperparameters.get('random_state', 42)))
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(buffer_size=len(X_train))\
            .batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))\
            .batch(batch_size)
        validation_dataset = tf.data.Dataset.from_tensor_slices((X_validation, y_validation)).batch(batch_size)

        # Train model
        history = model.fit(
            train_dataset, 
            epochs=epochs, 
            validation_data=validation_dataset,
            verbose=0
        )
        training_time = time.time() - start_time

        # Evaluate model
        train_loss, train_accuracy = model.evaluate(train_dataset, verbose=0)
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
        
        # Calculate predictions for detailed metrics
        y_pred_test = model.predict(test_dataset)
        y_pred_test_classes = np.argmax(y_pred_test, axis=1)
        
        # Calculate precision, recall, and F1-score
        average_type = 'weighted' if num_classes > 2 else 'binary'
        precision = precision_score(
            y_test, y_pred_test_classes, average=average_type, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred_test_classes, average=average_type, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred_test_classes, average=average_type, zero_division=0
        )

        logger.info(
            f"TensorFlow training completed in {training_time:.2f}s. "
            f"Test accuracy: {test_accuracy:.4f}, F1: {f1:.4f}"
        )

        metrics = {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "accuracy": float(test_accuracy),  # Alias for consistency
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "loss": float(test_loss),  # Alias for consistency
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "training_time": training_time,
            "history": {k: [float(val) for val in v] for k, v in history.history.items()}
        }
        return model, metrics

    def train_sklearn_model(
        self, 
        algorithm: str, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray, 
        hyperparameters: Dict
    ) -> Tuple[Any, Dict]:
        """
        Train scikit-learn model
        
        Args:
            algorithm: Model algorithm name
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            hyperparameters: Model hyperparameters
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        start_time = time.time()

        # Initialize model based on algorithm
        if algorithm == 'random_forest':
            model = RandomForestClassifier(**hyperparameters)
        elif algorithm == 'logistic_regression':
            model = LogisticRegression(**hyperparameters)
        elif algorithm == 'svm':
            model = SVC(**hyperparameters)
        elif algorithm == 'decision_tree':
            model = DecisionTreeClassifier(**hyperparameters)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Calculate precision, recall, and F1-score
        num_classes = len(np.unique(y_test))
        average_type = 'weighted' if num_classes > 2 else 'binary'
        precision = precision_score(
            y_test, y_pred_test, average=average_type, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred_test, average=average_type, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred_test, average=average_type, zero_division=0
        )
        
        training_time = time.time() - start_time
        
        metrics = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'accuracy': float(test_accuracy),  # Alias for consistency
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': training_time,
            'classification_report': classification_report(
                y_test, y_pred_test, output_dict=True
            )
        }
        
        logger.info(
            f"Model trained - Algorithm: {algorithm}, "
            f"Test Accuracy: {test_accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1: {f1:.4f}"
        )
        
        return model, metrics

    def train_model(
        self, 
        algorithm: str, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray, 
        hyperparameters: Dict
    ) -> Tuple[Any, Dict]:
        """
        Train model with specified algorithm
        
        Args:
            algorithm: Model algorithm name
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            hyperparameters: Model hyperparameters
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        try:
            if algorithm in ['cnn', 'dnn']:
                return self.train_tensorflow_model(
                    algorithm, X_train, X_test, y_train, y_test, hyperparameters
                )
            elif algorithm.startswith('pytorch_'):
                # Imported lazily so the service still runs without torch
                # installed when only sklearn/TF algorithms are used.
                from pytorch_trainer import train_pytorch_model
                return train_pytorch_model(
                    algorithm, X_train, X_test, y_train, y_test, hyperparameters
                )
            else:
                return self.train_sklearn_model(
                    algorithm, X_train, X_test, y_train, y_test, hyperparameters
                )
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise


def get_default_hyperparameters(algorithm: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for algorithm
    
    Args:
        algorithm: Model algorithm name
        
    Returns:
        Dictionary of default hyperparameters
    """
    defaults = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        },
        'logistic_regression': {
            'max_iter': 1000,
            'random_state': 42
        },
        'svm': {
            'kernel': 'rbf',
            'random_state': 42
        },
        'decision_tree': {
            'max_depth': None,
            'random_state': 42
        },
        'pytorch_logistic': {
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'random_state': 42
        },
        'pytorch_mlp': {
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'hidden_layers': [128, 64],
            'dropout': 0.2,
            'random_state': 42
        },
        'pytorch_cnn': {
            'epochs': 15,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'random_state': 42
        },
        'pytorch_lstm': {
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'hidden_size': 64,
            'num_layers': 1,
            'random_state': 42
        },
        'pytorch_transformer': {
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.0005,
            'optimizer': 'adamw',
            'd_model': 32,
            'nhead': 4,
            'num_encoder_layers': 2,
            'dropout': 0.1,
            'random_state': 42
        }
    }
    return defaults.get(algorithm, {})
