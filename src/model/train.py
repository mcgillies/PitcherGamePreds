"""
Model training for pitcher strikeout prediction.

Neural network model with optional hyperparameter tuning.
"""

from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def create_model(
    input_dim: int,
    hidden_layers: list[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> Sequential:
    """
    Create a neural network for strikeout prediction.

    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate between layers
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = Sequential()

    # Input layer
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer (regression)
    model.add(Dense(1, activation='linear'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )

    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: list[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 20,
    model_save_path: str | None = None,
) -> Tuple[Sequential, dict]:
    """
    Train the strikeout prediction model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        hidden_layers: Hidden layer sizes
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        epochs: Maximum epochs
        batch_size: Batch size
        patience: Early stopping patience
        model_save_path: Path to save best model

    Returns:
        Tuple of (trained model, training history)
    """
    input_dim = X_train.shape[1]

    print(f"Creating model with input_dim={input_dim}")
    print(f"Hidden layers: {hidden_layers}")

    model = create_model(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if model_save_path:
        save_path = Path(model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                str(save_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
            )
        )

    # Train
    print(f"\nTraining for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history.history


def tune_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_trials: int = 50,
    epochs_per_trial: int = 50,
    project_name: str = "k_prediction_tuning",
    output_dir: str = "models/tuning",
) -> Tuple[Sequential, dict]:
    """
    Tune hyperparameters using Keras Tuner.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        max_trials: Maximum tuning trials
        epochs_per_trial: Epochs per trial
        project_name: Name for tuning project
        output_dir: Output directory for tuning results

    Returns:
        Tuple of (best model, best hyperparameters)
    """
    try:
        import keras_tuner as kt
    except ImportError:
        raise ImportError("keras_tuner required for tuning. Install with: pip install keras-tuner")

    input_dim = X_train.shape[1]

    def build_model(hp):
        model = Sequential()

        # Number of layers
        n_layers = hp.Int('n_layers', min_value=1, max_value=5, default=3)

        for i in range(n_layers):
            units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)

            if i == 0:
                model.add(Dense(units, activation='relu', input_shape=(input_dim,)))
            else:
                model.add(Dense(units, activation='relu'))

            if hp.Boolean(f'batch_norm_{i}', default=True):
                model.add(BatchNormalization())

            dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='linear'))

        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae'],
        )

        return model

    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=epochs_per_trial,
        factor=3,
        directory=output_dir,
        project_name=project_name,
        overwrite=True,
    )

    print(f"Starting hyperparameter search ({max_trials} max trials)...")

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_per_trial,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    print("\nBest hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"  {param}: {value}")

    return best_model, best_hps.values


def save_model(model: Sequential, path: str) -> None:
    """Save model to disk."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


def load_model(path: str) -> Sequential:
    """Load model from disk."""
    return keras.models.load_model(path)


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)

    # Dummy data
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randn(200)

    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        hidden_layers=[64, 32],
        epochs=10,
        model_save_path="models/test_model.keras",
    )
