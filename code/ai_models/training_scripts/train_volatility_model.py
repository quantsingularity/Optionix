"""
Training script for the Optionix volatility prediction LSTM model.

Usage:
    python -m ai_models.training_scripts.train_volatility_model \
        --data path/to/historical_volatility.csv \
        --output ai_models/volatility_model.h5
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

WINDOW_SIZE = 60
FEATURES = ["open", "high", "low", "volume"]
TARGET = "volatility"


def load_and_scale(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Load CSV, scale features and target, return arrays + scalers."""
    df = pd.read_csv(filepath)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(df[FEATURES].values)
    y_scaled = target_scaler.fit_transform(df[[TARGET]].values)

    return X_scaled, y_scaled, feature_scaler, target_scaler


def create_sequences(
    X: np.ndarray, y: np.ndarray, window: int = WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build supervised (X_seq, y_seq) pairs.

    X_seq shape: (samples, window, n_features)
    y_seq shape: (samples,)
    """
    X_seq, y_seq = [], []
    for i in range(window, len(X)):
        X_seq.append(X[i - window : i])  # (window, n_features)
        y_seq.append(y[i, 0])  # scalar target
    return np.array(X_seq), np.array(y_seq)


def build_model(n_features: int, window: int = WINDOW_SIZE) -> tf.keras.Model:
    """Build and compile the LSTM model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, n_features)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(data_path: str, output_path: str) -> None:
    """End-to-end training pipeline."""
    logger.info("Loading data from %s", data_path)
    X_scaled, y_scaled, feature_scaler, target_scaler = load_and_scale(data_path)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window=WINDOW_SIZE)
    logger.info("Sequences: X=%s  y=%s", X_seq.shape, y_seq.shape)

    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    model = build_model(n_features=len(FEATURES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)
    logger.info("Model saved to %s", output_path)

    import joblib

    base = Path(output_path).with_suffix("")
    joblib.dump(feature_scaler, str(base.parent / "feature_scaler.pkl"))
    joblib.dump(target_scaler, str(base.parent / "target_scaler.pkl"))
    logger.info("Scalers saved next to model")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train Optionix volatility model")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to historical_volatility.csv",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parents[1] / "volatility_model.h5"),
        help="Output path for the saved model",
    )
    args = parser.parse_args()
    train_model(args.data, args.output)
