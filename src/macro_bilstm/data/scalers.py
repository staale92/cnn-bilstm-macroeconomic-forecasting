from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureTargetScalers:
    x_scaler: StandardScaler
    y_scaler: StandardScaler


def fit_scalers(X_train: np.ndarray, y_train: np.ndarray) -> FeatureTargetScalers:
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D (T, n_features)")
    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D (T,)")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have same length")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(X_train)
    y_scaler.fit(y_train.reshape(-1, 1))
    return FeatureTargetScalers(x_scaler=x_scaler, y_scaler=y_scaler)


def transform_X(scalers: FeatureTargetScalers, X: np.ndarray) -> np.ndarray:
    return scalers.x_scaler.transform(X)


def transform_y(scalers: FeatureTargetScalers, y: np.ndarray) -> np.ndarray:
    return scalers.y_scaler.transform(y.reshape(-1, 1)).reshape(-1)


def inverse_transform_y(scalers: FeatureTargetScalers, y_scaled: np.ndarray) -> np.ndarray:
    return scalers.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)


def save_scalers(scalers: FeatureTargetScalers, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"x_scaler": scalers.x_scaler, "y_scaler": scalers.y_scaler}, path)


def load_scalers(path: str | Path) -> FeatureTargetScalers:
    data = joblib.load(path)
    return FeatureTargetScalers(x_scaler=data["x_scaler"], y_scaler=data["y_scaler"])

