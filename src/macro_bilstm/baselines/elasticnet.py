from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet


@dataclass(frozen=True)
class ElasticNetConfig:
    alpha: float = 0.001
    l1_ratio: float = 0.5
    max_iter: int = 5000


def fit_predict_onestep(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, cfg: ElasticNetConfig) -> np.ndarray:
    model = ElasticNet(alpha=cfg.alpha, l1_ratio=cfg.l1_ratio, max_iter=cfg.max_iter)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_multistep(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, cfg: ElasticNetConfig
) -> np.ndarray:
    model = MultiTaskElasticNet(alpha=cfg.alpha, l1_ratio=cfg.l1_ratio, max_iter=cfg.max_iter)
    model.fit(X_train, y_train)
    return model.predict(X_test)

