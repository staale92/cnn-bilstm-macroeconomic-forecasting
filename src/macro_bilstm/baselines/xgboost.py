from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.multioutput import MultiOutputRegressor


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 500
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0


def _make_regressor(cfg: XGBoostConfig):
    try:
        import xgboost as xgb
    except Exception as e:  # pragma: no cover
        raise ImportError("xgboost is required for this baseline (install with 'pip install .[baselines]')") from e

    return xgb.XGBRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        objective="reg:squarederror",
        n_jobs=0,
    )


def fit_predict_onestep(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, cfg: XGBoostConfig) -> np.ndarray:
    model = _make_regressor(cfg)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_multistep(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, cfg: XGBoostConfig) -> np.ndarray:
    base = _make_regressor(cfg)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model.predict(X_test)

