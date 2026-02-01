from __future__ import annotations

import numpy as np


def rolling_forecast_var(
    *,
    Y_train: np.ndarray,  # (T_train, n_vars)
    Y_test: np.ndarray,  # (T_test, n_vars)
    lags: int,
    horizon: int,
    target_col: int = 0,
) -> np.ndarray:
    """Fit VAR on train once; rolling forecasts on test using observed history."""
    try:
        from statsmodels.tsa.api import VAR
    except Exception as e:  # pragma: no cover
        raise ImportError("statsmodels is required for VAR baseline (install with 'pip install .[baselines]')") from e

    Y_train = np.asarray(Y_train, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)
    if Y_train.ndim != 2 or Y_test.ndim != 2:
        raise ValueError("Y_train/Y_test must be 2D")
    if Y_train.shape[1] != Y_test.shape[1]:
        raise ValueError("Train/test must have same number of variables")
    if lags <= 0:
        raise ValueError("lags must be positive")

    model = VAR(endog=Y_train)
    res = model.fit(maxlags=lags, ic=None, trend="c")

    n_origins = len(Y_test) - horizon + 1
    if n_origins <= 0:
        raise ValueError("Test series too short for horizon")

    preds: list[np.ndarray] = []
    history = np.concatenate([Y_train, Y_test], axis=0)
    train_len = len(Y_train)

    for i in range(n_origins):
        origin = train_len + i
        y_last = history[origin - res.k_ar : origin]
        fc = res.forecast(y_last, steps=horizon)  # (horizon, n_vars)
        preds.append(fc[:, target_col])
    return np.stack(preds, axis=0)

