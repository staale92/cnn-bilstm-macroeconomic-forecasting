from __future__ import annotations

import numpy as np


def rolling_forecast_sarimax(
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    exog_train: np.ndarray | None,
    exog_test: np.ndarray | None,
    order: tuple[int, int, int],
    horizon: int,
) -> np.ndarray:
    """
    Fit SARIMAX on train once, then produce rolling forecasts on the test set by
    appending observed test values (walk-forward), without refitting parameters.

    Returns:
      y_pred: (N_origins, horizon) where N_origins = len(test) - horizon + 1
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as e:  # pragma: no cover
        raise ImportError("statsmodels is required for ARIMA/ARIMAX (install with 'pip install .[baselines]')") from e

    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    if exog_train is not None:
        exog_train = np.asarray(exog_train, dtype=np.float64)
    if exog_test is not None:
        exog_test = np.asarray(exog_test, dtype=np.float64)
        if len(exog_test) != len(y_test):
            raise ValueError("exog_test must match y_test length")

    model = SARIMAX(
        endog=y_train,
        exog=exog_train,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    n_origins = len(y_test) - horizon + 1
    if n_origins <= 0:
        raise ValueError("Test series too short for horizon")

    preds: list[np.ndarray] = []
    for i in range(n_origins):
        exog_future = None if exog_test is None else exog_test[i : i + horizon]
        fc = res.get_forecast(steps=horizon, exog=exog_future).predicted_mean
        preds.append(np.asarray(fc, dtype=np.float64))

        # Walk forward by one (append observed y at this step)
        step_exog = None if exog_test is None else exog_test[i : i + 1]
        res = res.append(endog=y_test[i : i + 1], exog=step_exog, refit=False)

    return np.stack(preds, axis=0)

