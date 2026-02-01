from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from macro_bilstm.evaluation.metrics import RegressionMetrics, regression_metrics


@dataclass(frozen=True)
class EvaluationResult:
    metrics: RegressionMetrics
    per_step: dict[str, list[float]] | None


def evaluate_direct_multistep(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResult:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have same shape")
    if yt.ndim != 2:
        raise ValueError("Expected (N, horizon) arrays")

    metrics = regression_metrics(yt, yp)
    per_step = {
        "mae": [float(np.mean(np.abs(yp[:, i] - yt[:, i]))) for i in range(yt.shape[1])],
        "rmse": [float(np.sqrt(np.mean((yp[:, i] - yt[:, i]) ** 2))) for i in range(yt.shape[1])],
    }
    return EvaluationResult(metrics=metrics, per_step=per_step)


def evaluate_onestep(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResult:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have same length")
    metrics = regression_metrics(yt, yp)
    return EvaluationResult(metrics=metrics, per_step=None)


def metrics_to_dict(res: EvaluationResult) -> dict[str, Any]:
    out: dict[str, Any] = {"mae": res.metrics.mae, "rmse": res.metrics.rmse}
    if res.per_step is not None:
        out["per_step"] = res.per_step
    return out

