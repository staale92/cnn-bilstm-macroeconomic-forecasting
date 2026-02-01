from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Windows:
    X: np.ndarray
    y: np.ndarray
    origins: np.ndarray  # t indices corresponding to y[t : t+horizon]


def make_windows(
    X: np.ndarray,  # (T, n_features)
    y: np.ndarray,  # (T,)
    *,
    lookback: int,
    horizon: int,
    stride: int = 1,
    start_t: int | None = None,
    end_t: int | None = None,
) -> Windows:
    if X.ndim != 2:
        raise ValueError("X must be 2D (T, n_features)")
    if y.ndim != 1:
        raise ValueError("y must be 1D (T,)")
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    if lookback <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("lookback, horizon, stride must be positive")

    T = len(X)
    min_t = lookback
    max_t = T - horizon
    if max_t < min_t:
        raise ValueError("Not enough data for given lookback/horizon")

    if start_t is None:
        start_t = min_t
    if end_t is None:
        end_t = max_t
    if start_t < min_t or end_t > max_t:
        raise ValueError(f"start_t/end_t must satisfy {min_t} <= start_t <= end_t <= {max_t}")

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    origins: list[int] = []
    for t in range(start_t, end_t + 1, stride):
        X_list.append(X[t - lookback : t, :])
        y_list.append(y[t : t + horizon])
        origins.append(t)

    Xw = np.stack(X_list).astype(np.float32)
    yw = np.stack(y_list).astype(np.float32)
    if horizon == 1:
        yw = yw.reshape(-1, 1)
    return Windows(X=Xw, y=yw, origins=np.asarray(origins, dtype=np.int64))

