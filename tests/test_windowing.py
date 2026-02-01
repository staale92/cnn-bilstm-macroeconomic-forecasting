from __future__ import annotations

import numpy as np

from macro_bilstm.data.windowing import make_windows


def test_make_windows_multistep_shapes_and_content() -> None:
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)

    w = make_windows(X, y, lookback=3, horizon=2, stride=1)
    assert w.X.shape == (6, 3, 2)
    assert w.y.shape == (6, 2)
    assert w.origins.tolist() == [3, 4, 5, 6, 7, 8]

    np.testing.assert_array_equal(w.X[0], X[0:3])
    np.testing.assert_array_equal(w.y[0], y[3:5])


def test_make_windows_onestep_y_is_2d() -> None:
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.arange(6, dtype=np.float32)

    w = make_windows(X, y, lookback=2, horizon=1, stride=1)
    assert w.X.shape == (4, 2, 2)
    assert w.y.shape == (4, 1)
    assert w.origins.tolist() == [2, 3, 4, 5]

