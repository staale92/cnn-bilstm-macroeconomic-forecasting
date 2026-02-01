from __future__ import annotations

import numpy as np

from macro_bilstm.data.scalers import fit_scalers, transform_X, transform_y


def test_fit_scalers_uses_only_train_data() -> None:
    # Train distribution centered at 0; test distribution centered far away.
    X_train = np.array([[0.0, 1.0], [0.0, -1.0], [0.0, 3.0]], dtype=np.float64)
    y_train = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    scalers = fit_scalers(X_train, y_train)
    np.testing.assert_allclose(scalers.x_scaler.mean_, X_train.mean(axis=0))
    np.testing.assert_allclose(scalers.y_scaler.mean_.reshape(-1), np.array([y_train.mean()]))

    X_train_scaled = transform_X(scalers, X_train)
    y_train_scaled = transform_y(scalers, y_train)
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), np.zeros(2), atol=1e-9)
    np.testing.assert_allclose(y_train_scaled.mean(), 0.0, atol=1e-9)

