from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from macro_bilstm.models._constraints import apply_max_norm


@dataclass(frozen=True)
class TrainResult:
    best_epoch: int
    train_losses: list[float]
    val_losses: list[float]


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def train_model(
    model: nn.Module,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    loss: str = "mse",
    grad_clip_norm: float | None = None,
    max_norm: float = 3.0,
    early_stopping: bool = True,
    patience: int = 20,
    device: str | torch.device | None = None,
) -> TrainResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_lc = loss.lower()
    if loss_lc == "mse":
        loss_fn: nn.Module = nn.MSELoss()
    elif loss_lc == "mae":
        loss_fn = nn.L1Loss()
    elif loss_lc == "huber":
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss: {loss!r} (expected: mse|mae|huber)")

    X_train_t = torch.from_numpy(_as_float32(X_train))
    y_train_t = torch.from_numpy(_as_float32(y_train))
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, drop_last=False
    )

    if X_val is not None and y_val is not None:
        X_val_t = torch.from_numpy(_as_float32(X_val))
        y_val_t = torch.from_numpy(_as_float32(y_val))
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False, drop_last=False
        )
    else:
        val_loader = None

    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            opt.step()
            apply_max_norm(model.parameters(), max_norm=max_norm)
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_losses.append(float(np.mean(epoch_losses)))

        if val_loader is None:
            continue

        model.eval()
        with torch.no_grad():
            v_losses: list[float] = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                v_losses.append(float(loss_fn(pred, yb).detach().cpu().item()))
        val = float(np.mean(v_losses))
        val_losses.append(val)

        if val < best_val - 1e-9:
            best_val = val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if early_stopping and wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if val_loader is None:
        best_epoch = len(train_losses) - 1

    return TrainResult(best_epoch=best_epoch, train_losses=train_losses, val_losses=val_losses)


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    X_t = torch.from_numpy(_as_float32(X))
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False, drop_last=False)
    preds: list[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        yhat = model(xb).detach().cpu().numpy()
        preds.append(yhat)
    return np.concatenate(preds, axis=0)
