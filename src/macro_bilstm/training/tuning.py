from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from macro_bilstm.training.train_torch import predict, train_model
from macro_bilstm.utils.artifacts import save_json
from macro_bilstm.utils.seed import set_global_seed


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {**base}
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def sample_from_space(space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, dict):
            out[k] = sample_from_space(v, rng)
        elif isinstance(v, list):
            if not v:
                raise ValueError(f"Empty search list for '{k}'")
            out[k] = v[int(rng.integers(0, len(v)))]
        else:
            raise TypeError(f"Search space values must be dict or list; got {type(v).__name__} for '{k}'")
    return out


def _inverse_y(y_scaler, arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    return y_scaler.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)


def _metric_value(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = metric.lower()
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have same shape")
    if m == "mae":
        return float(np.mean(np.abs(yp - yt)))
    if m == "rmse":
        return float(np.sqrt(np.mean((yp - yt) ** 2)))
    raise ValueError(f"Unknown metric: {metric!r} (expected: mae|rmse)")


@dataclass(frozen=True)
class TrialResult:
    trial: int
    score: float
    best_epoch: int
    sampled: dict[str, Any]
    repeat_scores: list[float] | None = None
    repeat_best_epochs: list[int] | None = None


@dataclass(frozen=True)
class TuningResult:
    best_score: float
    best: dict[str, Any]  # {"model": ..., "training": ...}
    trials: list[TrialResult]


def random_search_tune(
    *,
    build_model: Callable[[dict[str, Any]], torch.nn.Module],
    base_model_cfg: dict[str, Any],
    base_training_cfg: dict[str, Any],
    search_space: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_scaler: Any,
    metric: str,
    n_trials: int,
    repeats: int,
    seed: int,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    max_norm: float,
    early_stopping: bool,
    patience: int,
) -> TuningResult:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    rng = np.random.default_rng(seed)
    trials: list[TrialResult] = []
    best_score = float("inf")
    best_cfg: dict[str, Any] | None = None

    for trial in range(n_trials):
        sampled = sample_from_space(search_space, rng)
        sampled_model = sampled.get("model", {})
        sampled_training: dict[str, Any] = {}
        if "training" in sampled:
            tr = sampled["training"]
            if not isinstance(tr, dict):
                raise TypeError("tuning.search_space.training must be a mapping")
            sampled_training.update(tr)
        for k, v in sampled.items():
            if k in {"model", "training"}:
                continue
            sampled_training[k] = v

        model_cfg = deep_merge(base_model_cfg, sampled_model) if sampled_model else dict(base_model_cfg)
        training_cfg = deep_merge(base_training_cfg, sampled_training) if sampled_training else dict(base_training_cfg)

        repeat_scores: list[float] = []
        repeat_best_epochs: list[int] = []
        for r in range(repeats):
            trial_seed = int(seed + trial * 1000 + r)
            set_global_seed(trial_seed)

            model = build_model(model_cfg)
            tr = train_model(
                model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(training_cfg.get("weight_decay", 0.0)),
                loss=str(training_cfg.get("loss", "mse")),
                grad_clip_norm=training_cfg.get("grad_clip_norm"),
                max_norm=max_norm,
                early_stopping=early_stopping,
                patience=patience,
                device=device,
            )

            y_pred_scaled = predict(model, X_val, device=device)
            y_pred = _inverse_y(y_scaler, y_pred_scaled)
            y_true = _inverse_y(y_scaler, y_val)
            repeat_scores.append(_metric_value(metric, y_true, y_pred))
            repeat_best_epochs.append(int(tr.best_epoch))

        score = float(np.mean(repeat_scores))
        best_epoch = int(repeat_best_epochs[int(np.argmin(repeat_scores))])

        trials.append(
            TrialResult(
                trial=trial,
                score=score,
                best_epoch=best_epoch,
                sampled=sampled,
                repeat_scores=[float(s) for s in repeat_scores],
                repeat_best_epochs=repeat_best_epochs,
            )
        )

        if score < best_score:
            best_score = score
            best_cfg = {"model": model_cfg, "training": training_cfg}

    assert best_cfg is not None
    return TuningResult(best_score=best_score, best=best_cfg, trials=trials)


def save_tuning_artifacts(
    *,
    out_dir: str | Path,
    result: TuningResult,
    extra: dict[str, Any] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_payload: dict[str, Any] = {"best_score": result.best_score, "best": result.best}
    if extra is not None:
        best_payload["extra"] = extra
    save_json(out_dir / "best.json", best_payload)

    rows: list[str] = ["trial,score,best_epoch"]
    trials_jsonl: list[str] = []
    for t in result.trials:
        rows.append(f"{t.trial},{t.score:.8f},{t.best_epoch}")
        trials_jsonl.append(
            json_dumps_compact(
                {
                    "trial": t.trial,
                    "score": t.score,
                    "best_epoch": t.best_epoch,
                    "repeat_scores": t.repeat_scores,
                    "repeat_best_epochs": t.repeat_best_epochs,
                    "sampled": t.sampled,
                }
            )
        )
    (out_dir / "trials.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out_dir / "trials.jsonl").write_text("\n".join(trials_jsonl) + "\n", encoding="utf-8")


def json_dumps_compact(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
