from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from macro_bilstm.utils.hashing import stable_hash


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    config_json: Path
    metrics_json: Path
    predictions_npz: Path
    model_pt: Path


def experiment_id(config: dict[str, Any]) -> str:
    return stable_hash(config, n_chars=12)


def make_run_paths(
    *,
    results_dir: str | Path,
    exp_id: str,
    experiment_name: str,
    model_name: str,
    target: str,
    seed: int | None,
) -> RunPaths:
    results_dir = Path(results_dir)
    seed_part = "seed_na" if seed is None else f"seed_{seed:02d}"
    run_dir = results_dir / "experiments" / f"{experiment_name}__{exp_id}" / model_name / target / seed_part
    return RunPaths(
        run_dir=run_dir,
        config_json=run_dir / "config.json",
        metrics_json=run_dir / "metrics.json",
        predictions_npz=run_dir / "predictions.npz",
        model_pt=run_dir / "model.pt",
    )


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def save_predictions_npz(
    path: str | Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta is None:
        np.savez_compressed(path, y_true=y_true, y_pred=y_pred)
    else:
        np.savez_compressed(path, y_true=y_true, y_pred=y_pred, meta=json.dumps(meta, default=str))

