from __future__ import annotations

from pathlib import Path
from typing import Any

import sys


def bootstrap_src() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def load_experiment_bundle(exp_config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Returns:
      (exp_cfg_all, dataset_cfg, covariates_cfg, baselines_cfg)
    """
    bootstrap_src()
    from macro_bilstm.utils.config import load_yaml

    exp_cfg_all = load_yaml(exp_config_path)
    exp = exp_cfg_all.get("experiment", {})

    dataset_cfg = load_yaml(exp["dataset_config"])
    covariates_cfg = load_yaml(exp["covariates_config"])
    baselines_cfg = load_yaml(exp["baselines_config"])
    return exp_cfg_all, dataset_cfg, covariates_cfg, baselines_cfg


def get_targets(exp_cfg_all: dict[str, Any]) -> list[str]:
    exp = exp_cfg_all.get("experiment", {})
    targets = exp.get("targets")
    if not targets:
        raise ValueError("Experiment config must include experiment.targets")
    return list(targets)

