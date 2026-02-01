from __future__ import annotations

from typing import Any

from macro_bilstm.utils.artifacts import experiment_id


def experiment_fingerprint(exp_cfg: dict[str, Any], dataset_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Returns:
      (exp_id, fingerprint_dict)

    exp_id is a stable hash over the full experiment + dataset config content to make
    run directories reproducible and auditable.
    """
    fingerprint = {
        "dataset": dataset_cfg,
        "experiment": exp_cfg.get("experiment", {}),
        "training": exp_cfg.get("training", {}),
        "model": exp_cfg.get("model", {}),
        "repeats": exp_cfg.get("repeats", {}),
    }
    return experiment_id(fingerprint), fingerprint

