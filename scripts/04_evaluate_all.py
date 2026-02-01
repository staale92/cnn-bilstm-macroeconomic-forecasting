from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from _common import bootstrap_src, load_experiment_bundle


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    bootstrap_src()

    from macro_bilstm.utils.experiment import experiment_fingerprint

    p = argparse.ArgumentParser(description="Aggregate per-run metrics for an experiment into summary CSVs.")
    p.add_argument("--config", default="configs/experiment/univariate_multistep.yaml")
    args = p.parse_args()

    exp_cfg_all, dataset_cfg, _, _ = load_experiment_bundle(args.config)
    exp = exp_cfg_all["experiment"]
    results_dir = Path(exp.get("results_dir", "results"))

    exp_id, _fingerprint = experiment_fingerprint(exp_cfg_all, dataset_cfg)
    experiment_name = exp["name"]

    exp_root = results_dir / "experiments" / f"{experiment_name}__{exp_id}"
    if not exp_root.exists():
        raise SystemExit(f"Missing results dir: {exp_root}")

    rows: list[dict[str, Any]] = []
    for metrics_path in exp_root.glob("*/*/*/metrics.json"):
        # .../<model>/<target>/<seed_XX>/metrics.json
        seed_dir = metrics_path.parent
        target_dir = seed_dir.parent
        model_dir = target_dir.parent
        rows.append(
            {
                "experiment": experiment_name,
                "exp_id": exp_id,
                "model": model_dir.name,
                "target": target_dir.name,
                "seed": seed_dir.name,
                **_read_json(metrics_path),
            }
        )

    if not rows:
        raise SystemExit(f"No metrics found under {exp_root}")

    df = pd.DataFrame(rows)
    out_dir = results_dir / "summary" / f"{experiment_name}__{exp_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = out_dir / "metrics_per_run.csv"
    df.to_csv(per_run_path, index=False)

    summary = (
        df.groupby(["model", "target"], as_index=False)
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), runs=("mae", "count"))
        .sort_values(["target", "model"])
        .reset_index(drop=True)
    )
    summary_path = out_dir / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {per_run_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
