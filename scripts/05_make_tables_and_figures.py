from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import bootstrap_src, load_experiment_bundle


def _fmt(mean: float, std: float | None, runs: int) -> str:
    if runs >= 2 and std is not None and not np.isnan(std):
        return f"{mean:.4f} ({std:.4f})"
    return f"{mean:.4f}"


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["y_true"], data["y_pred"]


def main() -> None:
    bootstrap_src()

    from macro_bilstm.evaluation.plots import plot_onestep_series
    from macro_bilstm.evaluation.stats_tests import paired_t_test
    from macro_bilstm.utils.experiment import experiment_fingerprint

    p = argparse.ArgumentParser(description="Create paper-style tables/figures and paired t-tests from run artifacts.")
    p.add_argument("--config", default="configs/experiment/univariate_multistep.yaml")
    args = p.parse_args()

    exp_cfg_all, dataset_cfg, _, _ = load_experiment_bundle(args.config)
    exp = exp_cfg_all["experiment"]
    results_dir = Path(exp.get("results_dir", "results"))
    plots_cfg = exp_cfg_all.get("plots", {})

    exp_id, _fingerprint = experiment_fingerprint(exp_cfg_all, dataset_cfg)
    experiment_name = exp["name"]

    summary_dir = results_dir / "summary" / f"{experiment_name}__{exp_id}"
    per_run_path = summary_dir / "metrics_per_run.csv"
    summary_path = summary_dir / "metrics_summary.csv"
    if not per_run_path.exists() or not summary_path.exists():
        raise SystemExit(f"Missing summary files. Run `python scripts/04_evaluate_all.py --config {args.config}` first.")

    per_run = pd.read_csv(per_run_path)
    summary = pd.read_csv(summary_path)

    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Tables ----------------
    summary_fmt = summary.copy()
    summary_fmt["mae_fmt"] = [
        _fmt(m, s, r) for m, s, r in zip(summary_fmt["mae_mean"], summary_fmt["mae_std"], summary_fmt["runs"])
    ]
    summary_fmt["rmse_fmt"] = [
        _fmt(m, s, r) for m, s, r in zip(summary_fmt["rmse_mean"], summary_fmt["rmse_std"], summary_fmt["runs"])
    ]

    mae_table = summary_fmt.pivot(index="target", columns="model", values="mae_fmt").reset_index()
    rmse_table = summary_fmt.pivot(index="target", columns="model", values="rmse_fmt").reset_index()

    mae_path = tables_dir / f"{experiment_name}__{exp_id}__mae_table.csv"
    rmse_path = tables_dir / f"{experiment_name}__{exp_id}__rmse_table.csv"
    mae_table.to_csv(mae_path, index=False)
    rmse_table.to_csv(rmse_path, index=False)
    print(f"Wrote {mae_path}")
    print(f"Wrote {rmse_path}")

    # ---------------- Paired t-tests (CNN–BiLSTM vs BiLSTM) ----------------
    t_rows: list[dict[str, Any]] = []
    for target, g in per_run.groupby("target"):
        a = g[g["model"] == "cnn_bilstm"][["seed", "mae", "rmse"]].rename(columns={"mae": "mae_a", "rmse": "rmse_a"})
        b = g[g["model"] == "bilstm"][["seed", "mae", "rmse"]].rename(columns={"mae": "mae_b", "rmse": "rmse_b"})
        if a.empty or b.empty:
            continue
        merged = a.merge(b, on="seed", how="inner")
        if len(merged) < 2:
            continue

        for metric in ["mae", "rmse"]:
            res = paired_t_test(merged[f"{metric}_a"].to_list(), merged[f"{metric}_b"].to_list(), alpha=0.05)
            t_rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "t_statistic": res.statistic,
                    "pvalue": res.pvalue,
                    "significant_p_lt_0.05": res.significant,
                    "n_pairs": int(len(merged)),
                }
            )

    ttest_path = tables_dir / f"{experiment_name}__{exp_id}__ttest_cnn_bilstm_vs_bilstm.csv"
    pd.DataFrame(t_rows).sort_values(["metric", "target"]).to_csv(ttest_path, index=False)
    print(f"Wrote {ttest_path}")

    # ---------------- Figures (quick sanity plots) ----------------
    plot_targets = plots_cfg.get("targets", ["investments"])
    exp_root = results_dir / "experiments" / f"{experiment_name}__{exp_id}"
    for target in plot_targets:
        for model in ["cnn_bilstm", "bilstm", "arima", "elasticnet"]:
            pred_path = None
            # Prefer seed_00 if present; otherwise any.
            cand = sorted((exp_root / model / target).glob("seed_00/predictions.npz"))
            if cand:
                pred_path = cand[0]
            else:
                cand = sorted((exp_root / model / target).glob("seed_*/predictions.npz"))
                if cand:
                    pred_path = cand[0]
            if pred_path is None:
                continue

            y_true, y_pred = _load_npz(pred_path)
            # For multi-step experiments, plot the 1-step-ahead slice for readability.
            yt = y_true[:, 0] if y_true.ndim == 2 else y_true.reshape(-1)
            yp = y_pred[:, 0] if y_pred.ndim == 2 else y_pred.reshape(-1)
            out = figures_dir / f"{experiment_name}__{exp_id}__{target}__{model}.png"
            plot_onestep_series(y_true=yt, y_pred=yp, title=f"{target} - {model}", out_path=out)

    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
