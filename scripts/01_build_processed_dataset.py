from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import bootstrap_src, load_experiment_bundle


def main() -> None:
    root = bootstrap_src()

    from macro_bilstm.data.build import build_experiment_data
    from macro_bilstm.data.io import load_raw_dataset_xls
    from macro_bilstm.data.transform import (
        ensure_monotonic_dates,
        fill_missing_values,
        validate_monthly_continuity,
        validate_no_missing,
    )
    from macro_bilstm.utils.config import load_yaml, save_yaml
    from macro_bilstm.utils.experiment import experiment_fingerprint

    p = argparse.ArgumentParser(description="Build processed dataset + (optional) cached windows/scalers per experiment.")
    p.add_argument("--dataset-config", default="configs/dataset.yaml")
    p.add_argument(
        "--experiment-config",
        action="append",
        default=[],
        help="Experiment config(s) to precompute windows/scalers for (repeatable).",
    )
    args = p.parse_args()

    dataset_cfg = load_yaml(args.dataset_config)
    ds = dataset_cfg.get("dataset", {})
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = ds.get("raw_path", "data/data_all_variables.xls")
    sheet_name = ds.get("sheet_name", 0)
    date_col = ds.get("date_col", "date")
    missing_strategy = ds.get("fill_missing", {}).get("strategy", "backfill_then_forwardfill")

    df = load_raw_dataset_xls(raw_path, sheet_name=sheet_name, date_col=date_col)
    df = ensure_monotonic_dates(df, date_col=date_col)
    validate_monthly_continuity(df, date_col=date_col)
    df = fill_missing_values(df, strategy=missing_strategy, date_col=date_col)
    validate_no_missing(df, date_col=date_col)

    parquet_path = out_dir / "macro_processed.parquet"
    df.to_parquet(parquet_path, index=False)

    meta = {
        "raw_path": str(raw_path),
        "rows": int(len(df)),
        "columns": [c for c in df.columns],
        "min_date": str(df[date_col].min().date()),
        "max_date": str(df[date_col].max().date()),
        "missing_strategy": missing_strategy,
    }
    meta_path = out_dir / "macro_processed.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    save_yaml(dataset_cfg, out_dir / "dataset_config.snapshot.yaml")

    print(f"Wrote {parquet_path}")
    print(f"Wrote {meta_path}")

    for exp_config_path in args.experiment_config:
        exp_cfg_all, dataset_cfg2, cov_cfg, _ = load_experiment_bundle(exp_config_path)
        exp_id, fingerprint = experiment_fingerprint(exp_cfg_all, dataset_cfg2)
        exp = exp_cfg_all["experiment"]
        exp_name = exp["name"]

        cache_root = out_dir / "experiments" / f"{exp_name}__{exp_id}"
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / "fingerprint.json").write_text(json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8")

        print(f"\nCaching windows/scalers for {exp_name} ({exp_id}) -> {cache_root}")

        targets = exp["targets"]
        for target in targets:
            ed = build_experiment_data(
                raw_path=dataset_cfg2["dataset"]["raw_path"],
                sheet_name=dataset_cfg2["dataset"].get("sheet_name", 0),
                missing_strategy=dataset_cfg2["dataset"].get("fill_missing", {}).get("strategy", "backfill_then_forwardfill"),
                train_size=int(dataset_cfg2["train_test_split"]["train_size"]),
                target=target,
                feature_mode=exp["feature_mode"],
                covariates=cov_cfg,
                lookback=int(exp["lookback"]),
                horizon=int(exp["horizon"]),
            )

            tdir = cache_root / target
            tdir.mkdir(parents=True, exist_ok=True)

            from macro_bilstm.data.scalers import save_scalers

            save_scalers(ed.scalers, tdir / "scalers.joblib")

            np.savez_compressed(
                tdir / "windows.npz",
                X_train=ed.train_windows.X,
                y_train=ed.train_windows.y,
                origins_train=ed.train_windows.origins,
                X_test=ed.test_windows.X,
                y_test=ed.test_windows.y,
                origins_test=ed.test_windows.origins,
                features=np.asarray(ed.features),
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
