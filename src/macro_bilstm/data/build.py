from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macro_bilstm.data.io import load_raw_dataset_xls
from macro_bilstm.data.scalers import FeatureTargetScalers, fit_scalers, transform_X, transform_y
from macro_bilstm.data.splits import paper_split
from macro_bilstm.data.transform import ensure_monotonic_dates, fill_missing_values, validate_monthly_continuity, validate_no_missing
from macro_bilstm.data.windowing import Windows, make_windows


@dataclass(frozen=True)
class ExperimentData:
    target: str
    features: list[str]
    df: pd.DataFrame
    train_size: int
    scalers: FeatureTargetScalers
    train_windows: Windows
    test_windows: Windows


def build_experiment_data(
    *,
    raw_path: str | Path,
    target: str,
    feature_mode: str,  # "univariate" or "multivariate"
    covariates: dict[str, list[str]] | None,
    lookback: int,
    horizon: int,
    train_size: int = 697,
    sheet_name: str | int = 0,
    missing_strategy: str = "backfill_then_forwardfill",
) -> ExperimentData:
    df = load_raw_dataset_xls(raw_path, sheet_name=sheet_name)
    df = ensure_monotonic_dates(df)
    validate_monthly_continuity(df)
    df = fill_missing_values(df, strategy=missing_strategy)
    validate_no_missing(df)

    if target not in df.columns:
        raise ValueError(f"Unknown target '{target}'. Available: {sorted(c for c in df.columns if c!='date')}")

    if feature_mode not in {"univariate", "multivariate"}:
        raise ValueError("feature_mode must be 'univariate' or 'multivariate'")

    if feature_mode == "univariate":
        features = [target]
    else:
        if covariates is None or target not in covariates:
            raise ValueError(f"Missing covariates mapping for target '{target}'")
        features = [target] + list(covariates[target])

    for c in features:
        if c not in df.columns:
            raise ValueError(f"Missing feature '{c}' in dataset")

    split = paper_split(df, train_size=train_size)
    X_train = split.train[features].to_numpy(dtype=np.float64)
    y_train = split.train[target].to_numpy(dtype=np.float64)

    scalers = fit_scalers(X_train, y_train)
    X_all = df[features].to_numpy(dtype=np.float64)
    y_all = df[target].to_numpy(dtype=np.float64)
    X_scaled = transform_X(scalers, X_all)
    y_scaled = transform_y(scalers, y_all)

    train_windows = make_windows(
        X_scaled,
        y_scaled,
        lookback=lookback,
        horizon=horizon,
        start_t=lookback,
        end_t=train_size - horizon,
    )
    test_windows = make_windows(
        X_scaled,
        y_scaled,
        lookback=lookback,
        horizon=horizon,
        start_t=train_size,
        end_t=len(df) - horizon,
    )

    return ExperimentData(
        target=target,
        features=features,
        df=df,
        train_size=train_size,
        scalers=scalers,
        train_windows=train_windows,
        test_windows=test_windows,
    )


def config_to_experiment_kwargs(dataset_cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> dict[str, Any]:
    ds = dataset_cfg.get("dataset", {})
    split = dataset_cfg.get("train_test_split", {})
    return {
        "raw_path": ds.get("raw_path", "data/data_all_variables.xls"),
        "sheet_name": ds.get("sheet_name", 0),
        "missing_strategy": ds.get("fill_missing", {}).get("strategy", "backfill_then_forwardfill"),
        "train_size": int(split.get("train_size", 697)),
        "lookback": int(exp_cfg["lookback"]),
        "horizon": int(exp_cfg["horizon"]),
        "feature_mode": str(exp_cfg["feature_mode"]),
    }

