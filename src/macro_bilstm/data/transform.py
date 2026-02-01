from __future__ import annotations

from typing import Literal

import pandas as pd


def ensure_monotonic_dates(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)
    return out


def fill_missing_values(
    df: pd.DataFrame,
    *,
    strategy: Literal["backfill_then_forwardfill"] = "backfill_then_forwardfill",
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.sort_values(date_col).reset_index(drop=True).copy()
    cols = [c for c in out.columns if c != date_col]
    if strategy == "backfill_then_forwardfill":
        out[cols] = out[cols].bfill().ffill()
        return out
    raise ValueError(f"Unknown missing-value strategy: {strategy}")


def validate_no_missing(df: pd.DataFrame, *, date_col: str = "date") -> None:
    cols = [c for c in df.columns if c != date_col]
    na = df[cols].isna().sum()
    if (na > 0).any():
        bad = na[na > 0].to_dict()
        raise ValueError(f"Found missing values after preprocessing: {bad}")


def validate_monthly_continuity(df: pd.DataFrame, *, date_col: str = "date") -> None:
    dates = pd.to_datetime(df[date_col])
    inferred = pd.infer_freq(dates)
    if inferred is None:
        raise ValueError("Could not infer frequency from date index")
    # pandas may return 'ME' for month-end
    if inferred not in {"M", "ME", "MS"}:
        raise ValueError(f"Expected monthly frequency; inferred={inferred!r}")
    expected = pd.date_range(dates.min(), dates.max(), freq=inferred)
    actual = pd.DatetimeIndex(dates)
    if not actual.equals(expected):
        raise ValueError("Date index is not continuous monthly data")
