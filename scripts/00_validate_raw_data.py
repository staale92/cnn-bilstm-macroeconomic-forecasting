from __future__ import annotations

import argparse
from pathlib import Path

from _common import bootstrap_src


def main() -> None:
    bootstrap_src()

    from macro_bilstm.data.io import load_raw_dataset_xls
    from macro_bilstm.data.schema import CANONICAL_VARIABLES
    from macro_bilstm.data.transform import ensure_monotonic_dates, validate_monthly_continuity
    from macro_bilstm.utils.config import load_yaml

    p = argparse.ArgumentParser(description="Validate the raw macro dataset (shape/schema/dates/missingness).")
    p.add_argument("--dataset-config", default="configs/dataset.yaml")
    args = p.parse_args()

    cfg = load_yaml(args.dataset_config)
    ds = cfg.get("dataset", {})

    raw_path = Path(ds.get("raw_path", "data/data_all_variables.xls"))
    sheet_name = ds.get("sheet_name", 0)
    date_col = ds.get("date_col", "date")

    df = load_raw_dataset_xls(raw_path, sheet_name=sheet_name, date_col=date_col)
    df = ensure_monotonic_dates(df, date_col=date_col)
    validate_monthly_continuity(df, date_col=date_col)

    expected_vars = set(CANONICAL_VARIABLES)
    got_vars = set(df.columns) - {date_col}
    missing = expected_vars - got_vars
    extra = got_vars - expected_vars
    if missing:
        raise SystemExit(f"Missing expected columns: {sorted(missing)}")
    if extra:
        raise SystemExit(f"Found unexpected columns: {sorted(extra)}")

    if len(df) != 871:
        raise SystemExit(f"Expected 871 rows, got {len(df)}")

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    if str(min_date.date()) != "1947-06-30" or str(max_date.date()) != "2019-12-31":
        raise SystemExit(f"Unexpected date range: {min_date} .. {max_date}")

    # The provided XLS contains 7 leading NaNs for 3 labor-market variables.
    # Allow only that exact pattern; otherwise fail fast.
    na = df[CANONICAL_VARIABLES].isna().sum().to_dict()
    allowed = {"employment_rate", "unemployment_rate", "unempl_over_27w"}
    for col, cnt in na.items():
        if cnt == 0:
            continue
        if col in allowed and cnt == 7 and df.loc[:6, col].isna().all():
            continue
        raise SystemExit(f"Unexpected missingness: {col} has {cnt} NaNs (expected 0, or 7 leading NaNs for {sorted(allowed)})")

    print("OK")
    print(f"- rows: {len(df)}")
    print(f"- date range: {min_date.date()} .. {max_date.date()}")
    print("- columns:", ", ".join(CANONICAL_VARIABLES))
    print("- missing values: only 7 leading NaNs for employment/unemployment variables (expected)")


if __name__ == "__main__":
    main()
