from __future__ import annotations

from pathlib import Path

import pandas as pd

from macro_bilstm.data.schema import RAW_TO_CANONICAL


def load_raw_dataset_xls(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    date_col: str = "date",
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name)

    df.columns = [str(c).strip() for c in df.columns]
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {path}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    renamed = {raw: canon for raw, canon in RAW_TO_CANONICAL.items() if raw in df.columns}
    df = df.rename(columns=renamed)
    return df

