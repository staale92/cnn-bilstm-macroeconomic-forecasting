from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def paper_split(df: pd.DataFrame, *, train_size: int = 697) -> Split:
    if train_size <= 0 or train_size >= len(df):
        raise ValueError("train_size must be within (0, len(df))")
    train = df.iloc[:train_size].reset_index(drop=True)
    test = df.iloc[train_size:].reset_index(drop=True)
    return Split(train=train, test=test)

