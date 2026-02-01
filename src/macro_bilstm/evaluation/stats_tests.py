from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TTestResult:
    statistic: float
    pvalue: float
    significant: bool


def paired_t_test(a: list[float] | np.ndarray, b: list[float] | np.ndarray, *, alpha: float = 0.05) -> TTestResult:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have same shape")
    if a_arr.size < 2:
        raise ValueError("Need at least 2 paired samples for t-test")
    t = stats.ttest_rel(a_arr, b_arr, alternative="two-sided")
    return TTestResult(statistic=float(t.statistic), pvalue=float(t.pvalue), significant=bool(t.pvalue < alpha))

