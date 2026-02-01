from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_onestep_series(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)

    plt.figure(figsize=(10, 4))
    plt.plot(yt, label="y_true", linewidth=1.5)
    plt.plot(yp, label="y_pred", linewidth=1.5)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

