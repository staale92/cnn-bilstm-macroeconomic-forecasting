from __future__ import annotations

import torch


def init_xavier_uniform_(module: torch.nn.Module) -> None:
    for name, p in module.named_parameters():
        if "weight" in name and p.ndim >= 2:
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            torch.nn.init.zeros_(p)

