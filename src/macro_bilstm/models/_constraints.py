from __future__ import annotations

from collections.abc import Iterable

import torch


@torch.no_grad()
def apply_max_norm(
    parameters: Iterable[torch.nn.Parameter],
    *,
    max_norm: float,
    eps: float = 1e-7,
) -> None:
    for p in parameters:
        if p is None or not p.requires_grad:
            continue

        t = p.data
        if t.ndim == 1:
            norm = t.norm(p=2)
            if norm > max_norm:
                t.mul_(max_norm / (norm + eps))
        elif t.ndim == 2:
            norms = t.norm(p=2, dim=1, keepdim=True)
            desired = torch.clamp(norms, max=max_norm)
            t.mul_(desired / (norms + eps))
        elif t.ndim == 3:
            norms = t.norm(p=2, dim=(1, 2), keepdim=True)
            desired = torch.clamp(norms, max=max_norm)
            t.mul_(desired / (norms + eps))
        else:
            norm = t.norm(p=2)
            if norm > max_norm:
                t.mul_(max_norm / (norm + eps))

