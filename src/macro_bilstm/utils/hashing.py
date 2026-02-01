from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(data: Any, *, n_chars: int = 12) -> str:
    dumped = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    return digest[:n_chars]

