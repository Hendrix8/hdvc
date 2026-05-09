"""RaBitQ multi-bit: placeholder until lib/RaBitQ-Library is built and wired."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from distance_eval.base import ADCBackend

_RABITQ_MSG = (
    "RaBitQBackend is not wired to a native multi-bit kernel yet. "
    "Clone/build https://github.com/microsoft/RaBitQ-Library (or your fork) under "
    "lib/RaBitQ-Library, expose encode + IP distance via ctypes/pybind11, then update "
    "distance_eval/backends/rabitq.py. See distance_eval/README.md."
)


class RaBitQBackend(ADCBackend):
    method_name = "RaBitQ"

    def __init__(self, artifact_path: str | Path | None = None):
        self.artifact_path = Path(artifact_path) if artifact_path else None

    def encode(self, db: np.ndarray) -> Any:
        raise RuntimeError(_RABITQ_MSG)

    def db_norms_sq(self, codes: Any) -> np.ndarray:
        raise RuntimeError(_RABITQ_MSG)

    def prepare_query(self, q: np.ndarray) -> Any:
        raise RuntimeError(_RABITQ_MSG)

    def ip_estimate(self, qstate: Any, codes: Any) -> np.ndarray:
        raise RuntimeError(_RABITQ_MSG)
