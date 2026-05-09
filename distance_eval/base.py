"""Abstract ADC backend and shared L2 composition from IP + norms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def compose_l2(
    q_norms_sq: np.ndarray,
    db_norms_sq: np.ndarray,
    ip: np.ndarray,
) -> np.ndarray:
    """
    ||q - x̂||² = ||q||² + ||x̂||² - 2 <q, x̂>

    q_norms_sq: (nq,) float32
    db_norms_sq: (nb,) float32
    ip: (nq, nb) float32 — inner-product estimate <q, x̂>
    """
    qn = np.asarray(q_norms_sq, dtype=np.float32).reshape(-1, 1)
    dn = np.asarray(db_norms_sq, dtype=np.float32).reshape(1, -1)
    ip_arr = np.asarray(ip, dtype=np.float32)
    return qn + dn - 2.0 * ip_arr


class ADCBackend(ABC):
    """Per-method adapter: encode, DB norms, query prep, IP scan."""

    method_name: str = ""

    @abstractmethod
    def encode(self, db: np.ndarray) -> Any:
        """Encode database rows; returns opaque codes object."""

    @abstractmethod
    def db_norms_sq(self, codes: Any) -> np.ndarray:
        """Per-database-vector ||x̂||², shape (nb,), float32. Not part of per-query ADC."""

    @abstractmethod
    def prepare_query(self, q: np.ndarray) -> Any:
        """Build per-query state (LUTs etc.) for batch q, shape (nq, d)."""

    @abstractmethod
    def ip_estimate(self, qstate: Any, codes: Any) -> np.ndarray:
        """Return (nq, nb) float32 inner products <q, x̂>."""
