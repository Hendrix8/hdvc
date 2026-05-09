"""Lazy registry of method name → ADCBackend class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from distance_eval.base import ADCBackend

_BACKEND_REGISTRY: dict[str, Type["ADCBackend"]] | None = None


def _ensure_registry() -> dict[str, Type["ADCBackend"]]:
    global _BACKEND_REGISTRY
    if _BACKEND_REGISTRY is not None:
        return _BACKEND_REGISTRY

    from distance_eval.backends.lsq import LSQBackend
    from distance_eval.backends.opq import OPQBackend
    from distance_eval.backends.pq import PQBackend
    from distance_eval.backends.qinco2 import QINCo2Backend
    from distance_eval.backends.rabitq import RaBitQBackend
    from distance_eval.backends.vaq import VAQBackend

    _BACKEND_REGISTRY = {
        "PQ": PQBackend,
        "OPQ": OPQBackend,
        "LSQpp": LSQBackend,
        "VAQ": VAQBackend,
        "RaBitQ": RaBitQBackend,
        "QINCo2": QINCo2Backend,
    }
    return _BACKEND_REGISTRY


def get_backend(name: str) -> Type["ADCBackend"]:
    reg = _ensure_registry()
    key = name.strip()
    if key not in reg:
        raise KeyError(f"Unknown method {key!r}; known: {sorted(reg)}")
    return reg[key]


def backend_names() -> list[str]:
    return sorted(_ensure_registry().keys())
