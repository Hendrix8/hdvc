"""
Repository paths. Set ``HDVC_RESULTS_ROOT`` to override where CSVs and figures live.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def get_results_root() -> Path:
    env = os.environ.get("HDVC_RESULTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "results").resolve()
