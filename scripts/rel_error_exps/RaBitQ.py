#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin launcher: implementation lives in rabitq/eval.py."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rabitq.eval import main

if __name__ == "__main__":
    main()
