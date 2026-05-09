"""CPU process-time measurement (same spirit as scripts/evals/measure_adc_cpu_time)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class BenchResult:
    seconds: float


@contextmanager
def cpu_timer() -> Iterator[BenchResult]:
    t0 = time.process_time()
    br = BenchResult(seconds=0.0)
    try:
        yield br
    finally:
        br.seconds = time.process_time() - t0
