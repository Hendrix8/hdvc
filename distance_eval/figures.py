#!/usr/bin/env python3
"""Pareto-style plot: relative error vs unified ADC time per query–DB pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import pandas as pd


def plot_unified_timing(
    timing_csv: Path,
    out_png: Path,
    *,
    dataset: str | None = None,
) -> None:
    df = pd.read_csv(timing_csv)
    if dataset:
        df = df[df["dataset"] == dataset]
    if df.empty:
        raise ValueError("No rows to plot")

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, g in df.groupby("method"):
        ax.scatter(
            g["adc_total_time_pp_mean"],
            g["rel_error_mean"],
            label=method,
            alpha=0.75,
            s=36,
        )
    ax.set_xlabel("ADC total time / (nq·nb) [s]")
    ax.set_ylabel("rel_error_mean")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot unified ADC timing")
    p.add_argument(
        "--timing_csv",
        type=Path,
        default=_REPO / "distance_eval" / "results" / "unified_adc_timing.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "distance_eval" / "results" / "unified_adc_pareto.png",
    )
    p.add_argument("--dataset", type=str, default=None)
    args = p.parse_args()
    if not args.timing_csv.is_file():
        print(f"Missing {args.timing_csv}")
        sys.exit(1)
    plot_unified_timing(args.timing_csv, args.output, dataset=args.dataset)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
