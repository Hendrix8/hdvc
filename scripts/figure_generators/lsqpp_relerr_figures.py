#!/usr/bin/env python3
"""
LSQ++ — relative error vs nbits (same style as turboquant_plots.ipynb).

Loads ``*_*_LSQpp_adc_vs_exact_eval.csv`` (or ``*_LSQpp_adc_vs_exact_eval.csv``) from a
results directory. Each aggregate file can contain a full ``M × nbits × train_size``
grid; by default we filter to one ``train_size`` and draw **one curve per**
``n_subquantizers`` (M), matching how PQ/OPQ sweeps are read for line plots.

Example::

    python3 scripts/figure_generators/lsqpp_relerr_figures.py \\
        --data-dir /data/cpanourg/2-hdvc/results \\
        --train-size 100000

Figures go under ``{data_dir}/figures/LSQpp/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_GEN = Path(__file__).resolve().parent
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

from relerr_quick_plot_style import figure_relerr_vs_nbits_multi, savefig_relerr


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path)


def build_series_by_m(df: pd.DataFrame, train_size: int) -> list[tuple[pd.DataFrame, str]]:
    """One (sorted-by-nbits) slice per n_subquantizers value."""
    d = df[df["train_size"] == train_size].copy()
    if d.empty:
        return []
    series: list[tuple[pd.DataFrame, str]] = []
    for m in sorted(d["n_subquantizers"].unique()):
        sub = d[d["n_subquantizers"] == m].sort_values("nbits")
        if sub.empty:
            continue
        series.append((sub, f"M={int(m)}"))
    return series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/cpanourg/2-hdvc/results"),
        help="Directory that contains *_LSQpp_adc_vs_exact_eval.csv",
    )
    ap.add_argument(
        "--train-size",
        type=int,
        default=100_000,
        help="Keep only runs with this train_size (default: 100000).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override figure output root (default: {data_dir}/figures/LSQpp).",
    )
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    out_root = args.output_dir if args.output_dir is not None else data_dir / "figures" / "LSQpp"

    paths = sorted(data_dir.glob("*_*_LSQpp_adc_vs_exact_eval.csv"))
    if not paths:
        paths = sorted(data_dir.glob("*_LSQpp_adc_vs_exact_eval.csv"))
    if not paths:
        print(f"No LSQ++ aggregate CSVs under {data_dir} (glob *_LSQpp_adc_vs_exact_eval.csv).")
        sys.exit(1)

    for p in paths:
        df = _read_csv(p)
        need = {"dataset", "nbits", "rel_error_mean", "n_subquantizers", "train_size"}
        if not need.issubset(df.columns):
            print(f"Skip {p.name}: missing columns (need {sorted(need)})")
            continue

        series = build_series_by_m(df, args.train_size)
        if not series:
            sizes = sorted(df["train_size"].unique()) if "train_size" in df.columns else []
            print(
                f"Skip {p.name}: no rows with train_size={args.train_size}. "
                f"Available train_size values: {sizes}"
            )
            continue

        ds = str(df["dataset"].iloc[0])
        title = f"{ds}  (train_size={args.train_size})"
        fig, _ax = figure_relerr_vs_nbits_multi(series, title=title)
        out = out_root / f"{ds}_LSQpp_relerr_vs_nbits_train{args.train_size}.pdf"
        savefig_relerr(fig, out)
        print(out)


if __name__ == "__main__":
    main()
