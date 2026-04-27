#!/usr/bin/env python3
"""
All-methods relative-error figures from exported ``*_adc_vs_exact_eval.csv`` files.

Two outputs (see ``--mode``):

1. **bar** — Grouped bars per dataset (same logic as ``method_comparison_relerr.ipynb``):
   for each method, pick the row whose ``bits_per_vector`` is closest to
   ``--target-bpv`` (optional max distance).

2. **curves** — **Deep1B only**: one curve per method, ``rel_error_mean`` vs
   ``bits_per_vector``. PQ / OPQ / LSQ++ use a fixed subspace count ``M`` and
   training size so curves are comparable; TQMSE / TQProd / RaBitQ / VAQ use
   whatever is in their CSV (see ``--help`` for defaults).

Example::

    python3 scripts/figure_generators/all_methods_relerr_figures.py \\
        --data-root /data/cpanourg/2-hdvc/results \\
        --mode both \\
        --target-bpv 256 512 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_GEN = Path(__file__).resolve().parent
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

from method_comparison_relerr import (
    build_comparison_table,
    default_method_sources,
    load_concat_sources,
    plot_grouped_bar_relerr,
)
from relerr_quick_plot_style import figure_relerr_vs_bits_per_vector_multi, savefig_relerr


DEFAULT_DATASETS_ORDER = (
    "bigann",
    "deep",
    "gist",
    "msmarco",
    "openai",
    "sift",
)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path)


def _filter_train(df: pd.DataFrame, train_size: int | None) -> pd.DataFrame:
    if train_size is None or "train_size" not in df.columns:
        return df
    ts = pd.to_numeric(df["train_size"], errors="coerce")
    return df[ts == float(train_size)]


def build_deep_curve_series(
    data_root: Path,
    *,
    n_subq: int,
    pq_opq_lsq_train: int,
    rabitq_train: int,
    methods: list[str],
) -> list[tuple[pd.DataFrame, str]]:
    """(bits_per_vector, rel_error_mean) slices for Deep, one entry per method label."""
    series: list[tuple[pd.DataFrame, str]] = []

    def add_from_path(path: Path, label: str, slicer) -> None:
        if not path.is_file():
            return
        raw = _read_csv(path)
        if "dataset" not in raw.columns or ("deep" not in raw["dataset"].values):
            return
        d = raw[raw["dataset"] == "deep"].copy()
        d = slicer(d)
        if d.empty or "bits_per_vector" not in d.columns:
            return
        d = d.sort_values("bits_per_vector")
        series.append((d[["bits_per_vector", "rel_error_mean"]], label))

    rel = data_root / "relerr_cpp"
    tq = data_root / "turboquant"
    rb = data_root / "rabitq"
    vq = data_root / "vaq"

    if "PQ" in methods:
        add_from_path(
            rel / "deep_PQ_adc_vs_exact_eval.csv",
            f"PQ (M={n_subq}, train={pq_opq_lsq_train})",
            lambda d: _filter_train(
                d[(d["n_subquantizers"] == n_subq)],
                pq_opq_lsq_train,
            ),
        )
    if "OPQ" in methods:
        add_from_path(
            rel / "deep_OPQ_adc_vs_exact_eval.csv",
            f"OPQ (M={n_subq}, train={pq_opq_lsq_train})",
            lambda d: _filter_train(
                d[(d["n_subquantizers"] == n_subq)],
                pq_opq_lsq_train,
            ),
        )
    if "LSQ++" in methods:
        for lsq_path in sorted(data_root.glob("*_LSQpp_adc_vs_exact_eval.csv")):
            add_from_path(
                lsq_path,
                f"LSQ++ (M={n_subq}, train={pq_opq_lsq_train})",
                lambda d: _filter_train(
                    d[(d["method"] == "LSQpp") & (d["n_subquantizers"] == n_subq)],
                    pq_opq_lsq_train,
                ),
            )
            break

    if "TQMSE" in methods:
        add_from_path(
            tq / "deep_TQMSE_adc_vs_exact_eval.csv",
            "TQMSE",
            lambda d: d,
        )
    if "TQProd" in methods:
        add_from_path(
            tq / "deep_TQProd_adc_vs_exact_eval.csv",
            "TQProd",
            lambda d: d,
        )
    if "RaBitQ" in methods:
        add_from_path(
            rb / "deep_RaBitQ_adc_vs_exact_eval.csv",
            f"RaBitQ (train={rabitq_train})",
            lambda d: _filter_train(d, rabitq_train),
        )
    if "VAQ" in methods:
        add_from_path(
            vq / "deep_VAQ_adc_vs_exact_eval.csv",
            "VAQ",
            lambda d: d[d["method"] == "VAQ"] if "method" in d.columns else d,
        )

    return series


def run_bar_charts(
    *,
    data_root: Path,
    out_dir: Path,
    target_bpvs: list[float],
    datasets: list[str],
    methods: list[str],
    max_abs_bpv_diff: float | None,
    lsq_train_size_filter: int | None,
) -> None:
    sources = default_method_sources(data_root, lsq_train_size_filter=lsq_train_size_filter)
    raw = load_concat_sources(sources)
    if raw.empty:
        print("No rows loaded; check --data-root and CSV layout.")
        return
    print(f"Loaded {len(raw)} rows, methods: {sorted(raw['method'].unique())}")

    present = set(raw["dataset"].unique())
    ds_use = [d for d in datasets if d in present]
    missing = [d for d in datasets if d not in present]
    if missing:
        print("Datasets with no rows in merged CSV (skipped):", missing)

    for target in target_bpvs:
        comp = build_comparison_table(
            raw,
            ds_use,
            methods,
            float(target),
            max_abs_bpv_diff=max_abs_bpv_diff,
        )
        plot_grouped_bar_relerr(
            comp,
            methods,
            target_bits_per_vector=target,
            title_suffix=f"data_root={data_root}",
            output_path=out_dir / "bar" / f"relerr_all_methods_bpv~{int(target)}.pdf",
            close_fig=True,
        )
        print(out_dir / "bar" / f"relerr_all_methods_bpv~{int(target)}.pdf")


def run_curves(
    *,
    data_root: Path,
    out_dir: Path,
    methods: list[str],
    n_subq: int,
    pq_opq_lsq_train: int,
    rabitq_train: int,
) -> None:
    series = build_deep_curve_series(
        data_root,
        n_subq=n_subq,
        pq_opq_lsq_train=pq_opq_lsq_train,
        rabitq_train=rabitq_train,
        methods=methods,
    )
    if not series:
        print("Curve mode: no Deep slices produced (missing CSVs or empty filters).")
        return
    title = (
        f"Deep — rel. error vs bits/vector\n"
        f"PQ/OPQ/LSQ++: M={n_subq}, train={pq_opq_lsq_train} | "
        f"RaBitQ: train={rabitq_train}"
    )
    fig, _ax = figure_relerr_vs_bits_per_vector_multi(series, title=title)
    outp = out_dir / "curves" / "deep_all_methods_relerr_vs_bits_per_vector.pdf"
    savefig_relerr(fig, outp)
    print(outp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/cpanourg/2-hdvc/results"),
        help="Root directory containing turboquant/, relerr_cpp/, rabitq/, vaq/, and *_LSQpp_*.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: {data_root}/figures/all_methods)",
    )
    ap.add_argument("--mode", choices=("bar", "curves", "both"), default="both")
    ap.add_argument(
        "--target-bpv",
        type=float,
        nargs="+",
        default=[256.0, 512.0, 1024.0],
        help="Target bits/vector for bar charts (one PDF each).",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS_ORDER),
        help="Comma-separated dataset names (order preserved where data exists).",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="TQMSE,TQProd,RaBitQ,VAQ,PQ,OPQ,LSQ++",
        help="Comma-separated method labels (must match normalized CSV method names).",
    )
    ap.add_argument(
        "--max-abs-bpv-diff",
        type=float,
        default=None,
        help="If set, drop bar-chart cells whose closest row is farther than this from --target-bpv.",
    )
    ap.add_argument(
        "--lsq-train-size",
        type=int,
        default=1_000_000,
        help="For bar charts: keep only LSQ++ rows with this train_size (set -1 to disable filter).",
    )
    ap.add_argument(
        "--curve-m",
        type=int,
        default=32,
        help="Curves mode: n_subquantizers for PQ / OPQ / LSQ++ on Deep.",
    )
    ap.add_argument(
        "--curve-pq-opq-lsq-train",
        type=int,
        default=1_000_000,
        help="Curves mode: train_size for PQ / OPQ / LSQ++ Deep slices.",
    )
    ap.add_argument(
        "--curve-rabitq-train",
        type=int,
        default=100_000,
        help="Curves mode: train_size filter for RaBitQ Deep (if column present).",
    )
    args = ap.parse_args()

    data_root: Path = args.data_root
    out_dir = args.out_dir if args.out_dir is not None else data_root / "figures" / "all_methods"
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    lsq_f = None if args.lsq_train_size < 0 else int(args.lsq_train_size)

    if args.mode in ("bar", "both"):
        run_bar_charts(
            data_root=data_root,
            out_dir=out_dir,
            target_bpvs=list(args.target_bpv),
            datasets=datasets,
            methods=methods,
            max_abs_bpv_diff=args.max_abs_bpv_diff,
            lsq_train_size_filter=lsq_f,
        )

    if args.mode in ("curves", "both"):
        run_curves(
            data_root=data_root,
            out_dir=out_dir,
            methods=methods,
            n_subq=int(args.curve_m),
            pq_opq_lsq_train=int(args.curve_pq_opq_lsq_train),
            rabitq_train=int(args.curve_rabitq_train),
        )


if __name__ == "__main__":
    main()
