#!/usr/bin/env python3
"""
Convert turboquant/result/quant_benchmark_*.csv rows into relerr-style
`*_adc_vs_exact_eval.csv` files for scripts/figure_generators/relerr_cpp_plots.ipynb.

Splits TQ-MSE-* and TQ-Prod-* into separate method families (TQMSE, TQProd).

Usage:
  python export_results_for_figures.py \\
    --turboquant-root /home/cpanourg/projects/2-hdvc/turboquant \\
    --out-dir /data/cpanourg/2-hdvc/results/turboquant
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

METHOD_MAP = {
    "MSE": "TQMSE",
    "Prod": "TQProd",
}

# Columns aligned with RabitQ / relerr_cpp aggregated CSVs (see rabitq/*_RaBitQ_adc_vs_exact_eval.csv)
BASE_COLUMNS = [
    "method",
    "dataset",
    "experiment_folder",
    "nq",
    "nb",
    "nb_sample",
    "dim",
    "n_subquantizers",
    "nbits",
    "bits_per_vector",
    "train_size",
    "train_time_s",
    "encoding_time_s",
    "distance_table_time_s",
    "cdist_time_s",
    "adc_time_s",
    "rel_error_mean",
    "rel_error_std",
    "nq_sample",
    "sample_mode",
    "train_path",
    "seed",
]

EXTRA_COLUMNS = [
    "turboquant_method",
    "distortion",
    "l2_rel_err_median",
    "ip_rel_err_mean",
    "ip_rel_err_median",
    "l2_recall@1",
    "l2_recall@10",
    "l2_recall@100",
    "ip_recall@1",
    "ip_recall@10",
    "ip_recall@100",
]


def parse_turboquant_method(raw: str) -> tuple[str, int] | None:
    """
    TQ-MSE-4bit -> ('MSE', 4); TQ-Prod-12bit -> ('Prod', 12).
    """
    m = re.match(r"^TQ-(MSE|Prod)-(\d+)bit$", raw.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def row_experiment_folder(out_root: Path, dataset: str, short_method: str, nbits: int) -> str:
    return str(out_root / "runs" / dataset / short_method / f"nbits_{nbits}")


def convert_benchmark_csv(path: Path, out_root: Path) -> list[Path]:
    df = pd.read_csv(path)
    if df.empty:
        return []

    out_paths: list[Path] = []
    grouped: dict[tuple[str, str], list[dict]] = {}

    for _, row in df.iterrows():
        parsed = parse_turboquant_method(str(row["method"]))
        if parsed is None:
            continue
        family, nbits = parsed
        short = METHOD_MAP.get(family)
        if short is None:
            continue

        dataset = str(row["dataset"])
        dim = int(row["dim"])
        bits_per_dim = int(row["bits_per_dim"])
        assert bits_per_dim == nbits, (row["method"], bits_per_dim, nbits)

        nb = int(row["nb"])
        nq = int(row["nq"])
        bits_per_vector = bits_per_dim * dim

        rec = {
            "method": short,
            "dataset": dataset,
            "experiment_folder": row_experiment_folder(out_root, dataset, short, nbits),
            "nq": nq,
            "nb": nb,
            "nb_sample": nb,
            "dim": dim,
            "n_subquantizers": 1,
            "nbits": nbits,
            "bits_per_vector": bits_per_vector,
            "train_size": "",
            "train_time_s": "",
            "encoding_time_s": "",
            "distance_table_time_s": "",
            "cdist_time_s": "",
            "adc_time_s": "",
            "rel_error_mean": float(row["l2_rel_err_mean"]),
            "rel_error_std": "",
            "nq_sample": nq,
            "sample_mode": "benchmark",
            "train_path": str(path.resolve()),
            "seed": "",
            "turboquant_method": str(row["method"]),
            "distortion": float(row["distortion"]),
            "l2_rel_err_median": float(row["l2_rel_err_median"]),
            "ip_rel_err_mean": float(row["ip_rel_err_mean"]),
            "ip_rel_err_median": float(row["ip_rel_err_median"]),
            "l2_recall@1": float(row["l2_recall@1"]),
            "l2_recall@10": float(row["l2_recall@10"]),
            "l2_recall@100": float(row["l2_recall@100"]),
            "ip_recall@1": float(row["ip_recall@1"]),
            "ip_recall@10": float(row["ip_recall@10"]),
            "ip_recall@100": float(row["ip_recall@100"]),
        }
        key = (dataset, short)
        grouped.setdefault(key, []).append(rec)

    for (dataset, short), rows in grouped.items():
        sub = pd.DataFrame(rows)
        sub = sub.sort_values("nbits").reset_index(drop=True)
        # Filename pattern: {dataset}_{METHOD}_adc_vs_exact_eval.csv (see relerr_cpp_plots load_relerr_data)
        out_name = f"{dataset}_{short}_adc_vs_exact_eval.csv"
        out_path = out_root / out_name
        cols = BASE_COLUMNS + EXTRA_COLUMNS
        sub[cols].to_csv(out_path, index=False)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    p = argparse.ArgumentParser(description="Export turboquant benchmarks to relerr-style CSVs.")
    p.add_argument(
        "--turboquant-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing result/quant_benchmark_*.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/cpanourg/2-hdvc/results/turboquant"),
        help="Output directory (created if missing).",
    )
    args = p.parse_args()

    result_dir = args.turboquant_root / "result"
    if not result_dir.is_dir():
        raise SystemExit(f"Missing result dir: {result_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_written: list[Path] = []
    for csv_path in sorted(result_dir.glob("quant_benchmark_*.csv")):
        all_written.extend(convert_benchmark_csv(csv_path, args.out_dir))

    # Deduplicate (same dataset+method may appear if glob runs twice — shouldn't)
    unique = sorted(set(all_written), key=lambda x: x.name)
    readme = args.out_dir / "README.txt"
    readme.write_text(
        "Turboquant results exported for relerr_cpp_plots.ipynb-style tooling.\n\n"
        "Files: {dataset}_{TQMSE|TQProd}_adc_vs_exact_eval.csv\n\n"
        "method:\n"
        "  TQMSE  — rows from TQ-MSE-* in quant_benchmark_*.csv\n"
        "  TQProd — rows from TQ-Prod-* in quant_benchmark_*.csv\n\n"
        "rel_error_mean is L2 relative error (l2_rel_err_mean from benchmarks).\n"
        "Extra columns (distortion, IP/L2 recalls) are preserved for custom plots.\n"
        "Timing fields are left empty (not measured in this benchmark).\n"
        "n_subquantizers=1 placeholder so nbits x-axis plots use a single curve family.\n\n"
        "Styled plots (same fonts/grid as relerr_cpp_plots): scripts/figure_generators/\n"
        "  relerr_quick_plot_style.py, turboquant_plots.ipynb (TQMSE + TQProd),\n"
        "  and rabitq_RaBitQ_plots.ipynb — rel_error_mean vs nbits per dataset.\n\n"
        "Regenerate:\n"
        f"  python {Path(__file__).resolve()}\n\n"
        f"Written {len(unique)} CSV(s).\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(unique)} file(s) under {args.out_dir}")
    for path in unique:
        print(f"  {path}")


if __name__ == "__main__":
    main()
