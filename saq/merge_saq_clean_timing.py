#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLS = ["dataset", "n_centroids", "nbits", "caq_adj_rd_lmt", "searcher_vars_bound_m"]
TIMING_COLS = [
    "experiment_folder",
    "train_time_s",
    "encoding_time_s",
    "adc_time_s",
    "adc_time_s_std",
    "per_query_us_mean",
    "per_pair_ns_mean",
    "centroid_fvecs",
    "cluster_ids_ivecs",
    "eval_csv",
    "index_csv",
]


def merge_dataset(
    full_csv: Path,
    timing_csv: Path,
    out_csv: Path,
    caq_adj_rd_lmt: int,
    searcher_vars_bound_m: float,
) -> None:
    full_df = pd.read_csv(full_csv)
    timing_df = pd.read_csv(timing_csv)

    full_slice = full_df[
        (full_df["caq_adj_rd_lmt"] == caq_adj_rd_lmt)
        & (full_df["searcher_vars_bound_m"] == searcher_vars_bound_m)
    ].copy()

    if len(full_slice) != len(timing_df):
        raise ValueError(
            f"Row-count mismatch for {full_csv.name}: "
            f"full slice={len(full_slice)} timing={len(timing_df)}"
        )

    merged = full_slice.merge(
        timing_df[KEY_COLS + TIMING_COLS],
        on=KEY_COLS,
        how="inner",
        suffixes=("", "_timing"),
        validate="one_to_one",
    )

    if len(merged) != len(full_slice):
        raise ValueError(f"Key mismatch while merging {full_csv.name}")

    for col in TIMING_COLS:
        merged[col] = merged[f"{col}_timing"]
        merged.drop(columns=[f"{col}_timing"], inplace=True)

    merged["timing_rerun"] = True
    merged["timing_source_csv"] = str(timing_csv)
    merged["relerr_source_csv"] = str(full_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge SAQ full relerr CSVs with timing-rerun timing columns."
    )
    p.add_argument(
        "--full-dir",
        type=Path,
        default=Path("/home/cpanourg/projects/2-hdvc/results/saq"),
    )
    p.add_argument(
        "--timing-dir",
        type=Path,
        default=Path("/home/cpanourg/projects/2-hdvc/results_timing/saq"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/cpanourg/projects/2-hdvc/results/saq_clean_timing"),
    )
    p.add_argument("--caq-adj-rd-lmt", type=int, default=6)
    p.add_argument("--searcher-vars-bound-m", type=float, default=4.0)
    args = p.parse_args()

    full_csvs = sorted(args.full_dir.glob("*_SAQ_adc_vs_exact_eval.csv"))
    if not full_csvs:
        raise FileNotFoundError(f"No SAQ full CSVs found in {args.full_dir}")

    for full_csv in full_csvs:
        timing_csv = args.timing_dir / full_csv.name
        if not timing_csv.exists():
            raise FileNotFoundError(f"Missing timing CSV: {timing_csv}")
        out_csv = args.out_dir / full_csv.name
        merge_dataset(
            full_csv=full_csv,
            timing_csv=timing_csv,
            out_csv=out_csv,
            caq_adj_rd_lmt=args.caq_adj_rd_lmt,
            searcher_vars_bound_m=args.searcher_vars_bound_m,
        )
        print(out_csv)


if __name__ == "__main__":
    main()
