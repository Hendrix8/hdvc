#!/usr/bin/env python3
"""Aggregate PQ FAISS ADC timing + dense relative-error metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def iter_rows(out_root: Path):
    for meta_path in sorted((out_root / "pq_faiss_adc").rglob("metadata.json")):
        try:
            data = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        rel = data.get("relative_error", {})
        timing = data.get("faiss_adc_timing", {})
        yield {
            "method": "PQ",
            "dataset": data.get("dataset"),
            "row_index": data.get("row_index"),
            "nq": data.get("n_queries"),
            "nb_sample": data.get("n_db"),
            "dim": data.get("dim"),
            "n_subquantizers": data.get("n_subquantizers"),
            "nbits": data.get("nbits"),
            "n_centroids": data.get("n_centroids"),
            "bits_per_vector": data.get("bits_per_vector"),
            "train_size": data.get("train_size"),
            "adc_time_s": timing.get("adc_time_s_mean"),
            "adc_time_s_std": timing.get("adc_time_s_std"),
            "per_pair_adc_time_ns": timing.get("per_pair_adc_time_ns_mean"),
            "per_pair_adc_time_ns_std": timing.get("per_pair_adc_time_ns_std"),
            "rel_error_count": rel.get("count"),
            "rel_error_mean": rel.get("mean"),
            "rel_error_std": rel.get("std"),
            "rel_error_min": rel.get("min"),
            "rel_error_max": rel.get("max"),
            "adc_distances_sq": data.get("adc_distances_sq"),
            "exact_distances_sq": data.get("exact_distances_sq"),
            "relative_errors": data.get("relative_errors"),
            "artifact_index": data.get("artifact_index"),
            "metadata": str(meta_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default="/mnthdd/cpanourg/2-hdvc/results/pq")
    parser.add_argument("--output", default="/mnthdd/cpanourg/2-hdvc/results/pq/pq_faiss_adc_summary.csv")
    args = parser.parse_args()

    df = pd.DataFrame(iter_rows(Path(args.out_root)))
    if not df.empty:
        df = df.sort_values(["dataset", "bits_per_vector", "n_subquantizers", "nbits", "row_index"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
