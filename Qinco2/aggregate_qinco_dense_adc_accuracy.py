#!/usr/bin/env python3
"""Aggregate dense QINCo ADC accuracy metadata into one CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def iter_rows(out_root: Path):
    for meta_path in sorted((out_root / "dense_adc_distances").rglob("metadata.json")):
        try:
            data = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        rel = data.get("relative_error", {})
        yield {
            "dataset": data.get("dataset"),
            "n_queries": data.get("n_queries"),
            "n_db": data.get("n_db"),
            "dim": data.get("dim"),
            "n_subquantizers": data.get("n_subquantizers"),
            "n_centroids": data.get("n_centroids"),
            "nbits": data.get("nbits"),
            "bits_per_vector": data.get("bits_per_vector"),
            "rel_error_count": rel.get("count"),
            "rel_error_mean": rel.get("mean"),
            "rel_error_std": rel.get("std"),
            "rel_error_min": rel.get("min"),
            "rel_error_max": rel.get("max"),
            "adc_distances_sq": data.get("adc_distances_sq"),
            "exact_distances_sq": data.get("exact_distances_sq"),
            "relative_errors": data.get("relative_errors"),
            "metadata": str(meta_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default="/mnthdd/cpanourg/2-hdvc/results/qinco")
    parser.add_argument(
        "--output",
        default="/mnthdd/cpanourg/2-hdvc/results/qinco/qinco_dense_adc_accuracy_summary.csv",
    )
    args = parser.parse_args()

    rows = list(iter_rows(Path(args.out_root)))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "bits_per_vector", "n_subquantizers", "nbits"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
