#!/usr/bin/env python3
"""
Add `adc_time_2` column (= distance_table_time_s + adc_time_s) to all
evaluation CSVs in a results directory.

This is the *total* asymmetric distance computation time: lookup-table
construction plus the table-based distance scan.

Usage:
  python add_adc_total_time.py                          # default results dir
  python add_adc_total_time.py --results_dir /path/to   # custom dir
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_DIR = Path("/data/cpanourg/2-hdvc/results/relerr_cpp")
COL_NAME = "adc_time_2"


def add_adc_time_2(csv_path: Path) -> bool:
    """Add adc_time_2 = distance_table_time_s + adc_time_s to a CSV.

    Returns True if the file was modified, False if skipped.
    """
    df = pd.read_csv(csv_path)

    if "distance_table_time_s" not in df.columns or "adc_time_s" not in df.columns:
        return False

    df[COL_NAME] = df["distance_table_time_s"] + df["adc_time_s"]
    df.to_csv(csv_path, index=False)
    return True


def main():
    parser = argparse.ArgumentParser(description="Add adc_time_2 column to eval CSVs")
    parser.add_argument(
        "--results_dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory containing evaluation CSVs",
    )
    args = parser.parse_args()

    csv_paths = sorted(args.results_dir.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in {args.results_dir}")
        return

    modified = 0
    for p in csv_paths:
        if add_adc_time_2(p):
            print(f"  ✅ {p.name}")
            modified += 1
        else:
            print(f"  ⏩ {p.name} (missing distance_table_time_s or adc_time_s)")

    print(f"\nDone: {modified}/{len(csv_paths)} files updated with '{COL_NAME}' column.")


if __name__ == "__main__":
    main()
