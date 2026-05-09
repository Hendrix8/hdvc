#!/usr/bin/env python3
"""
DEPRECATED — use ``python -m distance_eval.harness`` (see ``distance_eval/README.md``).
Kept as PQ-only legacy reference.

Measure ADC timing with CPU process time (time.process_time).

time.process_time() counts only the CPU time consumed by the current
process, eliminating noise from other processes, I/O waits, context
switches, etc.  This is the Python equivalent of CLOCK_PROCESS_CPUTIME_ID.

For each PQ experiment row this script:
  1. Loads the trained PQ model (IndexPQ) from experiment_folder.
  2. Loads query vectors (nq) and a database sample (nb_sample).
  3. Times distance-table construction   (pq.compute_distance_tables).
  4. Times the ADC distance scan          (table lookups over codes).
  Both steps are measured with time.process_time().

New columns (per-pair = divided by nq × nb_sample):
  - adc_cpu_time    : total CPU time (seconds) for distance tables + ADC scan
  - adc_cpu_time_pp : per-pair CPU time (seconds per query-db pair)

Usage:
  python measure_adc_cpu_time.py
  python measure_adc_cpu_time.py --results_dir /path/to/results
  python measure_adc_cpu_time.py --max_mem_gb 4
"""

import argparse
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evals.config import DATASET_CONFIG, get_results_root
from scripts.evals.run_evals import load_db_and_queries, resolve_model_folder

DEFAULT_RESULTS_DIR = get_results_root() / "relerr_cpp"
CPU_COLS = ["adc_cpu_time", "adc_cpu_time_pp"]


# ── helpers ──────────────────────────────────────────────────────────────

def extract_pq(index):
    """Extract the ProductQuantizer from an IndexPQ or IndexPreTransform."""
    if hasattr(index, "pq"):
        return index.pq
    if hasattr(index, "index"):
        sub = faiss.downcast_index(index.index)
        if hasattr(sub, "pq"):
            return sub.pq
    raise ValueError("Cannot extract ProductQuantizer from index")


def decode_pq_codes(codes_flat, nb, M, nbits):
    """Decode FAISS packed PQ codes → (nb, M) centroid-index array.

    Handles arbitrary nbits (4, 6, 8, 10, 12, …).
    """
    if nbits == 8:
        return codes_flat.reshape(nb, M).astype(np.int32)

    code_size = (M * nbits + 7) // 8
    codes_2d = codes_flat.reshape(nb, code_size)
    mask = (1 << nbits) - 1
    result = np.zeros((nb, M), dtype=np.int32)

    for j in range(M):
        bit_start = j * nbits
        byte_idx = bit_start >> 3
        bit_off = bit_start & 7

        val = codes_2d[:, byte_idx].astype(np.int32) >> bit_off
        bits_read = 8 - bit_off

        while bits_read < nbits:
            byte_idx += 1
            if byte_idx < code_size:
                val |= codes_2d[:, byte_idx].astype(np.int32) << bits_read
            bits_read += 8

        result[:, j] = val & mask

    return result


# ── core measurement ────────────────────────────────────────────────────

def measure_pq_cpu_time(pq, queries, db_sample, max_mem_bytes):
    """Return total CPU seconds for (distance-table + ADC scan).

    Uses time.process_time() so only actual CPU cycles are counted.
    Processes queries in batches to stay within *max_mem_bytes*.
    """
    nq = queries.shape[0]
    nb = db_sample.shape[0]
    M = pq.M
    nbits_val = pq.nbits
    ksub = 1 << nbits_val

    queries_c = np.ascontiguousarray(queries, dtype=np.float32)
    db_c = np.ascontiguousarray(db_sample, dtype=np.float32)

    # Encode database → packed codes → centroid indices
    codes_flat = pq.compute_codes(db_c).ravel()
    codes_idx = decode_pq_codes(codes_flat, nb, M, nbits_val)

    # Batch size: keep distance tables + output distances within budget
    per_query = M * ksub * 4 + nb * 4  # bytes
    batch_nq = max(1, int(max_mem_bytes / per_query))
    batch_nq = min(batch_nq, nq)

    total_cpu = 0.0

    for start in range(0, nq, batch_nq):
        end = min(start + batch_nq, nq)
        q_batch = np.ascontiguousarray(queries_c[start:end])
        bsz = end - start

        # Distance-table construction (FAISS C++ via SWIG)
        dis_tables = np.empty((bsz, M * ksub), dtype=np.float32)
        t0 = time.process_time()
        pq.compute_distance_tables(
            bsz, faiss.swig_ptr(q_batch), faiss.swig_ptr(dis_tables)
        )
        t1 = time.process_time()
        total_cpu += t1 - t0

        # ADC scan: accumulate partial distances from lookup tables
        dis_tables_3d = dis_tables.reshape(bsz, M, ksub)
        t0 = time.process_time()
        distances = np.zeros((bsz, nb), dtype=np.float32)
        for j in range(M):
            distances += dis_tables_3d[:, j, codes_idx[:, j]]
        t1 = time.process_time()
        total_cpu += t1 - t0

    return total_cpu


# ── propagation to derived CSVs ─────────────────────────────────────────

def propagate_cpu_columns(results_dir):
    """Copy CPU-time columns from PQ adc_vs_exact_eval CSVs to derived CSVs."""
    merge_on = ["dataset", "method", "experiment_folder", "n_subquantizers", "nbits"]

    for base_csv in sorted(results_dir.glob("*_PQ_adc_vs_exact_eval.csv")):
        base_df = pd.read_csv(base_csv)
        if not all(c in base_df.columns for c in CPU_COLS):
            continue

        prefix = base_csv.stem.replace("_adc_vs_exact_eval", "")
        for suffix in ("_compression_rate", "_reconstruction_error", "_spearman", "_recall"):
            derived = results_dir / f"{prefix}{suffix}.csv"
            if not derived.exists():
                continue

            ddf = pd.read_csv(derived)
            ddf = ddf.drop(columns=[c for c in CPU_COLS if c in ddf.columns])
            usable = [c for c in merge_on if c in base_df.columns and c in ddf.columns]
            if not usable:
                continue

            lookup = base_df[usable + CPU_COLS].drop_duplicates()
            ddf = ddf.merge(lookup, on=usable, how="left")
            ddf.to_csv(derived, index=False)
            print(f"  ✅ {derived.name}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure ADC timing with CPU process time (PQ only)"
    )
    parser.add_argument(
        "--results_dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory with evaluation CSVs",
    )
    parser.add_argument(
        "--max_mem_gb", type=float, default=2.0,
        help="Memory budget (GB) for distance-table batching (default: 2)",
    )
    args = parser.parse_args()

    max_mem = int(args.max_mem_gb * 1024**3)
    results_dir = args.results_dir

    # PQ only
    csv_paths = sorted(results_dir.glob("*_PQ_adc_vs_exact_eval.csv"))
    if not csv_paths:
        print(f"No *_PQ_adc_vs_exact_eval.csv files in {results_dir}")
        return

    data_cache = {}  # dataset → (db, qr)

    for csv_path in csv_paths:
        print(f"\n{'=' * 60}")
        print(f"Processing {csv_path.name}")
        print(f"{'=' * 60}")

        df = pd.read_csv(csv_path)
        for c in CPU_COLS:
            df[c] = np.nan

        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=csv_path.stem, unit="exp"
        ):
            dataset_name = row["dataset"]
            exp_folder = Path(row["experiment_folder"])
            nq = int(row["nq"])
            nb_sample = int(row["nb_sample"])
            dim = int(row["dim"])
            n_subq = int(row.get("n_subquantizers", 0))
            nbits_v = int(row.get("nbits", 0))
            tr_size = int(row.get("train_size", 0))

            if dataset_name not in DATASET_CONFIG:
                tqdm.write(f"  ⚠️  Unknown dataset: {dataset_name}")
                continue

            # Load vectors once per dataset
            if dataset_name not in data_cache:
                ds_path, qr_path = DATASET_CONFIG[dataset_name]
                try:
                    db, qr = load_db_and_queries(
                        ds_path, qr_path, dim,
                        max_db=nb_sample, max_queries=nq,
                    )
                    data_cache[dataset_name] = (db, qr)
                    tqdm.write(f"  Loaded {dataset_name}: db={db.shape}, qr={qr.shape}")
                except Exception as e:
                    tqdm.write(f"  ❌ Load data ({dataset_name}): {e}")
                    continue

            db, qr = data_cache[dataset_name]

            # Load trained PQ model
            try:
                folder = resolve_model_folder(exp_folder, n_subq, nbits_v, tr_size)
                index = faiss.read_index(str(folder / "pq_model.index"))
            except Exception as e:
                tqdm.write(f"  ❌ Model: {e}")
                continue

            try:
                pq = extract_pq(index)
            except ValueError as e:
                tqdm.write(f"  ❌ Extract PQ: {e}")
                continue

            qr_sub = qr[:nq].astype(np.float32)
            db_sub = db[:nb_sample].astype(np.float32)

            try:
                total_cpu = measure_pq_cpu_time(pq, qr_sub, db_sub, max_mem)
                n_pairs = nq * nb_sample
                pp = total_cpu / n_pairs

                df.at[idx, "adc_cpu_time"] = total_cpu
                df.at[idx, "adc_cpu_time_pp"] = pp
                tqdm.write(
                    f"  {exp_folder.name}: "
                    f"total={total_cpu:.4f}s  "
                    f"pp={pp:.2e}s  "
                    f"(M={n_subq}, nbits={nbits_v})"
                )
            except Exception as e:
                tqdm.write(f"  ❌ Measure: {e}")

        df.to_csv(csv_path, index=False)
        print(f"✅ Updated {csv_path.name}")

    # Propagate to derived metric CSVs
    print(f"\nPropagating CPU-time columns to derived CSVs …")
    propagate_cpu_columns(results_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
