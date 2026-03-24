#!/usr/bin/env python3
"""
Measure ADC CPU time for optimal (n_subquantizers, nbits) per bits_per_vector.

For every unique bits_per_vector = n_subquantizers × nbits, this script:
  1. Finds the (n_subquantizers, nbits) pair with the lowest rel_error_mean.
  2. Loads the corresponding PQ model.
  3. Runs the full ADC pipeline N_RUNS times (default 20) using
     time.process_time() to measure CPU-only time.
  4. Saves per-run and summary statistics to a CSV.

Output CSV columns:
  dataset, n_subquantizers, nbits, bits_per_vector, rel_error_mean,
  run, adc_cpu_time, adc_cpu_time_pp,
  adc_cpu_time_mean, adc_cpu_time_std,
  adc_cpu_time_pp_mean, adc_cpu_time_pp_std

Usage:
  python measure_optimal_adc.py
  python measure_optimal_adc.py --n_runs 30 --max_mem_gb 4
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

from scripts.evals.config import DATASET_CONFIG
from scripts.evals.measure_adc_cpu_time import (
    decode_pq_codes,
    extract_pq,
)
from scripts.evals.run_evals import load_db_and_queries, resolve_model_folder

DEFAULT_RESULTS_DIR = Path("/data/cpanourg/2-hdvc/results/relerr_cpp")
DEFAULT_N_RUNS = 20


def measure_pq_cpu_time(pq, queries, db_sample, max_mem_bytes):
    """Single-run ADC CPU time (distance tables + scan). Returns seconds."""
    nq = queries.shape[0]
    nb = db_sample.shape[0]
    M = pq.M
    nbits_val = pq.nbits
    ksub = 1 << nbits_val

    queries_c = np.ascontiguousarray(queries, dtype=np.float32)
    db_c = np.ascontiguousarray(db_sample, dtype=np.float32)

    codes_flat = pq.compute_codes(db_c).ravel()
    codes_idx = decode_pq_codes(codes_flat, nb, M, nbits_val)

    per_query = M * ksub * 4 + nb * 4
    batch_nq = max(1, min(nq, int(max_mem_bytes / per_query)))

    total_cpu = 0.0
    for start in range(0, nq, batch_nq):
        end = min(start + batch_nq, nq)
        q_batch = np.ascontiguousarray(queries_c[start:end])
        bsz = end - start

        dis_tables = np.empty((bsz, M * ksub), dtype=np.float32)
        t0 = time.process_time()
        pq.compute_distance_tables(
            bsz, faiss.swig_ptr(q_batch), faiss.swig_ptr(dis_tables)
        )
        t1 = time.process_time()
        total_cpu += t1 - t0

        dis_tables_3d = dis_tables.reshape(bsz, M, ksub)
        t0 = time.process_time()
        distances = np.zeros((bsz, nb), dtype=np.float32)
        for j in range(M):
            distances += dis_tables_3d[:, j, codes_idx[:, j]]
        t1 = time.process_time()
        total_cpu += t1 - t0

    return total_cpu


def find_optimal_combos(df):
    """For each bits_per_vector value, return the row with min rel_error_mean."""
    idx = df.groupby("bits_per_vector")["rel_error_mean"].idxmin()
    return df.loc[idx].sort_values("bits_per_vector").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Measure ADC CPU time for optimal (nsubq, nbits) per bits_per_vector"
    )
    parser.add_argument(
        "--results_dir", type=Path, default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--n_runs", type=int, default=DEFAULT_N_RUNS,
        help="Number of ADC measurement repetitions (default: 20)",
    )
    parser.add_argument(
        "--max_mem_gb", type=float, default=2.0,
    )
    parser.add_argument(
        "--nq", type=int, default=None,
        help="Number of queries (default: use value from CSV)",
    )
    parser.add_argument(
        "--nb", type=int, default=None,
        help="Number of database vectors (default: use value from CSV)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: <results_dir>/optimal_adc_timing.csv)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Run only for this dataset (e.g. deep, gist). Default: all datasets.",
    )
    args = parser.parse_args()

    max_mem = int(args.max_mem_gb * 1024**3)
    results_dir = args.results_dir
    output_path = args.output or results_dir / "optimal_adc_timing.csv"
    n_runs = args.n_runs
    override_nq = args.nq
    override_nb = args.nb
    filter_dataset = args.dataset

    csv_paths = sorted(results_dir.glob("*_PQ_adc_vs_exact_eval.csv"))
    if filter_dataset:
        csv_paths = [p for p in csv_paths if p.stem.startswith(f"{filter_dataset}_")]
        if not csv_paths:
            print(f"No CSV found for dataset '{filter_dataset}'")
            return
    if not csv_paths:
        print(f"No *_PQ_adc_vs_exact_eval.csv files in {results_dir}")
        return

    data_cache = {}
    all_rows = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        dataset_name = df["dataset"].iloc[0]

        print(f"\n{'=' * 60}")
        print(f"{dataset_name} — {len(df)} experiments")
        print(f"{'=' * 60}")

        optimal = find_optimal_combos(df)
        print(f"  {len(optimal)} unique bits_per_vector values, optimal combos:")
        for _, r in optimal.iterrows():
            print(
                f"    bits={int(r.bits_per_vector):>5d}  →  "
                f"M={int(r.n_subquantizers):>3d}, nbits={int(r.nbits):>2d}  "
                f"(rel_err={r.rel_error_mean:.6f})"
            )

        if dataset_name not in DATASET_CONFIG:
            print(f"  ⚠️  Unknown dataset: {dataset_name}, skipping")
            continue

        nq = override_nq or int(optimal["nq"].iloc[0])
        nb_sample = override_nb or int(optimal["nb_sample"].iloc[0])
        dim = int(optimal["dim"].iloc[0])

        if dataset_name not in data_cache:
            ds_path, qr_path = DATASET_CONFIG[dataset_name]
            try:
                db, qr = load_db_and_queries(
                    ds_path, qr_path, dim,
                    max_db=nb_sample, max_queries=nq,
                )
                data_cache[dataset_name] = (db, qr)
                print(f"  Loaded data: db={db.shape}, qr={qr.shape}")
            except Exception as e:
                print(f"  ❌ Load data: {e}")
                continue

        db, qr = data_cache[dataset_name]
        qr_sub = qr[:nq].astype(np.float32)
        db_sub = db[:nb_sample].astype(np.float32)
        n_pairs = nq * nb_sample

        for _, row in tqdm(
            optimal.iterrows(), total=len(optimal),
            desc=dataset_name, unit="combo",
        ):
            n_subq = int(row["n_subquantizers"])
            nbits_v = int(row["nbits"])
            bpv = int(row["bits_per_vector"])
            tr_size = int(row.get("train_size", 0))
            exp_folder = Path(row["experiment_folder"])
            rel_err = row["rel_error_mean"]

            try:
                folder = resolve_model_folder(exp_folder, n_subq, nbits_v, tr_size)
                index = faiss.read_index(str(folder / "pq_model.index"))
                pq = extract_pq(index)
            except Exception as e:
                tqdm.write(f"  ❌ bits={bpv} M={n_subq} nbits={nbits_v}: {e}")
                continue

            run_times = []
            for run_i in range(n_runs):
                cpu_t = measure_pq_cpu_time(pq, qr_sub, db_sub, max_mem)
                run_times.append(cpu_t)

            run_times = np.array(run_times)
            pp_times = run_times / n_pairs

            for run_i, (t, pp) in enumerate(zip(run_times, pp_times)):
                all_rows.append({
                    "dataset": dataset_name,
                    "n_subquantizers": n_subq,
                    "nbits": nbits_v,
                    "bits_per_vector": bpv,
                    "rel_error_mean": rel_err,
                    "nq": nq,
                    "nb_sample": nb_sample,
                    "run": run_i,
                    "adc_cpu_time": t,
                    "adc_cpu_time_pp": pp,
                })

            tqdm.write(
                f"  bits={bpv:>5d}  M={n_subq:>3d}  nbits={nbits_v:>2d}  "
                f"mean={run_times.mean():.4f}s ± {run_times.std():.4f}s  "
                f"pp={pp_times.mean():.2e} ± {pp_times.std():.2e}"
            )

    if not all_rows:
        print("No results collected.")
        return

    out_df = pd.DataFrame(all_rows)

    summary = (
        out_df.groupby(["dataset", "n_subquantizers", "nbits", "bits_per_vector", "rel_error_mean"])
        .agg(
            adc_cpu_time_mean=("adc_cpu_time", "mean"),
            adc_cpu_time_std=("adc_cpu_time", "std"),
            adc_cpu_time_pp_mean=("adc_cpu_time_pp", "mean"),
            adc_cpu_time_pp_std=("adc_cpu_time_pp", "std"),
            n_runs=("run", "count"),
        )
        .reset_index()
    )

    out_df.to_csv(output_path, index=False)
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n✅ Per-run results: {output_path}")
    print(f"✅ Summary:         {summary_path}")
    print(f"   {len(out_df)} rows ({len(summary)} combos × {n_runs} runs)")


if __name__ == "__main__":
    main()
