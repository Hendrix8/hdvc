#!/usr/bin/env python3
"""
DEPRECATED — use ``python -m distance_eval.manifest`` + ``python -m distance_eval.harness``
(see ``distance_eval/README.md``). This script remains as a PQ-only legacy reference.

Measure ADC CPU time for ALL PQ experiments (not just optimal per bits_per_vector).

For every row in *_PQ_adc_vs_exact_eval.csv:
  1. Loads the corresponding PQ model.
  2. Runs the full ADC pipeline N_RUNS times using time.process_time() (CPU-only).
  3. Computes mean and std over runs.
  4. Saves per-experiment results and aggregated summaries (per nbits, per n_subquantizers).

Output files:
  - all_adc_timing.csv: per-experiment (dataset, M, nbits, adc_mean, adc_std, ...)
  - all_adc_timing_by_nbits.csv: aggregated by nbits (one row per dataset,nbits)
  - all_adc_timing_by_nsubq.csv: aggregated by n_subquantizers (one row per dataset,M)

Usage:
  python measure_all_adc.py
  python measure_all_adc.py --dataset deep --n_runs 20 --nq 1000 --nb 10000
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
from scripts.evals.measure_adc_cpu_time import decode_pq_codes, extract_pq
from scripts.evals.run_evals import load_db_and_queries, resolve_model_folder

DEFAULT_RESULTS_DIR = get_results_root() / "relerr_cpp"
DEFAULT_OUTPUT_DIR = get_results_root()
DEFAULT_N_RUNS = 5


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


def main():
    parser = argparse.ArgumentParser(
        description="Measure ADC CPU time for all PQ experiments"
    )
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--n_runs", type=int, default=DEFAULT_N_RUNS,
        help="Number of ADC measurement repetitions per experiment",
    )
    parser.add_argument("--max_mem_gb", type=float, default=2.0)
    parser.add_argument(
        "--nq", type=int, default=None,
        help="Number of queries (default: from CSV)",
    )
    parser.add_argument(
        "--nb", type=int, default=None,
        help="Number of database vectors (default: from CSV)",
    )
    parser.add_argument(
        "--dataset", type=str, default='deep',
        help="Run only for this dataset (e.g. deep, gist). Default: all.",
    )
    parser.add_argument(
        "--nbits", type=int, nargs="*", default=None,
        help="Filter to these nbits only (e.g. --nbits 4 6 8). Default: all.",
    )
    parser.add_argument(
        "--n_subq", type=int, nargs="*", default=[1, 4, 12, 24, 32, 96],
        help="Filter to these n_subquantizers only (e.g. --n_subq 1 4 8 32). Default: 1 4 12 24 32 96.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Output directory for CSV results (default: HDVC results root from hdvc_paths)",
    )
    parser.add_argument(
        "--output_suffix", type=str, default=None,
        help="Write to all_adc_timing_{suffix}.csv (for parallel runs). Omit to write to all_adc_timing.csv.",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all all_adc_timing_*.csv in output_dir and write all_adc_timing.csv + aggregated.",
    )
    args = parser.parse_args()

    max_mem = int(args.max_mem_gb * 1024**3)
    results_dir = args.results_dir
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        parts = sorted(output_dir.glob("all_adc_timing_*.csv"))
        if not parts:
            print(f"No all_adc_timing_*.csv found in {output_dir}")
            return
        out_df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        out_df = out_df.drop_duplicates(
            subset=["dataset", "n_subquantizers", "nbits"], keep="last"
        )
        out_path = output_dir / "all_adc_timing.csv"
        out_df.to_csv(out_path, index=False)
        print(f"✅ Merged {len(parts)} files → {out_path} ({len(out_df)} rows)")
        by_nbits = (
            out_df.groupby(["dataset", "nbits"])
            .agg(
                adc_cpu_time_pp_mean=("adc_cpu_time_pp_mean", "mean"),
                adc_cpu_time_pp_std=("adc_cpu_time_pp_mean", "std"),
                n_experiments=("n_subquantizers", "count"),
            )
            .reset_index()
        )
        by_nbits["adc_cpu_time_pp_std"] = by_nbits["adc_cpu_time_pp_std"].fillna(0)
        by_nbits.to_csv(output_dir / "all_adc_timing_by_nbits.csv", index=False)
        by_nsubq = (
            out_df.groupby(["dataset", "n_subquantizers"])
            .agg(
                adc_cpu_time_pp_mean=("adc_cpu_time_pp_mean", "mean"),
                adc_cpu_time_pp_std=("adc_cpu_time_pp_mean", "std"),
                n_experiments=("nbits", "count"),
            )
            .reset_index()
        )
        by_nsubq["adc_cpu_time_pp_std"] = by_nsubq["adc_cpu_time_pp_std"].fillna(0)
        by_nsubq.to_csv(output_dir / "all_adc_timing_by_nsubq.csv", index=False)
        print(f"✅ Aggregated: by_nbits, by_nsubq")
        return

    n_runs = args.n_runs
    override_nq = args.nq
    override_nb = args.nb
    filter_dataset = args.dataset
    filter_nbits = set(args.nbits) if args.nbits else None
    filter_nsubq = set(args.n_subq) if args.n_subq else None

    csv_paths = sorted(results_dir.glob("*_PQ_adc_vs_exact_eval.csv"))
    if filter_dataset:
        csv_paths = [p for p in csv_paths if p.stem.startswith(f"{filter_dataset}_")]
    if not csv_paths:
        print(f"No *_PQ_adc_vs_exact_eval.csv found in {results_dir}")
        return

    data_cache = {}
    all_rows = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        dataset_name = df["dataset"].iloc[0]

        print(f"\n{'=' * 60}")
        print(f"{dataset_name} — {len(df)} experiments")
        print(f"{'=' * 60}")

        if dataset_name not in DATASET_CONFIG:
            print(f"  ⚠️  Unknown dataset: {dataset_name}, skipping")
            continue

        nq = override_nq or int(df["nq"].iloc[0])
        nb_sample = override_nb or int(df["nb_sample"].iloc[0])
        dim = int(df["dim"].iloc[0])

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

        # Keep only (nbits × n_subq) combinations in the filters (saves time)
        df_work = df.copy()
        if filter_nbits is not None:
            df_work = df_work[df_work["nbits"].astype(int).isin(filter_nbits)]
        if filter_nsubq is not None:
            df_work = df_work[df_work["n_subquantizers"].astype(int).isin(filter_nsubq)]
        n_to_run = len(df_work)
        if n_to_run == 0:
            print(f"  No experiments match filters (nbits={filter_nbits}, n_subq={filter_nsubq})")
            continue
        print(f"  Running {n_to_run} experiments (filtered from {len(df)} in CSV)")

        for _, row in tqdm(
            df_work.iterrows(), total=n_to_run, desc=dataset_name, unit="exp",
        ):
            n_subq = int(row["n_subquantizers"])
            nbits_v = int(row["nbits"])
            bpv = int(row["bits_per_vector"])
            tr_size = int(row.get("train_size", 1000000))
            rel_err = row.get("rel_error_mean", np.nan)
            exp_folder = Path(row["experiment_folder"])

            try:
                folder = resolve_model_folder(exp_folder, n_subq, nbits_v, tr_size)
                index = faiss.read_index(str(folder / "pq_model.index"))
                pq = extract_pq(index)
            except Exception as e:
                tqdm.write(f"  ❌ M={n_subq} nbits={nbits_v}: {e}")
                continue

            run_times = []
            for _ in range(n_runs):
                cpu_t = measure_pq_cpu_time(pq, qr_sub, db_sub, max_mem)
                run_times.append(cpu_t)

            run_times = np.array(run_times)
            pp_times = run_times / n_pairs

            all_rows.append({
                "dataset": dataset_name,
                "n_subquantizers": n_subq,
                "nbits": nbits_v,
                "bits_per_vector": bpv,
                "rel_error_mean": rel_err,
                "nq": nq,
                "nb_sample": nb_sample,
                "adc_cpu_time_mean": float(run_times.mean()),
                "adc_cpu_time_std": float(run_times.std()) if len(run_times) > 1 else 0.0,
                "adc_cpu_time_pp_mean": float(pp_times.mean()),
                "adc_cpu_time_pp_std": float(pp_times.std()) if len(pp_times) > 1 else 0.0,
                "n_runs": n_runs,
            })

    if not all_rows:
        print("No results collected.")
        return

    new_df = pd.DataFrame(all_rows)

    # Per-experiment output: append/merge with existing (unless using suffix for parallel runs)
    base_name = "all_adc_timing"
    if args.output_suffix:
        base_name = f"{base_name}_{args.output_suffix}"
    out_path = output_dir / f"{base_name}.csv"
    use_append = not args.output_suffix and out_path.exists()
    if use_append:
        existing = pd.read_csv(out_path)
        new_keys = set(zip(new_df["dataset"], new_df["n_subquantizers"], new_df["nbits"]))
        # Keep existing rows that are not being replaced
        mask = [k not in new_keys for k in zip(existing["dataset"], existing["n_subquantizers"], existing["nbits"])]
        kept = existing[mask]
        out_df = pd.concat([kept, new_df], ignore_index=True)
        print(f"\n✅ Per-experiment: {out_path} (appended {len(new_df)} rows, kept {len(kept)} previous, total {len(out_df)})")
    else:
        out_df = new_df
        print(f"\n✅ Per-experiment: {out_path} ({len(out_df)} rows)")
    out_df.to_csv(out_path, index=False)

    if args.output_suffix:
        return  # Aggregated files built by --merge

    # Aggregated by nbits: one row per (dataset, nbits), mean ± std over all M with that nbits
    by_nbits = (
        out_df.groupby(["dataset", "nbits"])
        .agg(
            adc_cpu_time_pp_mean=("adc_cpu_time_pp_mean", "mean"),
            adc_cpu_time_pp_std=("adc_cpu_time_pp_mean", "std"),
            n_experiments=("n_subquantizers", "count"),
        )
        .reset_index()
    )
    by_nbits["adc_cpu_time_pp_std"] = by_nbits["adc_cpu_time_pp_std"].fillna(0)
    by_nbits_path = output_dir / "all_adc_timing_by_nbits.csv"
    by_nbits.to_csv(by_nbits_path, index=False)
    print(f"✅ By nbits:       {by_nbits_path} ({len(by_nbits)} rows)")

    # Aggregated by n_subquantizers: one row per (dataset, M), mean ± std over all nbits with that M
    by_nsubq = (
        out_df.groupby(["dataset", "n_subquantizers"])
        .agg(
            adc_cpu_time_pp_mean=("adc_cpu_time_pp_mean", "mean"),
            adc_cpu_time_pp_std=("adc_cpu_time_pp_mean", "std"),
            n_experiments=("nbits", "count"),
        )
        .reset_index()
    )
    by_nsubq["adc_cpu_time_pp_std"] = by_nsubq["adc_cpu_time_pp_std"].fillna(0)
    by_nsubq_path = output_dir / "all_adc_timing_by_nsubq.csv"
    by_nsubq.to_csv(by_nsubq_path, index=False)
    print(f"✅ By n_subq:     {by_nsubq_path} ({len(by_nsubq)} rows)")


if __name__ == "__main__":
    main()
