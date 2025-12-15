#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Product Quantization (PQ) on a dataset and compare ADC vs exact distances.
Works for .fvecs or .bin datasets (float32), dataset-agnostic.
"""

import numpy as np
import faiss
import time, csv, os, sys
from pathlib import Path
from scipy.spatial.distance import cdist
import argparse
from datetime import datetime

# =====================
# === Import helpers ===
# =====================
module_path = '/home/cpanourg/projects/2-hdvc/'
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import (
    read_fvecs, compute_distance_tables_threaded, compute_distance_tables_vectorized,
    adc_distances_batch_numba, read_fbin, load_dataset, ensure_dir
)


# =============================
# === Main experiment logic ===
# =============================
def run_pq_eval(
    dataset_path,
    query_path=None,
    train_path=None,
    dim=None,
    dataset_name='custom',
    data_root='/data/cpanourg/2-hdvc/',
    n_subquantizers=32,
    nbits=8,
    train_size=100_000,
    sample_db=10_000,
    sample_queries=1_000,
    results_dir='results/relerr',
    load_model=False,
    model_path=None,
):
    # --- Load datasets ---
    # Load three separate files: database (db), training data (train_db), and queries (qr)
    # We train on train_db, test on db, and query with qr
    # All three come from different files
    
    # Load database/test data (db) from dataset_path
    print(f"📂 Loading database from {dataset_path}")
    db, _ = load_dataset(dataset_path, None, dim, db_chunk_size=1_000_000, qr_chunk_size=None)
    print(f"Loaded database shape: {db.shape}")
    
    # Load training data (train_db) from train_path - separate file
    if not train_path:
        raise ValueError("train_path is required. train_db, db, and qr must come from separate files.")
    print(f"📂 Loading training data from {train_path}")
    train_db_full, _ = load_dataset(train_path, None, dim, db_chunk_size=train_size + 10_000, qr_chunk_size=None)
    print(f"Loaded training data shape: {train_db_full.shape}")
    
    # Load queries (qr) from query_path - separate file
    if not query_path:
        raise ValueError("query_path is required. train_db, db, and qr must come from separate files.")
    print(f"📂 Loading queries from {query_path}")
    qr, _ = load_dataset(query_path, None, dim, db_chunk_size=None, qr_chunk_size=sample_queries)
    print(f"Loaded queries shape: {qr.shape}")


    # --- Prepare train/test sets ---
    # We have three separate files: db (database/test), train_db (training), qr (queries)
    # Always train on train_db, test on db, query with qr
    
    # Use separate training data file (train_db) for training
    # Take first train_size samples
    if len(train_db_full) > train_size:
        train_db = train_db_full[:train_size]
    else:
        train_db = train_db_full
        print(f"⚠️  Training data has only {len(train_db)} samples, using all of them")
        train_size = len(train_db)
    
    # Use database (db) as test set - take first test_size samples
    test_size = min(1_000_000, len(db))
    test_db = db[:test_size]
    
    print(f"✅ Using three separate files: train_db from {train_path}, test_db from {dataset_path}, queries from {query_path}")
        # Data is already clean, no cleaning needed
    print(f"Final database shape: {db.shape}")
    print(f"Final training data shape: {train_db.shape}")
    print(f"Final queries shape: {qr.shape}")

    nb, nq, dim = test_db.shape[0], qr.shape[0], test_db.shape[1]
    print(f"Training on {len(train_db)} samples from train_db, testing on {nb} samples from db, queries: {nq}, dim={dim}")

    # --- Setup PQ ---
    if load_model and model_path:
        # Load existing model
        print(f"📥 Loading PQ model from {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        pq_index = faiss.read_index(str(model_path))
        pq = pq_index.pq  # Extract the ProductQuantizer
        
        # Verify dimensions match
        if pq.d != dim:
            raise ValueError(f"Model dimension ({pq.d}) doesn't match data dimension ({dim})")
        if pq.M != n_subquantizers:
            raise ValueError(f"Model n_subquantizers ({pq.M}) doesn't match expected ({n_subquantizers})")
        if pq.nbits != nbits:
            raise ValueError(f"Model nbits ({pq.nbits}) doesn't match expected ({nbits})")
        
        train_time = 0.0  # No training time when loading
        print(f"✅ PQ model loaded: {pq.M}x{pq.nbits} => {pq.M * pq.nbits} bits/vector")
        
        # Create output directory for results (but don't save model again)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"subq{n_subquantizers}_nbits{nbits}_train{train_size}_{timestamp}"
        out_dir = Path(data_root) / results_dir / 'pq' / dataset_name / folder_name
        ensure_dir(out_dir)
    else:
        # Train new model
        pq = faiss.ProductQuantizer(dim, n_subquantizers, nbits)
        print(f"Training PQ with {n_subquantizers}x{nbits} => {n_subquantizers * nbits} bits/vector")

        start = time.time()
        pq.train(train_db)
        train_time = time.time() - start
        print(f"✅ PQ trained in {train_time:.2f}s")
        
        # --- Save trained PQ model ---
        # Create output directory early to save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"subq{n_subquantizers}_nbits{nbits}_train{train_size}_{timestamp}"
        out_dir = Path(data_root) / results_dir / 'pq' / dataset_name / folder_name
        ensure_dir(out_dir)
        
        # Save PQ model by creating a ProductQuantization index and saving it
        # This is the standard way to save a ProductQuantizer in FAISS
        # To load it later:
        #   pq_index = faiss.read_index("pq_model.index")
        #   pq = pq_index.pq  # Extract the ProductQuantizer
        #   codes = pq.compute_codes(data)  # Use it to encode new data
        pq_index = faiss.IndexPQ(dim, n_subquantizers, nbits)
        pq_index.pq = pq  # Assign the trained PQ
        pq_model_path = out_dir / "pq_model.index"
        faiss.write_index(pq_index, str(pq_model_path))
        print(f"✅ PQ model saved to {pq_model_path}")

    # --- Encode database ---
    print("Encoding database...")
    start = time.time()
    try:
        codes = pq.compute_codes(test_db)
    except TypeError:
        codes = np.zeros((nb, pq.code_size), dtype='uint8')
        pq.compute_codes(test_db, codes)
    encoding_time = time.time() - start
    print(f"✅ Encoding done in {encoding_time:.2f}s | Codes shape: {codes.shape}")

    # --- Compute distance tables ---
    print("Computing distance tables...")
    ksub = 1 << nbits
    centroids = faiss.vector_to_array(pq.centroids).reshape(n_subquantizers, ksub, dim // n_subquantizers)

    start = time.time()
    if nq > 1000:
        dis_tables = compute_distance_tables_threaded(qr, centroids, n_subquantizers, ksub, dim)
    else:
        dis_tables = compute_distance_tables_vectorized(qr, centroids, n_subquantizers, ksub, dim)
    distance_table_time = time.time() - start
    print(f"✅ Distance tables computed in {distance_table_time:.2f}s")

    # --- Sample subset for ADC vs exact ---
    # Take first sample_queries queries and first sample_db database vectors
    n_sample_q = min(sample_queries, nq)
    n_sample_db = min(sample_db, nb)
    q_idx = np.arange(n_sample_q)
    db_idx = np.arange(n_sample_db)
    qr_sample, db_sample = qr[:n_sample_q], test_db[:n_sample_db]

    start = time.time()
    adc_sample = adc_distances_batch_numba(dis_tables[q_idx], codes[db_idx], n_subquantizers)
    adc_time = time.time() - start

    start = time.time()
    # Compute exact distances with overflow protection
    # Use float64 for computation, then clip before converting to float32
    with np.errstate(over='ignore'):
        exact_sample_64 = cdist(qr_sample, db_sample, metric="sqeuclidean")
    # Clip to safe float32 range before conversion (use conservative threshold)
    max_safe_value = np.finfo(np.float32).max / 10.0
    exact_sample = np.clip(exact_sample_64, 0, max_safe_value).astype(np.float32)
    cdist_time = time.time() - start

    # Handle overflow: clip ADC distances to prevent inf
    adc_sample = np.clip(adc_sample, 0, max_safe_value)

    # Compute relative error with better numerical stability
    # Use a larger epsilon and handle zero/very small exact distances
    epsilon = 1e-6
    denominator = np.maximum(exact_sample, epsilon)
    
    # Compute relative error with overflow protection
    # Use float64 for intermediate computation to avoid overflow
    diff_64 = np.abs(adc_sample.astype(np.float64) - exact_sample.astype(np.float64))
    denominator_64 = denominator.astype(np.float64)
    rel_error_64 = diff_64 / denominator_64
    # Clip to reasonable range and convert back to float32
    rel_error = np.clip(rel_error_64, 0, 1e6).astype(np.float32)
    
    # Filter out invalid values (inf, nan) before computing statistics
    # Use more robust filtering
    valid_mask = np.isfinite(rel_error) & (rel_error >= 0) & (rel_error < 1e6)
    if valid_mask.sum() == 0:
        print("⚠️  Warning: All relative errors are invalid (inf/nan)")
        mean_rel, std_rel = np.nan, np.nan
    else:
        rel_error_clean = rel_error[valid_mask].astype(np.float64)  # Use float64 for stats
        # Double-check for any remaining invalid values
        rel_error_clean = rel_error_clean[np.isfinite(rel_error_clean)]
        if len(rel_error_clean) == 0:
            print("⚠️  Warning: All relative errors became invalid after filtering")
            mean_rel, std_rel = np.nan, np.nan
        else:
            mean_rel = float(rel_error_clean.mean())
            std_rel = float(rel_error_clean.std())
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                print(f"⚠️  Warning: {invalid_count}/{rel_error.size} relative errors were invalid (inf/nan) and excluded")

    print(f"Mean rel. error: {mean_rel:.4f}, std: {std_rel:.4f}")

    # --- Save results ---
    # out_dir was already created when saving the PQ model
    out_bin = out_dir / f"rel_error_subq{n_subquantizers}_nbits{nbits}_db{sample_db//1000}k_qr{sample_queries//1000}k.bin"
    rel_error.astype(np.float32).tofile(out_bin)

    csv_path = Path(data_root) / results_dir / f"{dataset_name}_PQ_adc_vs_exact_eval.csv"
    summary = {
        "method": "PQ",
        "dataset": dataset_name,
        "experiment_folder": str(out_dir),
        "nq": nq,
        "nb": nb,
        "nb_sample": len(db_idx),
        "dim": dim,
        "n_subquantizers": n_subquantizers,
        "nbits": nbits,
        "bits_per_vector": n_subquantizers * nbits,
        "train_size": train_size,
        "train_time_s": float(train_time),
        "encoding_time_s": float(encoding_time),
        "distance_table_time_s": float(distance_table_time),
        "cdist_time_s": float(cdist_time),
        "adc_time_s": float(adc_time),
        "rel_error_mean": float(mean_rel),
        "rel_error_std": float(std_rel),
    }

    # Save summary CSV in the run-specific folder
    run_csv_path = out_dir / "summary.csv"
    with open(run_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # Append to main CSV file
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if need_header:
            writer.writeheader()
        writer.writerow(summary)

    print(f"✅ Results appended to {csv_path}")
    print(f"✅ Run-specific results saved to {out_dir}")
    if not (load_model and model_path):
        print(f"   - PQ model: pq_model.index")
    print(f"   - Binary rel. error: {out_bin.name}")
    print(f"   - Summary CSV: {run_csv_path.name}")


# ==================================
# === Command-line entry point ===
# ==================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PQ ADC vs exact evaluation (dataset-agnostic).")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to database/test dataset (.fvecs or .bin)")
    parser.add_argument("--query_path", type=str, required=True, help="Path to query dataset (.fvecs or .bin). train_db, db, and qr must come from separate files.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training/learning dataset (.fvecs or .bin). train_db, db, and qr must come from separate files.")
    parser.add_argument("--dim", type=int, default=None, help="Dimensionality (required for .bin datasets)")
    parser.add_argument("--dataset_name", type=str, default="custom", help="Dataset name label for results")
    parser.add_argument("--data_root", type=str, default="/data/cpanourg/2-hdvc/", help="Root path for saving results")
    parser.add_argument("--n_subquantizers", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--train_size", type=int, default=100_000)
    parser.add_argument("--sample_db", type=int, default=10_000)
    parser.add_argument("--sample_queries", type=int, default=1_000)
    parser.add_argument("--results_dir", type=str, default="results/relerr", help="Results subdirectory under data_root")
    parser.add_argument("--load_model", action="store_true", help="Load a pre-trained model instead of training")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved PQ model file (required if --load_model is set)")

    args = parser.parse_args()
    
    # Validate that model_path is provided if load_model is True
    if args.load_model and not args.model_path:
        parser.error("--model_path is required when --load_model is set")
    
    run_pq_eval(**vars(args))
