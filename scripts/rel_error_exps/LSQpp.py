#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Local Search Quantizer++ (LSQ++) on a dataset and compare ADC vs exact distances.
Works for .fvecs or .bin datasets (float32), dataset-agnostic.
"""

import numpy as np
import faiss
import time, csv, os, sys
from pathlib import Path
from scipy.spatial.distance import cdist
import argparse

# =====================
# === Import helpers ===
# =====================
module_path = '/home/cpanourg/projects/2-hdvc/'
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import (
    read_fvecs, lsq_alpha_from_codes, lsq_dot_tables_vectorized, lsq_distances_batch_numba, 
    read_fbin, load_dataset, ensure_dir
)


# =====================================================
# === Helper functions for data loading and IO setup ===
# =====================================================
# load_dataset and ensure_dir are now imported from src.utils


# =============================
# === Main experiment logic ===
# =============================
def run_lsqpp_eval(
    dataset_path,
    query_path=None,
    dim=None,
    dataset_name='custom',
    data_root='/data/cpanourg/2-hdvc/',
    n_subquantizers=32,
    nbits=8,
    train_size=100_000,
    sample_db=10_000,
    sample_queries=1_000,
    results_dir='results/relerr',
):
    # --- Load dataset ---
    print(f"📂 Loading dataset from {dataset_path}")
    db, qr = load_dataset(dataset_path, query_path, dim)
    print(f"Loaded database shape: {db.shape}, queries: {qr.shape}")

    # --- Validate and clean data (remove NaN/Inf by loading more clean data) ---
    def clean_data_by_reloading(data, filepath, initial_size):
        """Remove rows with NaN/Inf by loading additional clean data from file. Optimized for large datasets."""
        # Fast check: use vectorized operations on flattened view for better cache performance
        # Check for NaN/Inf more efficiently
        data_flat = data.view(np.float32).reshape(-1)
        has_nan = np.isnan(data_flat).any()
        has_inf = np.isinf(data_flat).any()
        
        if not has_nan and not has_inf:
            return data
        
        # Find invalid rows - optimized for large arrays
        # Use sum along axis which is faster than any() for large arrays
        nan_rows = np.isnan(data).sum(axis=1) > 0
        inf_rows = np.isinf(data).sum(axis=1) > 0
        has_invalid = nan_rows | inf_rows
        invalid_count = has_invalid.sum()
        
        if invalid_count == 0:
            return data
        
        print(f"⚠️  Warning: Found {invalid_count} rows with NaN/Inf ({invalid_count/len(data)*100:.2f}%). Loading clean replacements...")
        
        # Keep only valid rows - use boolean indexing (faster than copy for large arrays)
        valid_mask = ~has_invalid
        valid_data = data[valid_mask]
        remaining_invalid = invalid_count
        
        # If file is .fbin, load additional clean data
        if filepath.endswith('.fbin'):
            # Start loading from beyond what we initially loaded
            replacement_start = initial_size
            # Load larger chunks for better performance (min 100k, or 5x what we need)
            chunk_size = max(remaining_invalid * 5, 100000)
            max_attempts = 50  # More attempts for large datasets
            attempt = 0
            
            # Pre-allocate list to collect clean replacements (faster than repeated vstack)
            clean_replacement_list = []
            total_loaded = 0
            
            while remaining_invalid > 0 and attempt < max_attempts:
                try:
                    # Load replacement data
                    replacement = read_fbin(filepath, start_idx=replacement_start, chunk_size=chunk_size)
                    
                    if len(replacement) == 0:
                        # Reached end of file
                        break
                    
                    # Fast check for clean replacements
                    replacement_nan = np.isnan(replacement).sum(axis=1) == 0
                    replacement_inf = np.isinf(replacement).sum(axis=1) == 0
                    replacement_valid = replacement_nan & replacement_inf
                    clean_replacements = replacement[replacement_valid]
                    
                    if len(clean_replacements) > 0:
                        # Collect clean replacements
                        n_needed = remaining_invalid
                        n_available = len(clean_replacements)
                        n_to_add = min(n_needed, n_available)
                        
                        clean_replacement_list.append(clean_replacements[:n_to_add])
                        total_loaded += n_to_add
                        remaining_invalid -= n_to_add
                        
                        if remaining_invalid == 0:
                            print(f"✅ Replaced all {invalid_count} invalid rows with clean data")
                            break
                        
                        # Move forward for next attempt
                        replacement_start += len(replacement)
                    else:
                        # All replacements were invalid, try further with larger step
                        replacement_start += chunk_size
                    
                    attempt += 1
                except (ValueError, IndexError, IOError) as e:
                    # Reached end of file or other error
                    print(f"⚠️  Reached end of file or error loading more data (attempt {attempt+1}/{max_attempts})")
                    break
            
            # Concatenate all clean replacements at once (much faster than repeated vstack)
            if clean_replacement_list:
                all_replacements = np.vstack(clean_replacement_list)
                valid_data = np.vstack([valid_data, all_replacements])
            
            if remaining_invalid > 0:
                print(f"⚠️  Warning: Could not replace {remaining_invalid} invalid rows. Removing them...")
                print(f"   Final data size: {len(valid_data)} (removed {remaining_invalid} invalid rows)")
        else:
            # For non-.fbin files, just remove invalid rows
            print(f"⚠️  Removing {invalid_count} invalid rows (file format doesn't support chunked reloading)")
        
        return valid_data
    
    # Clean database (estimate initial size from loaded data)
    initial_db_size = len(db)
    db = clean_data_by_reloading(db, dataset_path, initial_size=initial_db_size)
    
    # Clean queries
    if query_path:
        initial_qr_size = len(qr)
        qr = clean_data_by_reloading(qr, query_path, initial_size=initial_qr_size)
    else:
        qr = clean_data_by_reloading(qr, dataset_path, initial_size=initial_db_size)
    
    print(f"Final database shape: {db.shape}, queries: {qr.shape}")

    # --- Split into train/test ---
    test_size = min(1_000_000, len(db))
    idxs = np.random.choice(db.shape[0], train_size + test_size, replace=False)
    train_idxs, test_idxs = idxs[:train_size], idxs[train_size:]

    train_db, test_db = db[train_idxs], db[test_idxs]
    nb, nq, dim = test_db.shape[0], qr.shape[0], test_db.shape[1]
    print(f"Training on {train_size} samples, testing on {nb}, dim={dim}")

    # --- Setup LSQ++ ---
    lsq = faiss.LocalSearchQuantizer(
        dim, n_subquantizers, nbits,
        faiss.LocalSearchQuantizer.ST_norm_qint8,  # LSQ++
    )
    
    print(f"Training LSQ++ with {n_subquantizers}x{nbits} => {n_subquantizers * nbits} bits/vector")

    start = time.time()
    lsq.train(train_db)
    train_time = time.time() - start
    print(f"✅ LSQ++ trained in {train_time:.2f}s")

    # --- Encode database ---
    print("Encoding database...")
    start = time.time()
    codes = lsq.compute_codes(test_db)
    encoding_time = time.time() - start
    print(f"✅ Encoding done in {encoding_time:.2f}s | Codes shape: {codes.shape}")

    # --- Compute distance tables ---
    print("Computing distance tables...")
    ksub = 1 << nbits
    table = faiss.vector_to_array(lsq.codebooks).reshape(-1, dim)
    codebooks = table.reshape(n_subquantizers, ksub, dim)

    start = time.time()
    # Sample subset for distance computation
    q_idx = np.random.choice(nq, min(sample_queries, nq), replace=False)
    db_idx = np.random.choice(nb, min(sample_db, nb), replace=False)
    qr_sample, db_sample = qr[q_idx], test_db[db_idx]
    
    # Precompute alpha for the DB
    alpha = lsq_alpha_from_codes(codebooks, codes[db_idx, :n_subquantizers])
    
    # Build per-query DOT tables
    dot_tables = lsq_dot_tables_vectorized(qr_sample, codebooks)
    
    # Query norms
    qnorms = np.einsum('ij,ij->i', qr_sample, qr_sample).astype(np.float32)
    
    # Distances
    adc_sample = lsq_distances_batch_numba(dot_tables, codes[db_idx], alpha, qnorms)
    distance_table_time = time.time() - start
    print(f"✅ Distance computation done in {distance_table_time:.2f}s")

    start = time.time()
    exact_sample = cdist(qr_sample, db_sample, metric="sqeuclidean").astype(np.float32)
    cdist_time = time.time() - start

    adc_time = distance_table_time  # ADC time is included in distance computation

    rel_error = np.abs(adc_sample - exact_sample) / (exact_sample + 1e-12)
    mean_rel, std_rel = rel_error.mean(), rel_error.std()

    print(f"Mean rel. error: {mean_rel:.4f}, std: {std_rel:.4f}")

    # --- Save results ---
    out_dir = Path(data_root) / results_dir / 'lsq_pp' / dataset_name
    ensure_dir(out_dir)
    out_bin = out_dir / f"rel_error_subq{n_subquantizers}_nbits{nbits}_db{sample_db//1000}k_qr{sample_queries//1000}k.bin"
    rel_error.astype(np.float32).tofile(out_bin)

    csv_path = Path(data_root) / results_dir / f"{dataset_name}_LSQpp_adc_vs_exact_eval.csv"
    summary = {
        "method": "LSQpp",
        "dataset": dataset_name,
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
        "distance_table_time_s": 0.0,
        "cdist_time_s": float(cdist_time),
        "adc_time_s": float(adc_time),
        "rel_error_mean": float(mean_rel),
        "rel_error_std": float(std_rel),
    }

    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if need_header:
            writer.writeheader()
        writer.writerow(summary)

    print(f"✅ Results appended to {csv_path}")
    print(f"Binary rel. error saved to {out_bin}")


# ==================================
# === Command-line entry point ===
# ==================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSQ++ ADC vs exact evaluation (dataset-agnostic).")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (.fvecs or .bin)")
    parser.add_argument("--query_path", type=str, default=None, help="Optional query dataset path (.fvecs or .bin)")
    parser.add_argument("--dim", type=int, default=None, help="Dimensionality (required for .bin datasets)")
    parser.add_argument("--dataset_name", type=str, default="custom", help="Dataset name label for results")
    parser.add_argument("--data_root", type=str, default="/data/cpanourg/2-hdvc/", help="Root path for saving results")
    parser.add_argument("--n_subquantizers", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--train_size", type=int, default=100_000)
    parser.add_argument("--sample_db", type=int, default=10_000)
    parser.add_argument("--sample_queries", type=int, default=1_000)
    parser.add_argument("--results_dir", type=str, default="results/relerr", help="Results subdirectory under data_root")

    args = parser.parse_args()
    run_lsqpp_eval(**vars(args))

