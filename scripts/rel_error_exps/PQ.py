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

# =====================
# === Import helpers ===
# =====================
module_path = '/home/cpanourg/projects/2-hdvc/'
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import (
    read_fvecs, compute_distance_tables_threaded, compute_distance_tables_vectorized,
    adc_distances_batch_numba
)


# =====================================================
# === Helper functions for data loading and IO setup ===
# =====================================================
def load_dataset(dataset_path, query_path=None, dim=None):
    """
    Loads dataset (supports .fvecs and .bin)
    Returns database vectors (db) and query vectors (qr)
    """
    if dataset_path.endswith('.fvecs'):
        db = np.array(read_fvecs(dataset_path))
    elif dataset_path.endswith('.bin'):
        db = np.fromfile(dataset_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    if query_path:
        if query_path.endswith('.fvecs'):
            qr = np.array(read_fvecs(query_path))
        elif query_path.endswith('.bin'):
            qr = np.fromfile(query_path, dtype=np.float32).reshape(-1, dim)
        else:
            raise ValueError(f"Unsupported query format: {query_path}")
    else:
        qr = db.copy()

    return db.astype(np.float32), qr.astype(np.float32)


def ensure_dir(path):
    """Create directory if it doesn’t exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# =============================
# === Main experiment logic ===
# =============================
def run_pq_eval(
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

    # --- Split into train/test ---
    test_size = min(1_000_000, len(db))
    idxs = np.random.choice(db.shape[0], train_size + test_size, replace=False)
    train_idxs, test_idxs = idxs[:train_size], idxs[train_size:]

    train_db, test_db = db[train_idxs], db[test_idxs]
    nb, nq, dim = test_db.shape[0], qr.shape[0], test_db.shape[1]
    print(f"Training on {train_size} samples, testing on {nb}, dim={dim}")

    # --- Setup PQ ---
    pq = faiss.ProductQuantizer(dim, n_subquantizers, nbits)
    print(f"Training PQ with {n_subquantizers}x{nbits} => {n_subquantizers * nbits} bits/vector")

    start = time.time()
    pq.train(train_db)
    train_time = time.time() - start
    print(f"✅ PQ trained in {train_time:.2f}s")

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
    q_idx = np.random.choice(nq, min(sample_queries, nq), replace=False)
    db_idx = np.random.choice(nb, min(sample_db, nb), replace=False)
    qr_sample, db_sample = qr[q_idx], test_db[db_idx]

    start = time.time()
    adc_sample = adc_distances_batch_numba(dis_tables[q_idx], codes[db_idx], n_subquantizers)
    adc_time = time.time() - start

    start = time.time()
    exact_sample = cdist(qr_sample, db_sample, metric="sqeuclidean").astype(np.float32)
    cdist_time = time.time() - start

    rel_error = np.abs(adc_sample - exact_sample) / (exact_sample + 1e-12)
    mean_rel, std_rel = rel_error.mean(), rel_error.std()

    print(f"Mean rel. error: {mean_rel:.4f}, std: {std_rel:.4f}")

    # --- Save results ---
    out_dir = Path(data_root) / results_dir / 'pq' / dataset_name
    ensure_dir(out_dir)
    out_bin = out_dir / f"rel_error_subq{n_subquantizers}_nbits{nbits}_db{sample_db//1000}k_qr{sample_queries//1000}k.bin"
    rel_error.astype(np.float32).tofile(out_bin)

    csv_path = Path(data_root) / results_dir / f"{dataset_name}_PQ_adc_vs_exact_eval.csv"
    summary = {
        "method": "PQ",
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
        "distance_table_time_s": float(distance_table_time),
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
    parser = argparse.ArgumentParser(description="Run PQ ADC vs exact evaluation (dataset-agnostic).")

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
    run_pq_eval(**vars(args))
