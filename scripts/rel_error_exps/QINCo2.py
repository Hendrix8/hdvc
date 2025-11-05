#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate QINCo2 on a dataset and compare ADC vs exact distances.
Works for .fvecs or .bin datasets (float32), dataset-agnostic.
Note: This is a simplified version - full implementation requires QINCo2 model setup.
"""

import numpy as np
import sys
import time, csv, os
from pathlib import Path
from scipy.spatial.distance import cdist
import argparse

# =====================
# === Import helpers ===
# =====================
module_path = '/home/cpanourg/projects/2-hdvc/'
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import read_fvecs, write_fvecs


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
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# =============================
# === Main experiment logic ===
# =============================
def run_qinco2_eval(
    dataset_path,
    query_path=None,
    dim=None,
    dataset_name='custom',
    data_root='/data/cpanourg/2-hdvc/',
    model_path=None,
    config_path='/home/cpanourg/projects/2-hdvc/lib/Qinco/config/qinco_cfg.yaml',
    M=8,
    K=256,
    train_size=100_000,
    sample_db=10_000,
    sample_queries=1_000,
    results_dir='results/relerr',
):
    """
    Run QINCo2 evaluation.
    Note: This is a placeholder implementation. Full QINCo2 requires:
    - Model loading from checkpoint
    - Proper encoding with QINCo2 model
    - ADC distance computation using codebooks
    """
    print("⚠️  QINCo2 evaluation requires full model setup")
    print("⚠️  This is a placeholder - please implement full QINCo2 pipeline")
    print("⚠️  See notebook 5-QINCo2.ipynb for reference implementation")
    
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

    # --- Sample subset for distance computation ---
    db_idx = np.random.choice(nb, min(sample_db, nb), replace=False)
    q_idx = np.random.choice(nq, min(sample_queries, nq), replace=False)
    test_db_sample = test_db[db_idx]
    qr_sample = qr[q_idx]

    # TODO: Implement QINCo2 encoding and distance computation
    # This requires:
    # 1. Loading the QINCo2 model
    # 2. Encoding database and queries
    # 3. Building lookup tables from codebooks
    # 4. Computing ADC distances
    
    print("⚠️  Using exact distances as placeholder")
    start = time.time()
    exact_sample = cdist(qr_sample, test_db_sample, metric="sqeuclidean").astype(np.float32)
    cdist_time = time.time() - start

    # Placeholder values
    adc_sample = exact_sample.copy()
    train_time = 0.0
    encoding_time = 0.0
    adc_time = 0.0

    rel_error = np.abs(adc_sample - exact_sample) / (exact_sample + 1e-12)
    mean_rel, std_rel = rel_error.mean(), rel_error.std()

    print(f"Mean rel. error: {mean_rel:.4f}, std: {std_rel:.4f}")

    # --- Save results ---
    out_dir = Path(data_root) / results_dir / 'qinco2' / dataset_name
    ensure_dir(out_dir)
    out_bin = out_dir / f"rel_error_M{M}_K{K}_db{sample_db//1000}k_qr{sample_queries//1000}k.bin"
    rel_error.astype(np.float32).tofile(out_bin)

    csv_path = Path(data_root) / results_dir / f"{dataset_name}_adc_vs_exact_eval.csv"
    summary = {
        "method": "QINCo2",
        "dataset": dataset_name,
        "nq": nq,
        "nb": nb,
        "nb_sample": len(db_idx),
        "dim": dim,
        "n_subquantizers": M,
        "nbits": int(np.log2(K)),
        "bits_per_vector": M * int(np.log2(K)),
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
    print("\n⚠️  NOTE: This is a placeholder implementation. Full QINCo2 requires:")
    print("   - Model loading from checkpoint")
    print("   - Proper encoding pipeline")
    print("   - ADC distance computation with codebooks")
    print("   - See notebook 5-QINCo2.ipynb for reference")


# ==================================
# === Command-line entry point ===
# ==================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QINCo2 ADC vs exact evaluation (dataset-agnostic).")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (.fvecs or .bin)")
    parser.add_argument("--query_path", type=str, default=None, help="Optional query dataset path (.fvecs or .bin)")
    parser.add_argument("--dim", type=int, default=None, help="Dimensionality (required for .bin datasets)")
    parser.add_argument("--dataset_name", type=str, default="custom", help="Dataset name label for results")
    parser.add_argument("--data_root", type=str, default="/data/cpanourg/2-hdvc/", help="Root path for saving results")
    parser.add_argument("--model_path", type=str, default=None, help="Path to QINCo2 model checkpoint")
    parser.add_argument("--config_path", type=str, default="/home/cpanourg/projects/2-hdvc/lib/Qinco/config/qinco_cfg.yaml", help="Path to QINCo2 config")
    parser.add_argument("--M", type=int, default=8, help="Number of codebooks")
    parser.add_argument("--K", type=int, default=256, help="Codebook size")
    parser.add_argument("--train_size", type=int, default=100_000)
    parser.add_argument("--sample_db", type=int, default=10_000)
    parser.add_argument("--sample_queries", type=int, default=1_000)
    parser.add_argument("--results_dir", type=str, default="results/relerr", help="Results subdirectory under data_root")

    args = parser.parse_args()
    # Map M and K to method parameters
    run_qinco2_eval(
        dataset_path=args.dataset_path,
        query_path=args.query_path,
        dim=args.dim,
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        model_path=args.model_path,
        config_path=args.config_path,
        M=args.M,
        K=args.K,
        train_size=args.train_size,
        sample_db=args.sample_db,
        sample_queries=args.sample_queries,
        results_dir=args.results_dir,
    )

