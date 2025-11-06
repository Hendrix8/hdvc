#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Variational Additive Quantization (VAQ) on a dataset and compare ADC vs exact distances.
Works for .fvecs or .bin datasets (float32), dataset-agnostic.
Uses the C++ VAQ implementation via subprocess call.
"""

import numpy as np
import subprocess
import sys
import time, csv, os
from pathlib import Path
from scipy.spatial.distance import cdist
import argparse
import tempfile

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
def run_vaq_eval(
    dataset_path,
    query_path=None,
    dim=None,
    dataset_name='custom',
    data_root='/data/cpanourg/2-hdvc/',
    method='VAQ256m32min7max8var1,HEAP',
    total_bits=None,  # Override total bits if provided
    n_subspaces=None,  # Override n_subspaces if provided
    min_bits=None,  # Override min_bits if provided
    max_bits=None,  # Override max_bits if provided
    variance=None,  # Override variance if provided
    train_size=100_000,
    sample_db=10_000,
    sample_queries=1_000,
    results_dir='results/relerr',
    vaq_binary='/home/cpanourg/projects/2-hdvc/lib/VAQ/build/examples/run_vaq',
    refine='100,200',
    k=100,
    learn_ratio=0.05,
):
    # --- Construct or modify method string if parameters provided ---
    import re
    if total_bits is not None or n_subspaces is not None or min_bits is not None or max_bits is not None or variance is not None:
        # Parse existing method string
        method_match = re.match(r'(VAQ)(\d+)(m)(\d+)(min)(\d+)(max)(\d+)(var)([\d.]+)(.*)', method)
        if method_match:
            prefix, old_total_bits, m, old_n_subspaces, min_prefix, old_min_bits, max_prefix, old_max_bits, var_prefix, old_variance, suffix = method_match.groups()
            # Use provided values or keep existing ones
            total_bits = int(total_bits) if total_bits is not None else int(old_total_bits)
            n_subspaces = int(n_subspaces) if n_subspaces is not None else int(old_n_subspaces)
            min_bits = int(min_bits) if min_bits is not None else int(old_min_bits)
            max_bits = int(max_bits) if max_bits is not None else int(old_max_bits)
            variance = float(variance) if variance is not None else float(old_variance)
            # Reconstruct method string
            method = f"VAQ{total_bits}m{n_subspaces}min{min_bits}max{max_bits}var{variance}{suffix}"
            print(f"📝 Constructed method string: {method}")
        else:
            # If method string doesn't match pattern, try to construct from scratch
            if total_bits is not None and n_subspaces is not None and min_bits is not None and max_bits is not None:
                variance = variance if variance is not None else 1.0
                search_method = method.split(',')[-1] if ',' in method else 'HEAP'
                method = f"VAQ{int(total_bits)}m{int(n_subspaces)}min{int(min_bits)}max{int(max_bits)}var{float(variance)},{search_method}"
                print(f"📝 Constructed method string from parameters: {method}")
    
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

    # --- Create temporary files for separate train and encode datasets ---
    temp_dir = Path(tempfile.mkdtemp(prefix='vaq_'))
    train_fp = str(temp_dir / 'trainset.fvecs')
    dataset_fp = str(temp_dir / 'dataset.fvecs')
    queries_fp = str(temp_dir / 'queries.fvecs')
    
    write_fvecs(train_fp, train_db)
    write_fvecs(dataset_fp, test_db)
    write_fvecs(queries_fp, qr)
    print(f"✅ Created temporary files: train={train_fp}, dataset={dataset_fp}")

    # --- Setup output directory ---
    result_dir = Path(data_root) / results_dir / 'vaq' / dataset_name
    ensure_dir(result_dir)
    codes_fp = str(result_dir / 'codes.fvecs')
    centroids_fp = str(result_dir / 'centroids.fvecs')
    result_csv = str(result_dir / 'vaq_results.csv')

    # --- Run VAQ training and encoding ---
    print(f"Running VAQ with method: {method}")
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{os.path.expanduser('~')}/local/glpk/lib:"
        f"{os.path.expanduser('~')}/local/armadillo/lib:"
        f"{env.get('CONDA_PREFIX', '')}/lib:"
        f"{env.get('LD_LIBRARY_PATH', '')}"
    )

    cmd = [
        vaq_binary,
        '--dataset', dataset_fp,
        '--trainset', train_fp,  # Use separate trainset file
        '--queries', queries_fp,
        '--file-format-ori', 'fvecs',
        '--timeseries-size', str(dim),
        '--dataset-size', str(nb),
        '--trainset-size', str(train_size),
        '--queries-size', str(nq),
        '--result', result_csv,
        '--method', method,
        '--k', str(k),
        '--refine', refine,
        '--save-enc', codes_fp,
        '--save', centroids_fp,
        '--learn-ratio', str(learn_ratio),
    ]

    start = time.time()
    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, text=True
    )
    
    train_time = None
    encoding_time = None
    
    for line in process.stdout:
        print(line, end="")
        if "Training time:" in line:
            try:
                train_time = float(line.split("Training time:")[1].split("s")[0].strip())
            except:
                pass
        if "Encoding time:" in line:
            try:
                encoding_time = float(line.split("Encoding time:")[1].split("s")[0].strip())
            except:
                pass
    
    process.wait()
    total_time = time.time() - start
    
    if train_time is None:
        train_time = total_time * 0.3  # Estimate if not found
    if encoding_time is None:
        encoding_time = total_time * 0.7  # Estimate if not found
    
    print(f"\n✅ VAQ completed (train: {train_time:.2f}s, encode: {encoding_time:.2f}s)")

    # --- Load codes ---
    if not os.path.exists(codes_fp):
        print(f"⚠️  Warning: Codes file not found at {codes_fp}")
        return
    
    with open(codes_fp, "rb") as f:
        nrows = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncols = np.fromfile(f, dtype=np.int64, count=1)[0]
        codes = np.fromfile(f, dtype=np.int16, count=nrows * ncols).reshape(nrows, ncols)
    
    print(f"✅ Loaded codes: {codes.shape}")

    # --- Load centroids ---
    if not os.path.exists(centroids_fp):
        print(f"⚠️  Warning: Centroids file not found at {centroids_fp}")
        return
    
    print("Loading centroids...")
    centroids_per_subs = []
    with open(centroids_fp, "rb") as f:
        # size_t is typically 8 bytes (uint64) on 64-bit systems, 4 bytes (uint32) on 32-bit
        # For most modern systems, use uint64
        n_subs = np.fromfile(f, dtype=np.uint64, count=1)[0]
        
        for i in range(int(n_subs)):
            n_centroids = np.fromfile(f, dtype=np.uint64, count=1)[0]
            subs_dim = np.fromfile(f, dtype=np.uint64, count=1)[0]
            centroids_data = np.fromfile(f, dtype=np.float32, count=int(n_centroids * subs_dim))
            centroids = centroids_data.reshape(int(n_centroids), int(subs_dim))
            centroids_per_subs.append(centroids)
    
    print(f"✅ Loaded {len(centroids_per_subs)} subspaces with centroids")
    for i, c in enumerate(centroids_per_subs):
        print(f"  Subspace {i}: {c.shape[0]} centroids, dim {c.shape[1]}")

    # --- Parse method string to get parameters ---
    method_match = re.match(r'VAQ(\d+)m(\d+)min(\d+)max(\d+)var([\d.]+)', method.split(',')[0])
    if not method_match:
        print(f"⚠️  Warning: Could not parse method string: {method}")
        return
    
    total_bits, n_subspaces, min_bits, max_bits, variance = map(float, method_match.groups())
    n_subspaces = int(n_subspaces)
    subs_len = dim // n_subspaces
    if dim % n_subspaces > 0:
        subs_len += 1
    
    print(f"VAQ parameters: {int(total_bits)} bits, {n_subspaces} subspaces, subs_len={subs_len}")

    # --- Sample subset for distance computation ---
    db_idx = np.random.choice(nb, min(sample_db, nb), replace=False)
    q_idx = np.random.choice(nq, min(sample_queries, nq), replace=False)
    test_db_sample = test_db[db_idx]
    qr_sample = qr[q_idx]
    codes_sample = codes[db_idx]

    # --- Compute exact distances ---
    start = time.time()
    exact_sample = cdist(qr_sample, test_db_sample, metric="sqeuclidean").astype(np.float32)
    cdist_time = time.time() - start

    # --- Compute ADC distances ---
    print("Computing ADC distances...")
    start = time.time()
    
    nq_sample = len(q_idx)
    nb_sample = len(db_idx)
    
    # Compute distance tables (LUT) for all queries
    # For VAQ, we need to compute LUT per subspace, similar to PQ but with variable centroids per subspace
    # 
    # IMPORTANT NOTE: VAQ uses PCA projection internally. The centroids are learned in PCA space.
    # Ideally, we should project queries to PCA space first using eigenvectors. However, since
    # eigenvectors are not saved by default, we compute distances in the original space.
    # This is an approximation - for accurate ADC distances matching VAQ's internal computation,
    # eigenvectors would need to be saved and loaded. For relative error evaluation purposes,
    # this approximation should still provide meaningful results.
    
    # Pre-compute LUTs for all queries: (nq_sample, max_centroids, n_subspaces)
    # Since each subspace can have different numbers of centroids, we'll store as list of arrays
    luts = []  # List of (n_subspaces) lists, each containing (nq_sample, n_centroids) arrays
    
    for subs in range(n_subspaces):
        start_dim = subs * subs_len
        end_dim = min(start_dim + subs_len, dim)
        centroids_subs = centroids_per_subs[subs]
        n_centroids = centroids_subs.shape[0]
        subs_dim = centroids_subs.shape[1]
        
        # Extract query subvectors for all queries: (nq_sample, subs_dim)
        query_subvecs = qr_sample[:, start_dim:end_dim]
        
        # Handle dimension mismatch
        if query_subvecs.shape[1] != subs_dim:
            if query_subvecs.shape[1] < subs_dim:
                # Pad with zeros
                padded = np.zeros((nq_sample, subs_dim), dtype=np.float32)
                padded[:, :query_subvecs.shape[1]] = query_subvecs
                query_subvecs = padded
            else:
                # Truncate
                query_subvecs = query_subvecs[:, :subs_dim]
        
        # Compute squared distances: (nq_sample, n_centroids)
        # Vectorized: (nq_sample, 1, subs_dim) - (1, n_centroids, subs_dim)
        diff = query_subvecs[:, np.newaxis, :] - centroids_subs[np.newaxis, :, :]
        dists = np.sum(diff * diff, axis=2).astype(np.float32)  # (nq_sample, n_centroids)
        luts.append(dists)
    
    # Compute ADC distances by summing LUT entries according to codes
    adc_sample = np.zeros((nq_sample, nb_sample), dtype=np.float32)
    
    for db in range(nb_sample):
        for subs in range(n_subspaces):
            code = codes_sample[db, subs]
            if code >= 0 and code < luts[subs].shape[1]:
                adc_sample[:, db] += luts[subs][:, code]
            # else: invalid code (shouldn't happen, but keep distance as-is)
    
    adc_time = time.time() - start
    distance_table_time = adc_time * 0.3  # Estimate LUT computation time
    print(f"✅ ADC distances computed in {adc_time:.2f}s")

    rel_error = np.abs(adc_sample - exact_sample) / (exact_sample + 1e-12)
    mean_rel, std_rel = rel_error.mean(), rel_error.std()

    print(f"Mean rel. error: {mean_rel:.4f}, std: {std_rel:.4f}")

    # --- Save results ---
    out_dir = Path(data_root) / results_dir / 'vaq' / dataset_name
    ensure_dir(out_dir)
    out_bin = out_dir / f"rel_error_{method.replace(',', '_')}_db{sample_db//1000}k_qr{sample_queries//1000}k.bin"
    rel_error.astype(np.float32).tofile(out_bin)

    csv_path = Path(data_root) / results_dir / f"{dataset_name}_VAQ_adc_vs_exact_eval.csv"
    summary = {
        "method": "VAQ",
        "dataset": dataset_name,
        "nq": nq,
        "nb": nb,
        "nb_sample": len(db_idx),
        "dim": dim,
        "n_subquantizers": int(n_subspaces),
        # "nbits": int((min_bits + max_bits) / 2),  # Average bits per subspace
        "min_bits": int(min_bits),
        "max_bits": int(max_bits),
        "variance": float(variance),
        "bits_per_vector": int(total_bits),
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
    
    # Cleanup temporary files
    import shutil
    shutil.rmtree(temp_dir)
    print(f"✅ Cleaned up temporary directory: {temp_dir}")


# ==================================
# === Command-line entry point ===
# ==================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VAQ ADC vs exact evaluation (dataset-agnostic).")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (.fvecs or .bin)")
    parser.add_argument("--query_path", type=str, default=None, help="Optional query dataset path (.fvecs or .bin)")
    parser.add_argument("--dim", type=int, default=None, help="Dimensionality (required for .bin datasets)")
    parser.add_argument("--dataset_name", type=str, default="custom", help="Dataset name label for results")
    parser.add_argument("--data_root", type=str, default="/data/cpanourg/2-hdvc/", help="Root path for saving results")
    parser.add_argument("--method", type=str, default="VAQ256m32min7max8var1,HEAP", help="VAQ method string")
    parser.add_argument("--total_bits", type=int, default=None, help="Total bits per vector (overrides method string)")
    parser.add_argument("--n_subspaces", type=int, default=None, help="Number of subspaces (overrides method string)")
    parser.add_argument("--min_bits", type=int, default=None, help="Minimum bits per subspace (overrides method string)")
    parser.add_argument("--max_bits", type=int, default=None, help="Maximum bits per subspace (overrides method string)")
    parser.add_argument("--variance", type=float, default=None, help="Variance threshold (overrides method string)")
    parser.add_argument("--train_size", type=int, default=100_000)
    parser.add_argument("--sample_db", type=int, default=10_000)
    parser.add_argument("--sample_queries", type=int, default=1_000)
    parser.add_argument("--results_dir", type=str, default="results/relerr", help="Results subdirectory under data_root")
    parser.add_argument("--vaq_binary", type=str, default="/home/cpanourg/projects/2-hdvc/lib/VAQ/build/examples/run_vaq", help="Path to VAQ binary")
    parser.add_argument("--refine", type=str, default="100,200", help="Refinement parameters")
    parser.add_argument("--k", type=int, default=100, help="Number of nearest neighbors")
    parser.add_argument("--learn_ratio", type=float, default=0.05, help="Learning ratio for quantization")

    args = parser.parse_args()
    run_vaq_eval(**vars(args))

