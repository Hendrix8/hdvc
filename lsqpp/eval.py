#!/usr/bin/env python3
"""LSQ++ (Faiss LocalSearchQuantizer ST_norm_qint8): train, encode, ADC vs exact L2² (PQ-aligned)."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lsqpp.data_io import (
    ensure_dir,
    infer_dim_from_path,
    load_dataset,
    load_vectors_prefix,
)
from lsqpp.lsq_adc import (
    lsq_alpha_from_codes,
    lsq_distances_batch_numba,
    lsq_dot_tables_vectorized,
)
from lsqpp.metrics import (
    compute_rel_error_pq_style,
    exact_distances_sqeuclidean,
    mean_spearman_rank,
    compute_lsq_reconstruction_error,
)
TEST_CAP = 1_000_000


def _unpack_lsq_codes(codes_packed: np.ndarray, M: int, nbits: int) -> np.ndarray:
    """
    Faiss `compute_codes` returns packed bytes, not one uint8 per subquantizer.
    Decode to (n, M) integer indices in [0, 2**nbits) for ADC math.
    """
    if not hasattr(faiss, "unpack_bitstrings"):
        raise RuntimeError(
            "faiss.unpack_bitstrings is missing; use a full faiss-cpu build with extra_wrappers."
        )
    b = np.ascontiguousarray(codes_packed, dtype=np.uint8)
    unpacked = faiss.unpack_bitstrings(b, M, nbits)
    return np.asarray(unpacked, dtype=np.int64)


def run_lsqpp_eval(
    dataset_path: str,
    query_path: str | None = None,
    train_path: str | None = None,
    dim: int | None = None,
    dataset_name: str = "custom",
    data_root: str = "/data/cpanourg/2-hdvc/",
    n_subquantizers: int = 32,
    nbits: int = 8,
    train_size: int = 100_000,
    sample_db: int = 10_000,
    sample_queries: int = 1_000,
    results_dir: str = "results/lsqpp",
    seed: int = 123,
    sample_mode: str = "first",
):
    if sample_mode not in ("first", "random"):
        raise ValueError("sample_mode must be 'first' or 'random'")
    np.random.seed(seed)

    data_root_p = Path(data_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    M = n_subquantizers
    ksub = 1 << nbits

    # --- Load data (PQ-style or legacy) ---
    if train_path:
        if query_path is None:
            raise ValueError("train_path requires query_path")
        if dim is None:
            dim = infer_dim_from_path(train_path)
        tp, dp = Path(train_path).resolve(), Path(dataset_path).resolve()
        same_file = tp == dp
        print(
            f"📂 PQ-style load: train={train_path}"
            + (" (same file as base)" if same_file else "")
        )
        if same_file:
            train_db = load_vectors_prefix(dataset_path, dim, train_size, 0)
            test_db = load_vectors_prefix(dataset_path, dim, TEST_CAP, train_size)
        else:
            train_db = load_vectors_prefix(train_path, dim, train_size, 0)
            test_db = load_vectors_prefix(dataset_path, dim, TEST_CAP, 0)
        if len(train_db) < train_size:
            print(f"⚠️  Training set has only {len(train_db)} vectors; using all")
            train_size = len(train_db)
        qr = load_vectors_prefix(query_path, dim, sample_queries, 0)
    else:
        print(f"📂 Loading dataset from {dataset_path}")
        db, qr = load_dataset(dataset_path, query_path, dim)
        test_size = min(TEST_CAP, len(db))
        n_need = train_size + test_size
        if n_need > len(db):
            train_size = max(1, min(train_size, len(db) - 1))
            test_size = min(test_size, len(db) - train_size)
            print(
                f"⚠️  Shrinking split: train_size={train_size}, test_size={test_size}"
            )
        idxs = np.random.choice(db.shape[0], train_size + test_size, replace=False)
        train_db, test_db = db[idxs[:train_size]], db[idxs[train_size:]]

    nb, nq, dim = test_db.shape[0], qr.shape[0], test_db.shape[1]
    print(
        f"train_db={train_db.shape}, test_db={test_db.shape}, qr={qr.shape}, dim={dim}"
    )

    index_lsq = faiss.IndexLocalSearchQuantizer(
        dim,
        M,
        nbits,
        faiss.LocalSearchQuantizer.ST_norm_qint8,
    )
    print(f"Training LSQ++ {M}x{nbits} => {M * nbits} bits/vector")

    t0 = time.time()
    index_lsq.train(train_db)
    train_time = time.time() - t0
    print(f"✅ LSQ++ trained in {train_time:.2f}s")
    lsq = index_lsq.lsq

    t0 = time.time()
    codes_u8 = np.asarray(index_lsq.sa_encode(test_db), dtype=np.uint8)
    encoding_time = time.time() - t0
    csz = codes_u8.shape[1]
    min_cs = (M * nbits + 7) // 8
    if csz < min_cs:
        raise ValueError(
            f"Faiss code_size {csz} < packed minimum {min_cs} for M={M}, nbits={nbits}"
        )
    codes_ix = _unpack_lsq_codes(codes_u8, M, nbits)
    if codes_ix.shape != (codes_u8.shape[0], M):
        raise ValueError(
            f"unpack shape {codes_ix.shape} != (n, M) with M={M}"
        )
    print(f"✅ Encoding in {encoding_time:.2f}s | packed {codes_u8.shape} -> indices {codes_ix.shape}")

    table = np.ascontiguousarray(
        faiss.vector_to_array(lsq.codebooks).reshape(-1, dim), dtype=np.float32
    )
    codebooks = table.reshape(M, ksub, dim)

    n_sample_q = min(sample_queries, nq)
    n_sample_db = min(sample_db, nb)
    if sample_mode == "first":
        test_db_sample = test_db[:n_sample_db]
        qr_sample = qr[:n_sample_q]
        codes_sub = codes_ix[:n_sample_db]
    else:
        db_ch = np.random.choice(nb, n_sample_db, replace=False)
        q_ch = np.random.choice(nq, n_sample_q, replace=False)
        test_db_sample = test_db[db_ch]
        qr_sample = qr[q_ch]
        codes_sub = codes_ix[db_ch]

    t0 = time.time()
    alpha = lsq_alpha_from_codes(codebooks, codes_sub)
    dot_tables = lsq_dot_tables_vectorized(qr_sample, codebooks)
    distance_table_time = time.time() - t0

    qnorms = np.einsum("ij,ij->i", qr_sample, qr_sample).astype(np.float32)

    t0 = time.time()
    adc_sample = lsq_distances_batch_numba(dot_tables, codes_sub, alpha, qnorms)
    adc_time = time.time() - t0

    t0 = time.time()
    exact_sample = exact_distances_sqeuclidean(qr_sample, test_db_sample)
    cdist_time = time.time() - t0

    rel_error, mean_rel, std_rel = compute_rel_error_pq_style(adc_sample, exact_sample)
    print(f"Mean rel. error: {mean_rel:.4f}, std: {std_rel:.4f}")

    spearman = mean_spearman_rank(adc_sample, exact_sample)
    print(f"Spearman rank: {spearman:.4f}")

    recon_error = compute_lsq_reconstruction_error(test_db_sample, codes_sub, codebooks)
    print(f"Reconstruction error: {recon_error:.4f}")

    per_pair_adc_time_ns = ((distance_table_time + adc_time) / (nq * n_sample_db)) * 1e9

    safe = f"{M}x{nbits}"
    out_dir = data_root_p / results_dir / dataset_name / f"lsqpp_{safe}_{ts}"
    ensure_dir(out_dir)
    lsq_index_path = out_dir / "lsq_model.index"
    faiss.write_index(index_lsq, str(lsq_index_path))
    print(f"✅ LSQ++ index saved to {lsq_index_path}")
    out_bin = out_dir / f"rel_error_lsqpp_{safe}_db{n_sample_db}_qr{n_sample_q}.bin"
    rel_error.astype(np.float32).tofile(out_bin)

    bits_per_vector = M * nbits
    summary = {
        "method": "LSQpp",
        "dataset": dataset_name,
        "experiment_folder": str(out_dir),
        "nq": nq,
        "nb": nb,
        "nb_sample": int(n_sample_db),
        "nq_sample": int(n_sample_q),
        "dim": dim,
        "n_subquantizers": M,
        "nbits": nbits,
        "bits_per_vector": int(bits_per_vector),
        "train_size": train_size,
        "train_time_s": float(train_time),
        "encoding_time_s": float(encoding_time),
        "distance_table_time_s": float(distance_table_time),
        "cdist_time_s": float(cdist_time),
        "adc_time_s": float(adc_time),
        "per_pair_adc_time_ns": float(per_pair_adc_time_ns),
        "rel_error_mean": mean_rel,
        "rel_error_std": std_rel,
        "spearman": float(spearman),
        "reconstruction_error": float(recon_error),
        "sample_mode": sample_mode,
        "train_path": train_path or "",
        "seed": seed,
    }

    run_csv = out_dir / "summary.csv"
    with open(run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    csv_agg = data_root_p / results_dir / f"{dataset_name}_LSQpp_adc_vs_exact_eval.csv"
    need_header = not csv_agg.exists() or csv_agg.stat().st_size == 0
    with open(csv_agg, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if need_header:
            w.writeheader()
        w.writerow(summary)

    print(f"✅ Run summary: {run_csv}")
    print(f"✅ Appended: {csv_agg}")
    print(f"✅ Rel. error bin: {out_bin}")


def main() -> None:
    p = argparse.ArgumentParser(description="LSQ++ ADC vs exact (PQ-aligned).")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--query_path", type=str, default=None)
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--dataset_name", type=str, default="custom")
    p.add_argument("--data_root", type=str, default="/data/cpanourg/2-hdvc/")
    p.add_argument("--n_subquantizers", type=int, default=32)
    p.add_argument("--nbits", type=int, default=8)
    p.add_argument("--train_size", type=int, default=100_000)
    p.add_argument("--sample_db", type=int, default=10_000)
    p.add_argument("--sample_queries", type=int, default=1_000)
    p.add_argument("--results_dir", type=str, default="results/lsqpp")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--sample_mode", type=str, default="first", choices=("first", "random")
    )
    args = p.parse_args()
    run_lsqpp_eval(**vars(args))


if __name__ == "__main__":
    main()
