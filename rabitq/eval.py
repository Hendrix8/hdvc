#!/usr/bin/env python3
"""RaBitQ ADC vs exact relative error evaluation — aligned with PQ/OPQ where applicable."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rabitq.data_io import ensure_dir, infer_dim_from_path, load_dataset, load_vectors_prefix
from rabitq.reference_rabitq import ReferenceRabitQ, random_rotation

TEST_CAP = 1_000_000
# Match scripts/rel_error_exps/PQ.py and cpp pq_eval relative error
REL_EPSILON = 1e-6
MAX_REL_CLIP = 1e6
MAX_SAFE_FLOAT32 = np.finfo(np.float32).max / 10.0


def compute_rel_error_pq_style(
    adc_sample: np.ndarray, exact_sample: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Same relative error definition as PQ.py (epsilon, float64, clip, valid mask)."""
    adc_sample = np.clip(adc_sample.astype(np.float32), 0, MAX_SAFE_FLOAT32)
    exact_64 = exact_sample.astype(np.float64)
    adc_64 = adc_sample.astype(np.float64)
    denominator = np.maximum(exact_64, REL_EPSILON)
    diff_64 = np.abs(adc_64 - exact_64)
    rel_error_64 = diff_64 / denominator
    rel_error = np.clip(rel_error_64, 0, MAX_REL_CLIP).astype(np.float32)

    valid_mask = np.isfinite(rel_error) & (rel_error >= 0) & (rel_error < MAX_REL_CLIP)
    if valid_mask.sum() == 0:
        return rel_error, float("nan"), float("nan")
    rel_clean = rel_error[valid_mask].astype(np.float64)
    rel_clean = rel_clean[np.isfinite(rel_clean)]
    if rel_clean.size == 0:
        return rel_error, float("nan"), float("nan")
    return rel_error, float(rel_clean.mean()), float(rel_clean.std())


def _exact_distances_pq_style(qr_sample: np.ndarray, db_sample: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        exact_64 = cdist(qr_sample, db_sample, metric="sqeuclidean")
    return np.clip(exact_64, 0, MAX_SAFE_FLOAT32).astype(np.float32)


def run_rabitq_eval(
    dataset_path: str,
    query_path: str | None = None,
    train_path: str | None = None,
    dim: int | None = None,
    dataset_name: str = "custom",
    data_root: str = "/data/cpanourg/2-hdvc/",
    bits_per_query_dim: int = 2,
    train_size: int = 100_000,
    sample_db: int = 10_000,
    sample_queries: int = 1_000,
    results_dir: str = "results/rabitq",
    seed: int = 123,
    sample_mode: str = "first",
):
    if bits_per_query_dim < 1:
        raise ValueError("bits_per_query_dim must be >= 1")
    if sample_mode not in ("first", "random"):
        raise ValueError("sample_mode must be 'first' or 'random'")

    data_root_p = Path(data_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Bq = bits_per_query_dim

    # --- Load data: PQ-style (separate train / base / queries) or legacy single file ---
    if train_path:
        if query_path is None:
            raise ValueError("train_path requires query_path")
        if dim is None:
            dim = infer_dim_from_path(train_path)
        tp, dp = Path(train_path).resolve(), Path(dataset_path).resolve()
        same_file = tp == dp
        print(
            f"📂 PQ-style load: train={train_path}"
            + (" (same file as base — non-overlapping slices)" if same_file else "")
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

        print(f"📂 Base / test vectors: {dataset_path}")
        print(f"📂 Queries: {query_path}")
        qr = load_vectors_prefix(query_path, dim, sample_queries, 0)
        print(
            f"Shapes: train_db={train_db.shape}, test_db={test_db.shape}, qr={qr.shape}"
        )
    else:
        print(f"📂 Loading dataset from {dataset_path}")
        db, qr = load_dataset(dataset_path, query_path, dim)

        def clean_data_by_reloading(data, filepath, initial_size):
            data_flat = data.view(np.float32).reshape(-1)
            if not (np.isnan(data_flat).any() or np.isinf(data_flat).any()):
                return data
            nan_rows = np.isnan(data).sum(axis=1) > 0
            inf_rows = np.isinf(data).sum(axis=1) > 0
            has_invalid = nan_rows | inf_rows
            invalid_count = has_invalid.sum()
            if invalid_count == 0:
                return data
            valid_mask = ~has_invalid
            return data[valid_mask]

        initial_db_size = len(db)
        db = clean_data_by_reloading(db, dataset_path, initial_db_size)
        if query_path:
            qr = clean_data_by_reloading(qr, query_path, len(qr))
        else:
            qr = clean_data_by_reloading(qr, dataset_path, initial_db_size)

        test_size = min(TEST_CAP, len(db))
        n_need = train_size + test_size
        if n_need > len(db):
            train_size = max(1, min(train_size, len(db) - 1))
            test_size = min(test_size, len(db) - train_size)
            print(
                f"⚠️  Database smaller than train+test request; using "
                f"train_size={train_size}, test_size={test_size}"
            )
        idxs = np.random.choice(db.shape[0], train_size + test_size, replace=False)
        train_db, test_db = db[idxs[:train_size]], db[idxs[train_size:]]
        print(f"Legacy random split: train_db={train_db.shape}, test_db={test_db.shape}")

    nb, nq, dim = test_db.shape[0], qr.shape[0], test_db.shape[1]
    print(
        f"Training on {train_db.shape[0]} samples, base/test nb={nb}, "
        f"queries nq={nq}, dim={dim}"
    )

    bits_per_vector = Bq * dim
    print(f"Training RaBitQ: {Bq} bits/query dim => {bits_per_vector} bits/vector")

    P = random_rotation(dim, seed=seed)
    rabitq = ReferenceRabitQ(dim, Bq=Bq)

    t0 = time.time()
    rabitq.train(train_db, P)
    train_time = time.time() - t0
    print(f"✅ RaBitQ trained in {train_time:.2f}s")

    print("Encoding full base/test set...")
    t0 = time.time()
    rabitq.add(test_db)
    encoding_time = time.time() - t0
    print(f"✅ Encoding done in {encoding_time:.2f}s")

    # Evaluation subset (PQ: first sample_queries × first sample_db)
    n_sample_q = min(sample_queries, nq)
    n_sample_db = min(sample_db, nb)
    if sample_mode == "first":
        db_idx = np.arange(n_sample_db)
        q_idx = np.arange(n_sample_q)
        test_db_sample = test_db[:n_sample_db]
        qr_sample = qr[:n_sample_q]
    else:
        db_idx = np.random.choice(nb, n_sample_db, replace=False)
        q_idx = np.random.choice(nq, n_sample_q, replace=False)
        test_db_sample = test_db[db_idx]
        qr_sample = qr[q_idx]

    rabitq_sub = ReferenceRabitQ(dim, Bq=Bq)
    rabitq_sub.centroid = rabitq.centroid
    rabitq_sub.P = rabitq.P
    rabitq_sub.add(test_db_sample)

    # No separate distance-table phase (PQ: distance_table_time); set to 0 for schema match
    distance_table_time = 0.0

    print("Computing ADC distances (RaBitQ estimates vs stored base subset)...")
    t0 = time.time()
    adc_sample = rabitq_sub.distances(qr_sample)
    adc_time = time.time() - t0
    if not isinstance(adc_sample, np.ndarray):
        adc_sample = np.array(adc_sample, dtype=np.float32)

    t0 = time.time()
    exact_sample = _exact_distances_pq_style(qr_sample, test_db_sample)
    cdist_time = time.time() - t0

    rel_error, mean_rel, std_rel = compute_rel_error_pq_style(adc_sample, exact_sample)
    print(f"Mean rel. error: {mean_rel:.4f}, std: {std_rel:.4f}")

    folder_name = f"bits{Bq}_train{train_size}_{ts}"
    out_dir = data_root_p / results_dir / dataset_name / folder_name
    ensure_dir(out_dir)
    out_bin = out_dir / (
        f"rel_error_bits{bits_per_vector}_db{n_sample_db}_qr{n_sample_q}.bin"
    )
    rel_error.astype(np.float32).tofile(out_bin)

    csv_aggregate = data_root_p / results_dir / f"{dataset_name}_RaBitQ_adc_vs_exact_eval.csv"
    # Columns mirror PQ/OPQ summary CSVs; RaBitQ uses n_subquantizers=0 and nbits=bits_per_query_dim.
    summary = {
        "method": "RaBitQ",
        "dataset": dataset_name,
        "experiment_folder": str(out_dir),
        "nq": nq,
        "nb": nb,
        "nb_sample": int(n_sample_db),
        "dim": dim,
        "n_subquantizers": 0,
        "nbits": Bq,
        "bits_per_vector": bits_per_vector,
        "train_size": train_size,
        "train_time_s": float(train_time),
        "encoding_time_s": float(encoding_time),
        "distance_table_time_s": float(distance_table_time),
        "cdist_time_s": float(cdist_time),
        "adc_time_s": float(adc_time),
        "rel_error_mean": mean_rel,
        "rel_error_std": std_rel,
        "nq_sample": int(n_sample_q),
        "sample_mode": sample_mode,
        "train_path": train_path or "",
        "seed": seed,
    }

    run_csv = out_dir / "summary.csv"
    with open(run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    need_header = not csv_aggregate.exists() or csv_aggregate.stat().st_size == 0
    with open(csv_aggregate, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if need_header:
            w.writeheader()
        w.writerow(summary)

    print(f"✅ Run summary: {run_csv}")
    print(f"✅ Appended aggregate CSV: {csv_aggregate}")
    print(f"✅ Rel. error matrix saved: {out_bin}")


def main():
    p = argparse.ArgumentParser(
        description="RaBitQ ADC vs exact (PQ-aligned options: --train_path, --sample_mode first)."
    )
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--query_path", type=str, default=None)
    p.add_argument("--train_path", type=str, default=None,
                   help="Optional. If set: train on this file (first train_size vectors), "
                        "evaluate on base dataset_path and queries query_path (PQ-style).")
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--dataset_name", type=str, default="custom")
    p.add_argument("--data_root", type=str, default="/data/cpanourg/2-hdvc/")
    p.add_argument("--bits_per_query_dim", type=int, default=2)
    p.add_argument("--train_size", type=int, default=100_000)
    p.add_argument("--sample_db", type=int, default=10_000)
    p.add_argument("--sample_queries", type=int, default=1_000)
    p.add_argument(
        "--results_dir",
        type=str,
        default="results/rabitq",
        help="Under data_root, e.g. results/rabitq -> .../results/rabitq/",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--sample_mode",
        type=str,
        default="first",
        choices=("first", "random"),
        help="Use first N base vectors and first N queries (PQ default) or random subsets.",
    )
    args = p.parse_args()
    run_rabitq_eval(**vars(args))


if __name__ == "__main__":
    main()
