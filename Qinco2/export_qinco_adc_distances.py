#!/usr/bin/env python3
"""
Export dense QINCo AQ ADC distances and relative-error statistics.

This is intentionally separate from the FAISS timing benchmark. It computes
dense approximate distances with vectorized NumPy so we can save per-pair
distance matrices and accuracy statistics without using those Python timings
as benchmark numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from run_discovered_adc_jobs import _DATASET_VEC, iter_jobs


DENSE_ADC_VERSION = 1


def xvecs_mmap(path: Path, dtype: np.dtype) -> np.ndarray:
    dim = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    itemsize = np.dtype(dtype).itemsize
    record_size = 4 + dim * itemsize
    n_vecs = path.stat().st_size // record_size
    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(n_vecs, record_size))
    return raw[:, 4:].view(dtype).reshape(n_vecs, dim)


def load_vectors(path: Path, count: int) -> np.ndarray:
    if path.suffix == ".bvecs":
        data = xvecs_mmap(path, np.uint8)
    elif path.suffix == ".fvecs":
        data = xvecs_mmap(path, np.float32)
    else:
        raise ValueError(f"Unsupported vector file suffix: {path}")
    return np.asarray(data[:count], dtype=np.float32)


def load_codes(encoded_db: Path, n_db: int, expected_m: int) -> np.ndarray:
    header = np.load(encoded_db)
    n_parts = int(header["n_parts"])
    part_base = str(encoded_db)[:-4]
    parts = []
    remaining = n_db
    for part_idx in range(n_parts):
        if remaining <= 0:
            break
        part = np.load(f"{part_base}.part_{part_idx}.npz")["codes"]
        parts.append(part[:remaining].astype(np.int64))
        remaining -= len(parts[-1])
    codes = np.concatenate(parts, axis=0)
    if codes.ndim == 1:
        codes = codes[:, None]
    if codes.shape[1] != expected_m:
        if codes.shape[0] != expected_m:
            raise ValueError(
                f"Unexpected code shape {codes.shape}; expected (N,{expected_m})"
            )
        codes = codes.T
    return np.ascontiguousarray(codes[:n_db], dtype=np.int64)


def output_dir_for_job(result_root: Path, train_qinco_dir: Path, out_root: Path) -> Path:
    rel = train_qinco_dir.relative_to(result_root)
    return out_root / "dense_adc_distances" / rel / "adc_dense_accuracy" / "setting_0"


def complete(out_dir: Path) -> bool:
    meta = out_dir / "metadata.json"
    required = [
        out_dir / "adc_distances_sq.npy",
        out_dir / "exact_distances_sq.npy",
        out_dir / "relative_errors.npy",
        meta,
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        data = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return False
    return int(data.get("dense_adc_version", 0)) >= DENSE_ADC_VERSION


def compute_db_recon_norms(codes_nm: np.ndarray, codebooks_mkd: np.ndarray) -> np.ndarray:
    n_db, n_subquantizers = codes_nm.shape
    dim = codebooks_mkd.shape[2]
    recon = np.zeros((n_db, dim), dtype=np.float32)
    for subquantizer in range(n_subquantizers):
        recon += codebooks_mkd[subquantizer, codes_nm[:, subquantizer]]
    return np.sum(recon * recon, axis=1, dtype=np.float32)


def update_stats(stats: Dict[str, float], values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    stats["count"] += int(finite.size)
    stats["sum"] += float(np.sum(finite, dtype=np.float64))
    stats["sum_sq"] += float(np.sum(finite.astype(np.float64) ** 2))
    stats["min"] = min(stats["min"], float(np.min(finite)))
    stats["max"] = max(stats["max"], float(np.max(finite)))


def finalize_stats(stats: Dict[str, float]) -> Dict[str, float]:
    count = int(stats["count"])
    if count == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    mean = stats["sum"] / count
    var = max(0.0, stats["sum_sq"] / count - mean * mean)
    return {
        "count": count,
        "mean": mean,
        "std": float(np.sqrt(var)),
        "min": stats["min"],
        "max": stats["max"],
    }


def export_job(
    result_root: Path,
    out_root: Path,
    train_qinco_dir: Path,
    codebooks_npz: Path,
    encoded_db: Path,
    dataset: str,
    data_root: Path,
    n_db: int,
    n_queries: int,
    query_batch_size: int,
) -> Path:
    out_dir = output_dir_for_job(result_root, train_qinco_dir, out_root)
    if complete(out_dir):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = np.load(codebooks_npz)
    codebooks_mkd = np.ascontiguousarray(loaded["codebooks"], dtype=np.float32)
    n_subquantizers = int(loaded["M"])
    n_centroids = int(loaded["K"])
    dim = int(loaded["D"])

    db_rel, query_rel = _DATASET_VEC[dataset]
    db = load_vectors(data_root / db_rel, n_db)
    queries = load_vectors(data_root / query_rel, n_queries)
    codes_nm = load_codes(encoded_db, db.shape[0], n_subquantizers)
    if codes_nm.shape[0] != db.shape[0]:
        db = db[: codes_nm.shape[0]]
    if db.shape[1] != dim or queries.shape[1] != dim:
        raise ValueError(
            f"Dimension mismatch for {codebooks_npz}: D={dim}, "
            f"db={db.shape}, queries={queries.shape}"
        )

    n_query = queries.shape[0]
    n_database = db.shape[0]
    db_recon_norms = compute_db_recon_norms(codes_nm, codebooks_mkd)
    db_exact_norms = np.sum(db * db, axis=1, dtype=np.float32)

    adc_path = out_dir / "adc_distances_sq.npy"
    exact_path = out_dir / "exact_distances_sq.npy"
    rel_path = out_dir / "relative_errors.npy"
    adc_mm = np.lib.format.open_memmap(
        adc_path, mode="w+", dtype=np.float32, shape=(n_query, n_database)
    )
    exact_mm = np.lib.format.open_memmap(
        exact_path, mode="w+", dtype=np.float32, shape=(n_query, n_database)
    )
    rel_mm = np.lib.format.open_memmap(
        rel_path, mode="w+", dtype=np.float32, shape=(n_query, n_database)
    )

    stats = {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": 0.0}
    for q_start in range(0, n_query, query_batch_size):
        q_end = min(q_start + query_batch_size, n_query)
        q_batch = np.ascontiguousarray(queries[q_start:q_end], dtype=np.float32)
        query_norms = np.sum(q_batch * q_batch, axis=1, dtype=np.float32)

        luts_mkq = np.matmul(codebooks_mkd, q_batch.T)
        dot_nq = np.zeros((n_database, q_batch.shape[0]), dtype=np.float32)
        for subquantizer in range(n_subquantizers):
            dot_nq += luts_mkq[subquantizer, codes_nm[:, subquantizer], :]
        approx = query_norms[:, None] + db_recon_norms[None, :] - 2.0 * dot_nq.T

        exact_dot = np.matmul(q_batch, db.T)
        exact = query_norms[:, None] + db_exact_norms[None, :] - 2.0 * exact_dot
        np.maximum(exact, 0.0, out=exact)

        rel = np.divide(
            np.abs(approx - exact),
            exact,
            out=np.zeros_like(approx, dtype=np.float32),
            where=exact > 0,
        )

        adc_mm[q_start:q_end] = approx
        exact_mm[q_start:q_end] = exact
        rel_mm[q_start:q_end] = rel
        update_stats(stats, rel)
        print(
            f"{dataset} {codebooks_npz.parent.parent.parent.parent.parent.name} "
            f"queries {q_end}/{n_query}",
            flush=True,
        )

    adc_mm.flush()
    exact_mm.flush()
    rel_mm.flush()

    rel_stats = finalize_stats(stats)
    metadata = {
        "dense_adc_version": DENSE_ADC_VERSION,
        "dataset": dataset,
        "n_queries": n_query,
        "n_db": n_database,
        "dim": dim,
        "n_subquantizers": n_subquantizers,
        "n_centroids": n_centroids,
        "nbits": int(np.log2(n_centroids)),
        "bits_per_vector": int(n_subquantizers * np.log2(n_centroids)),
        "relative_error": rel_stats,
        "adc_distances_sq": str(adc_path),
        "exact_distances_sq": str(exact_path),
        "relative_errors": str(rel_path),
        "source_codebooks": str(codebooks_npz),
        "source_encoded_db": str(encoded_db),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_root",
        default="/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/compressed_qinco_res/unzip",
    )
    parser.add_argument("--out_root", default="/mnthdd/cpanourg/2-hdvc/results/qinco")
    parser.add_argument("--n_db", type=int, default=10000)
    parser.add_argument("--n_queries", type=int, default=1000)
    parser.add_argument("--query_batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("QINCO2_DATA_ROOT", "")).expanduser()
    if not str(data_root):
        raise SystemExit("ERROR: set QINCO2_DATA_ROOT")

    result_root = Path(args.result_root)
    out_root = Path(args.out_root)
    ran = skipped = failed = 0
    for train_dir, codebooks_npz, encoded_db, dataset, _faiss_out in iter_jobs(result_root):
        out_dir = output_dir_for_job(result_root, train_dir, out_root)
        if complete(out_dir):
            skipped += 1
            continue
        if args.limit is not None and ran >= args.limit:
            break
        try:
            print(f"Exporting dense ADC: {dataset} {codebooks_npz}", flush=True)
            export_job(
                result_root,
                out_root,
                train_dir,
                codebooks_npz,
                encoded_db,
                dataset,
                data_root,
                args.n_db,
                args.n_queries,
                args.query_batch_size,
            )
            ran += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED {codebooks_npz}: {exc}", file=sys.stderr, flush=True)
    print(f"Done. ran={ran} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
