#!/usr/bin/env python3
"""
Export PQ FAISS ADC timing and dense ADC distance/relative-error matrices.

This mirrors the QINCo workflow:
  - timing uses FAISS C++ IndexPQ.search(k=1), including internal LUT build + scan
  - relative-error statistics use dense ADC distances computed separately
  - dense matrices are saved as .npy memmaps for later inspection
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from distance_eval.artifacts import resolve_pq_index_dir
from distance_eval.codecs import decode_pq_codes


PQ_ADC_VERSION = 1

DATASET_VEC = {
    "bigann": ("bigann/SIFT1M/bigann_base.bvecs", "bigann/SIFT1M/bigann_query.bvecs"),
    "deep": ("deep/test_1m.fvecs", "deep/query_10k.fvecs"),
    "gist": ("gist/gist_base.fvecs", "gist/gist_query.fvecs"),
    "msmarco": ("msmarco/base.fvecs", "msmarco/query.fvecs"),
    "openai": ("openai/openai_base1m.fvecs", "openai/openai_query10k.fvecs"),
}


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


def map_experiment_folder(path: str, from_prefix: str, to_prefix: str) -> Path:
    path = str(path)
    if from_prefix and to_prefix and path.startswith(from_prefix):
        path = to_prefix + path[len(from_prefix) :]
    return Path(path)


def safe_name(row: pd.Series, artifact_dir: Path) -> str:
    folder = artifact_dir.name
    folder = re.sub(r"[^A-Za-z0-9_.=-]+", "_", folder)
    return f"{row['dataset']}_{folder}"


def value_present(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def complete(out_dir: Path) -> bool:
    meta = out_dir / "metadata.json"
    required = [
        meta,
        out_dir / "adc_distances_sq.npy",
        out_dir / "exact_distances_sq.npy",
        out_dir / "relative_errors.npy",
    ]
    if not all(p.is_file() for p in required):
        return False
    try:
        data = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return False
    return int(data.get("pq_adc_version", 0)) >= PQ_ADC_VERSION


def extract_pq(index: faiss.Index) -> faiss.ProductQuantizer:
    if hasattr(index, "pq"):
        return index.pq
    if hasattr(index, "index"):
        sub = faiss.downcast_index(index.index)
        if hasattr(sub, "pq"):
            return sub.pq
    raise ValueError("Cannot extract ProductQuantizer")


def pq_centroids(pq: faiss.ProductQuantizer) -> np.ndarray:
    centroids = faiss.vector_to_array(pq.centroids)
    return centroids.reshape(int(pq.M), 1 << int(pq.nbits), int(pq.dsub)).astype(
        np.float32, copy=False
    )


def pq_codes(pq: faiss.ProductQuantizer, db: np.ndarray) -> np.ndarray:
    codes_flat = pq.compute_codes(np.ascontiguousarray(db, dtype=np.float32)).ravel()
    return decode_pq_codes(codes_flat, db.shape[0], int(pq.M), int(pq.nbits))


def update_stats(stats: dict[str, float], values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    stats["count"] += int(finite.size)
    stats["sum"] += float(np.sum(finite, dtype=np.float64))
    stats["sum_sq"] += float(np.sum(finite.astype(np.float64) ** 2))
    stats["min"] = min(stats["min"], float(np.min(finite)))
    stats["max"] = max(stats["max"], float(np.max(finite)))


def finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    count = int(stats["count"])
    if count == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = stats["sum"] / count
    var = max(0.0, stats["sum_sq"] / count - mean * mean)
    return {
        "count": count,
        "mean": mean,
        "std": float(np.sqrt(var)),
        "min": stats["min"],
        "max": stats["max"],
    }


def dense_pq_distances(
    queries: np.ndarray,
    db: np.ndarray,
    codes: np.ndarray,
    centroids: np.ndarray,
    out_dir: Path,
    query_batch_size: int,
) -> dict[str, float]:
    nq, dim = queries.shape
    nb = db.shape[0]
    M = centroids.shape[0]
    dsub = centroids.shape[2]
    if dim != M * dsub:
        raise ValueError(f"dim={dim} incompatible with M={M}, dsub={dsub}")

    adc_path = out_dir / "adc_distances_sq.npy"
    exact_path = out_dir / "exact_distances_sq.npy"
    rel_path = out_dir / "relative_errors.npy"
    adc_mm = np.lib.format.open_memmap(adc_path, mode="w+", dtype=np.float32, shape=(nq, nb))
    exact_mm = np.lib.format.open_memmap(exact_path, mode="w+", dtype=np.float32, shape=(nq, nb))
    rel_mm = np.lib.format.open_memmap(rel_path, mode="w+", dtype=np.float32, shape=(nq, nb))

    db_norms = np.sum(db * db, axis=1, dtype=np.float32)
    stats = {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": 0.0}

    for q_start in range(0, nq, query_batch_size):
        q_end = min(q_start + query_batch_size, nq)
        q_batch = np.ascontiguousarray(queries[q_start:q_end], dtype=np.float32)
        q_norms = np.sum(q_batch * q_batch, axis=1, dtype=np.float32)

        approx = np.zeros((q_batch.shape[0], nb), dtype=np.float32)
        for m in range(M):
            q_sub = q_batch[:, m * dsub : (m + 1) * dsub]
            c = centroids[m]
            c_norms = np.sum(c * c, axis=1, dtype=np.float32)
            table = (
                np.sum(q_sub * q_sub, axis=1, dtype=np.float32)[:, None]
                + c_norms[None, :]
                - 2.0 * np.matmul(q_sub, c.T)
            )
            approx += table[:, codes[:, m]]

        exact_dot = np.matmul(q_batch, db.T)
        exact = q_norms[:, None] + db_norms[None, :] - 2.0 * exact_dot
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
        print(f"queries {q_end}/{nq}", flush=True)

    adc_mm.flush()
    exact_mm.flush()
    rel_mm.flush()
    return finalize_stats(stats)


def time_faiss_pq_search(index_path: Path, db: np.ndarray, queries: np.ndarray, warmup_runs: int, n_runs: int) -> dict[str, float]:
    index = faiss.read_index(str(index_path))
    index.add(np.ascontiguousarray(db, dtype=np.float32))
    q = np.ascontiguousarray(queries, dtype=np.float32)
    for _ in range(max(0, warmup_runs)):
        index.search(q, 1)
    times = []
    import time

    for _ in range(max(1, n_runs)):
        start = time.perf_counter()
        index.search(q, 1)
        times.append(time.perf_counter() - start)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "adc_time_s_mean": float(arr.mean()),
        "adc_time_s_std": float(arr.std()) if len(arr) > 1 else 0.0,
        "n_runs": int(n_runs),
    }


def export_row(
    row: pd.Series,
    row_index: int,
    out_root: Path,
    data_root: Path,
    path_map_from: str,
    path_map_to: str,
    n_db: int,
    n_queries: int,
    query_batch_size: int,
    warmup_runs: int,
    n_runs: int,
) -> Path:
    if "artifact_index" in row and value_present(row["artifact_index"]):
        index_path = Path(str(row["artifact_index"]))
        artifact_dir = index_path.parent
    else:
        exp = map_experiment_folder(str(row["experiment_folder"]), path_map_from, path_map_to)
        artifact_dir = resolve_pq_index_dir(
            exp,
            int(row["n_subquantizers"]),
            int(row["nbits"]),
            int(row["train_size"]),
        )
        index_path = artifact_dir / "pq_model.index"
    out_dir = out_root / "pq_faiss_adc" / str(row["dataset"]) / safe_name(row, artifact_dir)
    out_dir = out_dir / f"row_{row_index:04d}"
    if complete(out_dir):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    db_rel, query_rel = DATASET_VEC[str(row["dataset"])]
    db = load_vectors(data_root / db_rel, n_db)
    queries = load_vectors(data_root / query_rel, n_queries)
    index = faiss.read_index(str(index_path))
    pq = extract_pq(index)
    row_dim = int(row["dim"]) if "dim" in row and value_present(row["dim"]) else int(index.d)
    row_m = int(row["n_subquantizers"]) if "n_subquantizers" in row and value_present(row["n_subquantizers"]) else int(pq.M)
    row_nbits = int(row["nbits"]) if "nbits" in row and value_present(row["nbits"]) else int(pq.nbits)
    row_bpv = int(row["bits_per_vector"]) if "bits_per_vector" in row and value_present(row["bits_per_vector"]) else row_m * row_nbits
    row_train_size = int(row["train_size"]) if "train_size" in row and value_present(row["train_size"]) else -1
    centroids = pq_centroids(pq)
    codes = pq_codes(pq, db)

    rel_stats = dense_pq_distances(queries, db, codes, centroids, out_dir, query_batch_size)
    timing = time_faiss_pq_search(index_path, db, queries, warmup_runs, n_runs)
    n_pairs = int(queries.shape[0] * db.shape[0])

    metadata = {
        "pq_adc_version": PQ_ADC_VERSION,
        "method": "PQ",
        "dataset": str(row["dataset"]),
        "row_index": int(row_index),
        "n_queries": int(queries.shape[0]),
        "n_db": int(db.shape[0]),
        "dim": row_dim,
        "n_subquantizers": row_m,
        "nbits": row_nbits,
        "n_centroids": int(1 << row_nbits),
        "bits_per_vector": row_bpv,
        "train_size": row_train_size,
        "relative_error": rel_stats,
        "faiss_adc_timing": {
            **timing,
            "per_pair_adc_time_ns_mean": timing["adc_time_s_mean"] / max(1, n_pairs) * 1e9,
            "per_pair_adc_time_ns_std": timing["adc_time_s_std"] / max(1, n_pairs) * 1e9,
            "mode": "IndexPQ.search_top1",
        },
        "adc_distances_sq": str(out_dir / "adc_distances_sq.npy"),
        "exact_distances_sq": str(out_dir / "exact_distances_sq.npy"),
        "relative_errors": str(out_dir / "relative_errors.npy"),
        "artifact_index": str(index_path),
        "source_experiment_folder": str(row["experiment_folder"]) if "experiment_folder" in row else str(artifact_dir),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_dir


def load_rows(input_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(input_dir.glob("*_PQ_adc_vs_exact_eval.csv")):
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *_PQ_adc_vs_exact_eval.csv in {input_dir}")
    return pd.concat(frames, ignore_index=True)


def load_rows_from_index_root(index_root: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"subq(?P<M>\d+)_nbits(?P<nbits>\d+)_train(?P<train>\d+)")
    for row_index, index_path in enumerate(sorted(index_root.glob("*/*/pq_model.index"))):
        dataset = index_path.parent.parent.name
        match = pattern.search(index_path.parent.name)
        if not match:
            print(f"Skipping unparseable PQ folder: {index_path.parent}", file=sys.stderr)
            continue
        M = int(match.group("M"))
        nbits = int(match.group("nbits"))
        train_size = int(match.group("train"))
        rows.append(
            {
                "dataset": dataset,
                "row_index": row_index,
                "n_subquantizers": M,
                "nbits": nbits,
                "bits_per_vector": M * nbits,
                "train_size": train_size,
                "artifact_index": str(index_path),
                "experiment_folder": str(index_path.parent),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No */*/pq_model.index files in {index_root}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="/mnthdd/cpanourg/2-hdvc/results/urania_results/results/relerr_cpp")
    parser.add_argument("--index_root", default=None, help="Scan dataset/*/pq_model.index artifacts instead of CSV rows")
    parser.add_argument("--out_root", default="/mnthdd/cpanourg/2-hdvc/results/pq")
    parser.add_argument("--data_root", default="/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/.adc_data_symlinks")
    parser.add_argument("--path_map_from", default="/data/cpanourg/2-hdvc/results")
    parser.add_argument("--path_map_to", default="/mnthdd/cpanourg/2-hdvc/results/urania_results/results")
    parser.add_argument("--n_db", type=int, default=10000)
    parser.add_argument("--n_queries", type=int, default=1000)
    parser.add_argument("--query_batch_size", type=int, default=8)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    faiss.omp_set_num_threads(max(1, int(args.num_threads)))
    rows = load_rows_from_index_root(Path(args.index_root)) if args.index_root else load_rows(Path(args.input_dir))
    if args.dataset:
        rows = rows[rows["dataset"] == args.dataset].copy()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ran = skipped = failed = 0
    for row_index, row in rows.iterrows():
        exp = map_experiment_folder(str(row["experiment_folder"]), args.path_map_from, args.path_map_to)
        try:
            if "artifact_index" in row and value_present(row["artifact_index"]):
                artifact_dir = Path(str(row["artifact_index"])).parent
            else:
                artifact_dir = resolve_pq_index_dir(
                    exp,
                    int(row["n_subquantizers"]),
                    int(row["nbits"]),
                    int(row["train_size"]),
                )
            out_dir = out_root / "pq_faiss_adc" / str(row["dataset"]) / safe_name(row, artifact_dir) / f"row_{row_index:04d}"
            if complete(out_dir):
                skipped += 1
                continue
            if args.limit is not None and ran >= args.limit:
                break
            print(f"Exporting PQ row={row_index} {row['dataset']} M={row['n_subquantizers']} nbits={row['nbits']}", flush=True)
            export_row(
                row,
                row_index,
                out_root,
                Path(args.data_root),
                args.path_map_from,
                args.path_map_to,
                args.n_db,
                args.n_queries,
                args.query_batch_size,
                args.warmup_runs,
                args.n_runs,
            )
            ran += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED row={row_index}: {exc}", file=sys.stderr, flush=True)
    print(f"Done. ran={ran} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
