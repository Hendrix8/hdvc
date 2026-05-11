#!/usr/bin/env python3
"""Compute PQ quality metrics for existing FAISS PQ artifacts.

Metrics:
- distortion_error: mean ||x - decode(encode(x))|| over database sample
- spearman: mean Spearman rank correlation between exact and ADC distances

This script is resumable: it writes one JSON per input row, then aggregates CSVs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from distance_eval.codecs import decode_pq_codes
from scripts.evals.export_pq_faiss_adc_distances import DATASET_VEC, load_vectors


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    return Path(text) if text else None


def _rank_1d(values: np.ndarray) -> np.ndarray:
    """Fast ranks for distance arrays. Ties get arbitrary stable ranks; adequate for float distances."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.arange(order.shape[0], dtype=np.float64)
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(x, y) / denom)


def mean_spearman_from_distance_files(
    exact_path: Path,
    adc_path: Path,
    max_queries: int | None,
) -> tuple[float, int]:
    exact = np.load(exact_path, mmap_mode="r")
    adc = np.load(adc_path, mmap_mode="r")
    if exact.shape != adc.shape:
        raise ValueError(f"shape mismatch exact={exact.shape}, adc={adc.shape}")
    nq = exact.shape[0] if max_queries is None else min(int(max_queries), exact.shape[0])
    vals: list[float] = []
    for query_index in range(nq):
        r_exact = _rank_1d(np.asarray(exact[query_index], dtype=np.float32))
        r_adc = _rank_1d(np.asarray(adc[query_index], dtype=np.float32))
        corr = _pearson_corr(r_exact, r_adc)
        if np.isfinite(corr):
            vals.append(corr)
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def distortion_error(
    index_path: Path,
    dataset: str,
    data_root: Path,
    max_samples: int,
    chunk_size: int,
) -> tuple[float, int]:
    if dataset not in DATASET_VEC:
        raise ValueError(f"unknown dataset {dataset!r}")
    db_rel, _ = DATASET_VEC[dataset]
    db = load_vectors(data_root / db_rel, int(max_samples))
    index = faiss.read_index(str(index_path))
    if not hasattr(index, "pq"):
        raise ValueError(f"index has no ProductQuantizer: {index_path}")
    pq = index.pq
    centroids = faiss.vector_to_array(pq.centroids).reshape(
        int(pq.M), 1 << int(pq.nbits), int(pq.dsub)
    ).astype(np.float32, copy=False)
    distances = []
    for start in range(0, len(db), int(chunk_size)):
        chunk = np.ascontiguousarray(db[start : start + int(chunk_size)], dtype=np.float32)
        codes_flat = pq.compute_codes(chunk).ravel()
        codes = decode_pq_codes(codes_flat, chunk.shape[0], int(pq.M), int(pq.nbits))
        recon = np.empty_like(chunk)
        for m in range(int(pq.M)):
            begin = m * int(pq.dsub)
            end = begin + int(pq.dsub)
            recon[:, begin:end] = centroids[m, codes[:, m]]
        distances.append(np.linalg.norm(chunk - recon, axis=1))
    if not distances:
        return float("nan"), 0
    all_distances = np.concatenate(distances)
    return float(np.mean(all_distances)), int(all_distances.size)


def result_path_for_row(out_dir: Path, row_index: int) -> Path:
    return out_dir / "row_results" / f"row_{int(row_index):04d}.json"


def compute_row(row: dict[str, Any], args_dict: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(args_dict["out_dir"])
    row_index = int(row["row_index"])
    result_path = result_path_for_row(out_dir, row_index)
    if result_path.exists() and not args_dict.get("force", False):
        return json.loads(result_path.read_text())

    index_path = _as_path(row.get("artifact_index"))
    exact_path = _as_path(row.get("exact_distances_sq"))
    adc_path = _as_path(row.get("adc_distances_sq"))
    if index_path is None or not index_path.exists():
        raise FileNotFoundError(f"missing artifact_index for row {row_index}: {index_path}")
    if exact_path is None or not exact_path.exists():
        raise FileNotFoundError(f"missing exact_distances_sq for row {row_index}: {exact_path}")
    if adc_path is None or not adc_path.exists():
        raise FileNotFoundError(f"missing adc_distances_sq for row {row_index}: {adc_path}")

    out = {
        "method": "PQ",
        "dataset": row["dataset"],
        "row_index": row_index,
        "M": int(row["n_subquantizers"]),
        "nbits": int(row["nbits"]),
        "bits_per_vector": int(row["bits_per_vector"]),
        "train_size": int(row["train_size"]),
        "nq": int(row["nq"]),
        "nb_sample": int(row["nb_sample"]),
        "dim": int(row["dim"]),
        "artifact_index": str(index_path),
        "exact_distances_sq": str(exact_path),
        "adc_distances_sq": str(adc_path),
    }

    if args_dict.get("distortion", True):
        dist, count = distortion_error(
            index_path,
            str(row["dataset"]),
            Path(args_dict["data_root"]),
            int(args_dict["max_rec_samples"]),
            int(args_dict["rec_chunk_size"]),
        )
        out["distortion_error"] = dist
        out["distortion_count"] = count

    if args_dict.get("spearman", True):
        sp, count = mean_spearman_from_distance_files(
            exact_path,
            adc_path,
            None if args_dict["max_spearman_queries"] is None else int(args_dict["max_spearman_queries"]),
        )
        out["spearman"] = sp
        out["spearman_query_count"] = count

    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, sort_keys=True))
    tmp.replace(result_path)
    return out


def aggregate_outputs(out_dir: Path, clean_res_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((out_dir / "row_results").glob("row_*.json")):
        rows.append(json.loads(path.read_text()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    keys = ["method", "dataset", "M", "nbits", "bits_per_vector", "train_size", "nq", "nb_sample", "dim"]
    agg_spec = {
        "row_index": "count",
        "distortion_error": "mean",
        "distortion_count": "max",
        "spearman": "mean",
        "spearman_query_count": "max",
    }
    present_agg = {k: v for k, v in agg_spec.items() if k in df.columns}
    clean = df.groupby(keys, dropna=False).agg(present_agg).reset_index()
    clean = clean.rename(columns={"row_index": "n_exports"})
    clean = clean.sort_values(["dataset", "bits_per_vector", "M", "nbits"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_dir / "pq_quality_metrics.csv", index=False)
    clean_res_dir.mkdir(parents=True, exist_ok=True)
    for dataset, sub in clean.groupby("dataset", sort=True):
        sub.to_csv(clean_res_dir / f"{dataset}_PQ_quality_metrics.csv", index=False)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("/mnthdd/cpanourg/2-hdvc/results/pq/pq_faiss_adc_summary.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("/mnthdd/cpanourg/2-hdvc/results/pq/quality_metrics"))
    parser.add_argument("--clean_res_dir", type=Path, default=Path("/home/cpanourg/projects/2-hdvc/results/clean_res"))
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/.adc_data_symlinks"),
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max_workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--max_rec_samples", type=int, default=10_000)
    parser.add_argument("--rec_chunk_size", type=int, default=2_000)
    parser.add_argument("--max_spearman_queries", type=int, default=None, help="default: all queries in saved dense files")
    parser.add_argument("--no_distortion", action="store_true")
    parser.add_argument("--no_spearman", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    if args.datasets:
        summary = summary[summary["dataset"].isin(args.datasets)].copy()
    if summary.empty:
        raise SystemExit("No rows to process")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args_dict = {
        "out_dir": str(args.out_dir),
        "data_root": str(args.data_root),
        "max_rec_samples": args.max_rec_samples,
        "rec_chunk_size": args.rec_chunk_size,
        "max_spearman_queries": args.max_spearman_queries,
        "distortion": not args.no_distortion,
        "spearman": not args.no_spearman,
        "force": args.force,
    }

    rows = summary.to_dict("records")
    print(f"Processing {len(rows)} rows with {args.max_workers} workers", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(compute_row, row, args_dict) for row in rows]
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                res = fut.result()
                print(
                    f"[{done}/{len(futures)}] {res['dataset']} row={res['row_index']} M={res['M']} nbits={res['nbits']}",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                print(f"[{done}/{len(futures)}] FAILED: {exc}", flush=True)

    clean = aggregate_outputs(args.out_dir, args.clean_res_dir)
    print(f"Wrote {args.out_dir / 'pq_quality_metrics.csv'} ({len(clean)} rows)", flush=True)
    print(f"Wrote per-dataset quality CSVs to {args.clean_res_dir}", flush=True)
    if failures:
        raise SystemExit(f"Completed with {failures} failures")


if __name__ == "__main__":
    main()
