#!/usr/bin/env python3
"""Run native RaBitQ-Library on local 10k/1k subsets with reusable IVF artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

for _anc in Path(__file__).resolve().parents:
    if (_anc / "hdvc_paths.py").is_file():
        if str(_anc) not in sys.path:
            sys.path.insert(0, str(_anc))
        break
from hdvc_paths import get_results_root  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    base_path: Path
    query_path: Path
    vec_type: str
    dim: int


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_SPECS = {
    "deep10k": DatasetSpec(
        dataset="deep10k",
        base_path=REPO_ROOT / "local" / "dataset_subsets" / "deep" / "deep_base_10k.fvecs",
        query_path=REPO_ROOT / "local" / "dataset_subsets" / "deep" / "deep_query_1k.fvecs",
        vec_type="fvecs",
        dim=96,
    ),
    "bigann10k": DatasetSpec(
        dataset="bigann10k",
        base_path=REPO_ROOT / "local" / "bigann_subsets" / "bigann_base_10k.bvecs",
        query_path=REPO_ROOT / "local" / "bigann_subsets" / "bigann_query_1k.bvecs",
        vec_type="bvecs",
        dim=128,
    ),
    "gist10k": DatasetSpec(
        dataset="gist10k",
        base_path=REPO_ROOT / "local" / "dataset_subsets" / "gist" / "gist_base_10k.fvecs",
        query_path=REPO_ROOT / "local" / "dataset_subsets" / "gist" / "gist_query_1k.fvecs",
        vec_type="fvecs",
        dim=960,
    ),
    "msmarco10k": DatasetSpec(
        dataset="msmarco10k",
        base_path=REPO_ROOT / "local" / "dataset_subsets" / "msmarco" / "msmarco_base_10k.fvecs",
        query_path=REPO_ROOT / "local" / "dataset_subsets" / "msmarco" / "msmarco_query_1k.fvecs",
        vec_type="fvecs",
        dim=1024,
    ),
    "openai10k": DatasetSpec(
        dataset="openai10k",
        base_path=REPO_ROOT / "local" / "dataset_subsets" / "openai" / "openai_base_10k.fvecs",
        query_path=REPO_ROOT / "local" / "dataset_subsets" / "openai" / "openai_query_1k.fvecs",
        vec_type="fvecs",
        dim=1536,
    ),
}


def _run(cmd: list[str], cwd: Path, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return
    with log_path.open("a") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=str(cwd), check=True, stdout=log, stderr=log)
        log.write("\n")


def _read_ivecs(path: Path) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:]


def _read_fvecs(path: Path) -> np.ndarray:
    return _read_ivecs(path).view(np.float32)


def _read_bvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    dim = np.fromfile(path, dtype=np.int32, count=1)[0]
    stride = dim + 4
    n = raw.size // stride
    arr = raw[: n * stride].reshape(n, stride)[:, 4:]
    return arr.astype(np.float32)


def _write_fvecs(path: Path, x: np.ndarray) -> None:
    x = np.ascontiguousarray(x.astype(np.float32))
    n, d = x.shape
    with path.open("wb") as f:
        for i in range(n):
            np.array([d], dtype=np.int32).tofile(f)
            x[i].view(np.int32).tofile(f)


def _write_ivecs(path: Path, x: np.ndarray) -> None:
    x = np.ascontiguousarray(x.astype(np.int32))
    n, d = x.shape
    with path.open("wb") as f:
        for i in range(n):
            np.array([d], dtype=np.int32).tofile(f)
            x[i].tofile(f)


def _load_vectors(path: Path, vec_type: str) -> np.ndarray:
    if vec_type == "fvecs":
        return _read_fvecs(path)
    if vec_type == "bvecs":
        return _read_bvecs(path)
    raise ValueError(vec_type)


def _logical_cluster_grid(nb: int) -> list[int]:
    vals = [1, 16, 64, 256, 1024]
    out = []
    for val in vals:
        if val <= nb:
            out.append(val)
    return out


def _prepare_shared_ivf(
    spec: DatasetSpec,
    shared_dir: Path,
    clusters: int,
    force: bool,
) -> tuple[Path, Path, Path, Path, float]:
    shared_dir.mkdir(parents=True, exist_ok=True)
    base_fvec = shared_dir / f"{spec.dataset}_base.fvecs"
    query_fvec = shared_dir / f"{spec.dataset}_query.fvecs"
    centroid_fvec = shared_dir / f"{spec.dataset}_centroid_{clusters}.fvecs"
    cid_ivecs = shared_dir / f"{spec.dataset}_cluster_id_{clusters}.ivecs"

    base = _load_vectors(spec.base_path, spec.vec_type)
    query = _load_vectors(spec.query_path, spec.vec_type)

    if force or not base_fvec.exists():
        _write_fvecs(base_fvec, base)
    if force or not query_fvec.exists():
        _write_fvecs(query_fvec, query)

    start = time.perf_counter()
    if force or not centroid_fvec.exists() or not cid_ivecs.exists():
        if clusters == 1:
            centroid = np.mean(base, axis=0, keepdims=True, dtype=np.float32)
            cluster_id = np.zeros((base.shape[0], 1), dtype=np.int32)
        else:
            index = faiss.index_factory(spec.dim, f"IVF{clusters},Flat", faiss.METRIC_L2)
            index.verbose = False
            xb = np.ascontiguousarray(base.astype(np.float32))
            index.train(xb)
            centroid = index.quantizer.reconstruct_n(0, index.nlist).astype(np.float32)
            _, cluster_id = index.quantizer.search(xb, 1)
            cluster_id = cluster_id.astype(np.int32)
        _write_fvecs(centroid_fvec, centroid)
        _write_ivecs(cid_ivecs, cluster_id)
    prep_time = time.perf_counter() - start
    return base_fvec, query_fvec, centroid_fvec, cid_ivecs, prep_time


def _build_binaries(lib_root: Path, build_log: Path, skip_build: bool) -> None:
    build_dir = lib_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    if skip_build:
        return
    _run(["cmake", ".."], cwd=build_dir, log_path=build_log)
    _run(["cmake", "--build", ".", "-j"], cwd=build_dir, log_path=build_log)


def _parse_eval_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="") as f:
        return next(csv.DictReader(f))


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="RaBitQ-Library IVF grid on local 10k/1k subsets.")
    p.add_argument("--lib_root", default=str(REPO_ROOT / "lib" / "RaBitQ-Library"))
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["deep10k", "bigann10k", "gist10k", "msmarco10k", "openai10k"],
    )
    p.add_argument("--bits", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--clusters", type=int, nargs="+", default=None)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--default_nprobe", type=int, default=10)
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--warmup_runs", type=int, default=2)
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--force_index", action="store_true")
    p.add_argument("--skip_build", action="store_true")
    p.add_argument("--use_hacc", dest="use_hacc", action="store_true")
    p.add_argument("--no_hacc", dest="use_hacc", action="store_false")
    p.add_argument(
        "--output_root",
        type=Path,
        default=get_results_root() / "rabitq_native_ivf",
    )
    p.set_defaults(use_hacc=True)
    args = p.parse_args()

    lib_root = Path(args.lib_root).resolve()
    bin_dir = lib_root / "bin"
    index_bin = bin_dir / "ivf_rabitq_indexing"
    eval_bin = bin_dir / "ivf_rabitq_eval"

    args.output_root.mkdir(parents=True, exist_ok=True)
    build_log = args.output_root / "build.log"
    progress_log = args.output_root / "progress.jsonl"

    requested_bits = sorted(set(int(b) for b in args.bits))
    supported_bits = [b for b in requested_bits if 1 <= b <= 9]
    skipped_bits = [b for b in requested_bits if b not in supported_bits]
    if skipped_bits:
        _append_jsonl(
            progress_log,
            {
                "event": "skip_bits",
                "reason": "RaBitQ-Library currently supports total bits 1..9 only",
                "bits": skipped_bits,
                "ts": time.time(),
            },
        )

    _build_binaries(lib_root, build_log, args.skip_build)
    if not index_bin.exists() or not eval_bin.exists():
        raise FileNotFoundError("Expected ivf_rabitq_indexing and ivf_rabitq_eval in lib/RaBitQ-Library/bin")

    for dataset in args.datasets:
        if dataset not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset: {dataset}")
        spec = DATASET_SPECS[dataset]
        if not spec.base_path.exists() or not spec.query_path.exists():
            raise FileNotFoundError(f"Missing subset vectors for {dataset}")

        shared_dir = args.output_root / "shared_ivf" / dataset
        dataset_dir = args.output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_log = dataset_dir / "run.log"
        rows: list[dict] = []

        base = _load_vectors(spec.base_path, spec.vec_type)
        query = _load_vectors(spec.query_path, spec.vec_type)
        nb = int(base.shape[0])
        nq = int(query.shape[0])
        cluster_grid = sorted(set(args.clusters if args.clusters else _logical_cluster_grid(nb)))

        for clusters in cluster_grid:
            base_fvec, query_fvec, centroid_fvec, cid_ivecs, prep_time = _prepare_shared_ivf(
                spec,
                shared_dir,
                clusters,
                args.force_prepare,
            )
            nprobe = 1 if clusters == 1 else min(int(args.default_nprobe), int(clusters))

            _append_jsonl(
                progress_log,
                {
                    "event": "prepared_ivf",
                    "dataset": dataset,
                    "clusters": clusters,
                    "nprobe": nprobe,
                    "prep_time_s": prep_time,
                    "shared_dir": str(shared_dir),
                    "ts": time.time(),
                },
            )

            for bits in supported_bits:
                run_dir = dataset_dir / f"K{clusters}_b{bits}"
                run_dir.mkdir(parents=True, exist_ok=True)
                eval_csv = run_dir / "eval.csv"
                meta_json = run_dir / "metadata.json"
                index_path = run_dir / f"ivf_rabitqlib_K{clusters}_b{bits}.index"

                if meta_json.exists() and not args.force_index:
                    rows.append(json.loads(meta_json.read_text())["row"])
                    continue

                _append_jsonl(
                    progress_log,
                    {
                        "event": "start_run",
                        "dataset": dataset,
                        "clusters": clusters,
                        "bits": bits,
                        "nprobe": nprobe,
                        "ts": time.time(),
                    },
                )

                t0 = time.perf_counter()
                _run(
                    [
                        str(index_bin),
                        str(base_fvec),
                        str(centroid_fvec),
                        str(cid_ivecs),
                        str(bits),
                        str(index_path),
                        "l2",
                        "false",
                    ],
                    cwd=bin_dir,
                    log_path=dataset_log,
                )
                encoding_time_s = time.perf_counter() - t0

                _run(
                    [
                        str(eval_bin),
                        str(index_path),
                        str(base_fvec),
                        str(query_fvec),
                        str(bits),
                        str(args.topk),
                        str(nprobe),
                        str(eval_csv),
                        str(args.n_runs),
                        str(args.warmup_runs),
                        "true" if args.use_hacc else "false",
                    ],
                    cwd=bin_dir,
                    log_path=dataset_log,
                )
                metrics = _parse_eval_csv(eval_csv)

                row = {
                    "method": "RaBitQ-Library",
                    "dataset": dataset,
                    "experiment_folder": str(run_dir),
                    "nq": nq,
                    "nb": nb,
                    "nb_sample": int(metrics["topk"]),
                    "dim": spec.dim,
                    "n_subquantizers": 0,
                    "nbits": bits,
                    "bits_per_vector": spec.dim * bits,
                    "train_size": nb,
                    "train_time_s": float(prep_time),
                    "encoding_time_s": float(encoding_time_s),
                    "distance_table_time_s": np.nan,
                    "cdist_time_s": np.nan,
                    "adc_time_s": float(metrics["adc_time_s_mean"]),
                    "adc_time_s_std": float(metrics["adc_time_s_std"]),
                    "per_query_us_mean": float(metrics["per_query_us_mean"]),
                    "per_pair_ns_mean": float(metrics["per_pair_ns_mean"]),
                    "rel_error_mean": float(metrics["rel_error_mean"]),
                    "rel_error_std": float(metrics["rel_error_std"]),
                    "nq_sample": int(metrics["nq"]),
                    "sample_mode": (
                        f"rabitqlib_native_topk={metrics['topk']}_nprobe={metrics['nprobe']}"
                    ),
                    "train_path": str(base_fvec),
                    "seed": np.nan,
                    "n_centroids": clusters,
                    "nprobe": int(metrics["nprobe"]),
                    "centroid_fvecs": str(centroid_fvec),
                    "cluster_ids_ivecs": str(cid_ivecs),
                    "eval_csv": str(eval_csv),
                }
                payload = {
                    "dataset": dataset,
                    "clusters": clusters,
                    "bits": bits,
                    "row": row,
                    "metrics": metrics,
                }
                meta_json.write_text(json.dumps(payload, indent=2))
                rows.append(row)

                _append_jsonl(
                    progress_log,
                    {
                        "event": "finish_run",
                        "dataset": dataset,
                        "clusters": clusters,
                        "bits": bits,
                        "adc_time_s": row["adc_time_s"],
                        "rel_error_mean": row["rel_error_mean"],
                        "ts": time.time(),
                    },
                )

        if rows:
            out_csv = get_results_root() / "rabitq" / f"{dataset}_RaBitQ-Library_adc_vs_exact_eval.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd

            df = pd.DataFrame(rows).sort_values(["n_centroids", "nbits"])
            df.to_csv(out_csv, index=False)
            _append_jsonl(
                progress_log,
                {
                    "event": "write_csv",
                    "dataset": dataset,
                    "path": str(out_csv),
                    "rows": len(df),
                    "ts": time.time(),
                },
            )
            print(out_csv, flush=True)


if __name__ == "__main__":
    main()
