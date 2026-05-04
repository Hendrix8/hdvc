#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for _anc in Path(__file__).resolve().parents:
    if (_anc / "hdvc_paths.py").is_file():
        if str(_anc) not in sys.path:
            sys.path.insert(0, str(_anc))
        break
from hdvc_paths import get_results_root  # noqa: E402

import numpy as np


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


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


def _load_vectors(path: Path, vec_type: str) -> np.ndarray:
    if vec_type == "bvecs":
        return _read_bvecs(path)
    if vec_type == "fvecs":
        return _read_fvecs(path)
    raise ValueError(f"Unsupported vec_type: {vec_type}")


def _prepare_dataset(
    lib_root: Path,
    dataset: str,
    base_path: Path,
    query_path: Path,
    gt_ivecs_path: Path,
    vec_type: str,
    force_prepare: bool,
    max_base: int | None,
    max_query: int | None,
) -> tuple[Path, int, int, int]:
    ds_dir = lib_root / "data" / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    base = _load_vectors(base_path, vec_type)
    query = _load_vectors(query_path, vec_type)
    if max_base is not None:
        base = base[:max_base]
    if max_query is not None:
        query = query[:max_query]

    base_out = ds_dir / f"{dataset}_base.fvecs"
    query_out = ds_dir / f"{dataset}_query.fvecs"
    gt_out = ds_dir / f"{dataset}_groundtruth.ivecs"

    if force_prepare or not base_out.exists():
        _write_fvecs(base_out, base)
    if force_prepare or not query_out.exists():
        _write_fvecs(query_out, query)
    if force_prepare or not gt_out.exists():
        shutil.copyfile(gt_ivecs_path, gt_out)

    return ds_dir, int(base.shape[0]), int(query.shape[0]), int(base.shape[1])


def _link_or_copy(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _prepare_from_existing_ivf(
    lib_root: Path,
    dataset: str,
    prepared_dir: Path,
    clusters: int,
    force: bool,
) -> tuple[Path, int, int, int]:
    """Reuse an existing prepared dataset and IVF partition.

    This is the fairest SAQ comparison path: RaBitQ and SAQ consume the exact
    same base/query vectors, centroids, and cluster ids instead of retraining
    IVF with a merely matching random seed.
    """
    src_files = {
        "base": prepared_dir / f"{dataset}_base.fvecs",
        "query": prepared_dir / f"{dataset}_query.fvecs",
        "centroid": prepared_dir / f"{dataset}_centroid_{clusters}.fvecs",
        "cid": prepared_dir / f"{dataset}_cluster_id_{clusters}.ivecs",
    }
    missing = [str(p) for p in src_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing reusable SAQ/IVF artifacts:\n" + "\n".join(missing))

    ds_dir = lib_root / "data" / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    dst_files = {
        "base": ds_dir / f"{dataset}_base.fvecs",
        "query": ds_dir / f"{dataset}_query.fvecs",
        "centroid": ds_dir / f"{dataset}_centroid_{clusters}.fvecs",
        "cid": ds_dir / f"{dataset}_cluster_id_{clusters}.ivecs",
    }
    for key in src_files:
        _link_or_copy(src_files[key], dst_files[key], force)

    base = _read_fvecs(dst_files["base"])
    query = _read_fvecs(dst_files["query"])
    return ds_dir, int(base.shape[0]), int(query.shape[0]), int(base.shape[1])


def _parse_relerr(path: Path) -> dict[str, str]:
    with path.open("r", newline="") as f:
        return next(csv.DictReader(f))


def _write_project_csv(
    out_csv: Path,
    dataset: str,
    experiment_folder: Path,
    base_path: Path,
    nq: int,
    nb: int,
    dim: int,
    bits: list[int],
    relerrs: dict[int, dict[str, str]],
    times: dict[int, dict[str, float]] | None = None,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "dataset",
        "experiment_folder",
        "nq",
        "nb",
        "nb_sample",
        "dim",
        "n_subquantizers",
        "nbits",
        "bits_per_vector",
        "train_size",
        "train_time_s",
        "encoding_time_s",
        "distance_table_time_s",
        "cdist_time_s",
        "adc_time_s",
        "rel_error_mean",
        "rel_error_std",
        "nq_sample",
        "sample_mode",
        "train_path",
        "seed",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for b in bits:
            rr = relerrs[b]
            tm = times.get(b, {}) if times is not None else {}
            w.writerow(
                {
                    "method": "RaBitQ",
                    "dataset": dataset,
                    "experiment_folder": str(experiment_folder),
                    "nq": nq,
                    "nb": nb,
                    "nb_sample": 100,
                    "dim": dim,
                    "n_subquantizers": 0,
                    "nbits": b,
                    "bits_per_vector": dim * b,
                    "train_size": nb,
                    "train_time_s": np.nan,
                    "encoding_time_s": tm.get("encoding_time_s", np.nan),
                    "distance_table_time_s": np.nan,
                    "cdist_time_s": np.nan,
                    "adc_time_s": tm.get("adc_time_s", np.nan),
                    "rel_error_mean": float(rr["rel_error_mean"]),
                    "rel_error_std": float(rr["rel_error_std"]),
                    "nq_sample": nq,
                    "sample_mode": f"rabitqlib_topk={rr['topk']}_nprobe={rr['nprobe']}",
                    "train_path": str(base_path),
                    "seed": np.nan,
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Run IVF + RaBitQ-Library and export relerr CSV.")
    p.add_argument("--lib_root", default="/home/cpanourg/projects/2-hdvc/lib/RaBitQ-Library")
    p.add_argument("--dataset", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--groundtruth", default=None)
    p.add_argument("--vec_type", choices=["bvecs", "fvecs"], default="bvecs")
    p.add_argument("--clusters", type=int, default=4096)
    p.add_argument("--bits", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--nprobe", type=int, default=200)
    p.add_argument("--skip_build", action="store_true")
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument(
        "--reuse_prepared_dir",
        default=None,
        help=(
            "Directory containing <dataset>_base/query.fvecs plus "
            "<dataset>_centroid_<K>.fvecs and <dataset>_cluster_id_<K>.ivecs. "
            "Use this to share SAQ's exact IVF partition."
        ),
    )
    p.add_argument(
        "--force_ivf",
        action="store_true",
        help="Retrain/rewrite IVF artifacts when not using --reuse_prepared_dir.",
    )
    p.add_argument("--faster_quant", action="store_true")
    p.add_argument("--max_base", type=int, default=None)
    p.add_argument("--max_query", type=int, default=None)
    p.add_argument(
        "--out_csv",
        default=None,
        help="Project-schema CSV path; default {results}/rabitq/<dataset>_RaBitQ-Library_adc_vs_exact_eval.csv",
    )
    args = p.parse_args()

    lib_root = Path(args.lib_root).resolve()
    base_path = Path(args.base).resolve()
    query_path = Path(args.query).resolve()
    gt_path = Path(args.groundtruth).resolve() if args.groundtruth else None
    bits = sorted(set(args.bits))

    if args.reuse_prepared_dir:
        ds_dir, nb, nq, dim = _prepare_from_existing_ivf(
            lib_root=lib_root,
            dataset=args.dataset,
            prepared_dir=Path(args.reuse_prepared_dir).resolve(),
            clusters=args.clusters,
            force=args.force_prepare,
        )
    else:
        if gt_path is None:
            raise ValueError("--groundtruth is required unless --reuse_prepared_dir is used")
        ds_dir, nb, nq, dim = _prepare_dataset(
            lib_root,
            args.dataset,
            base_path,
            query_path,
            gt_path,
            args.vec_type,
            args.force_prepare,
            args.max_base,
            args.max_query,
        )

    base_fvec = ds_dir / f"{args.dataset}_base.fvecs"
    query_fvec = ds_dir / f"{args.dataset}_query.fvecs"
    centroid_fvec = ds_dir / f"{args.dataset}_centroid_{args.clusters}.fvecs"
    cid_ivecs = ds_dir / f"{args.dataset}_cluster_id_{args.clusters}.ivecs"

    if not args.reuse_prepared_dir and (
        args.force_ivf or not centroid_fvec.exists() or not cid_ivecs.exists()
    ):
        _run(
            [
                "python",
                "python/ivf.py",
                str(base_fvec),
                str(args.clusters),
                str(centroid_fvec),
                str(cid_ivecs),
                "l2",
            ],
            cwd=lib_root,
        )

    build_dir = lib_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        _run(["cmake", ".."], cwd=build_dir)
        _run(["cmake", "--build", ".", "-j"], cwd=build_dir)

    bin_dir = lib_root / "bin"
    index_bin = bin_dir / "ivf_rabitq_indexing"
    relerr_bin = bin_dir / "ivf_rabitq_relerr"
    if not index_bin.exists() or not relerr_bin.exists():
        raise FileNotFoundError("Missing binaries in lib/RaBitQ-Library/bin")

    out_dir = lib_root / "results" / "rabitqlib_relerr"
    out_dir.mkdir(parents=True, exist_ok=True)
    relerrs: dict[int, dict[str, str]] = {}
    times: dict[int, dict[str, float]] = {}

    for b in bits:
        index_path = ds_dir / f"ivf_rabitqlib_{b}.index"
        t0 = time.time()
        _run(
            [
                str(index_bin),
                str(base_fvec),
                str(centroid_fvec),
                str(cid_ivecs),
                str(b),
                str(index_path),
                "l2",
                "true" if args.faster_quant else "false",
            ],
            cwd=bin_dir,
        )
        encoding_time = time.time() - t0
        relerr_path = out_dir / f"{args.dataset}_rabitqlib_{b}_relerr.csv"
        t0 = time.time()
        _run(
            [
                str(relerr_bin),
                str(index_path),
                str(base_fvec),
                str(query_fvec),
                str(b),
                str(args.topk),
                str(args.nprobe),
                str(relerr_path),
            ],
            cwd=bin_dir,
        )
        adc_time = time.time() - t0
        relerrs[b] = _parse_relerr(relerr_path)
        times[b] = {"encoding_time_s": encoding_time, "adc_time_s": adc_time}

    out_csv = (
        Path(args.out_csv).resolve()
        if args.out_csv
        else get_results_root() / "rabitq" / f"{args.dataset}_RaBitQ-Library_adc_vs_exact_eval.csv"
    )
    _write_project_csv(
        out_csv=out_csv,
        dataset=args.dataset,
        experiment_folder=out_dir,
        base_path=base_path,
        nq=nq,
        nb=nb,
        dim=dim,
        bits=bits,
        relerrs=relerrs,
        times=times,
    )
    print(out_csv)


if __name__ == "__main__":
    main()
