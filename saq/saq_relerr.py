#!/usr/bin/env python3
"""Run lib/SAQ and export native approximate-distance relerr to project CSV.

SAQ's C++ `test_relative_error` computes estimated distances through
`IVF::estimate`, which calls the SAQ/CAQ accurate compressed-domain estimator.
This wrapper prepares the file layout that lib/SAQ expects, runs the native
binaries, and converts their aggregate CSV into the schema used by the plotting
scripts in this repository.
"""

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


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), check=True, text=True)


def _conda_cmake_prefix_path() -> str:
    roots: list[Path] = []
    roots.append(Path(sys.prefix))
    if "CONDA_PREFIX" in os.environ:
        roots.append(Path(os.environ["CONDA_PREFIX"]))
    roots.append(Path.home() / ".miniconda3")
    roots.append(Path.home() / ".conda")
    roots.append(Path.home() / ".local" / "saq-deps")

    prefixes: list[Path] = []
    package_globs = ("glog-*", "gflags-*", "fmt-*")
    for root in roots:
        if (root / "lib" / "cmake").exists():
            prefixes.append(root)
        pkgs = root / "pkgs"
        if pkgs.exists():
            for pat in package_globs:
                prefixes.extend(sorted(p for p in pkgs.glob(pat) if (p / "lib" / "cmake").exists()))

    seen: set[str] = set()
    uniq: list[str] = []
    for p in prefixes:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return ";".join(uniq)


def _read_ivecs(path: Path, max_vectors: int | None = None) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=1)
    if header.size == 0:
        raise ValueError(f"Empty ivecs/fvecs file: {path}")
    d = int(header[0])
    row = d + 1
    count = -1 if max_vectors is None else max_vectors * row
    a = np.fromfile(path, dtype=np.int32, count=count)
    if a.size == 0:
        raise ValueError(f"Empty ivecs/fvecs file: {path}")
    return a.reshape(-1, d + 1)[:, 1:]


def _read_fvecs(path: Path, max_vectors: int | None = None) -> np.ndarray:
    return _read_ivecs(path, max_vectors=max_vectors).view(np.float32)


def _read_bvecs(path: Path, max_vectors: int | None = None) -> np.ndarray:
    dim = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    stride = dim + 4
    count = -1 if max_vectors is None else max_vectors * stride
    raw = np.fromfile(path, dtype=np.uint8, count=count)
    n = raw.size // stride
    return raw[: n * stride].reshape(n, stride)[:, 4:].astype(np.float32)


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


def _load_vectors(path: Path, vec_type: str, max_vectors: int | None = None) -> np.ndarray:
    if vec_type == "bvecs":
        return _read_bvecs(path, max_vectors=max_vectors)
    if vec_type == "fvecs":
        return _read_fvecs(path, max_vectors=max_vectors)
    raise ValueError(f"Unsupported vec_type: {vec_type}")


def _prepare_raw_files(
    saq_root: Path,
    dataset: str,
    base_path: Path,
    query_path: Path,
    gt_ivecs_path: Path | None,
    vec_type: str,
    force: bool,
    max_base: int | None,
    max_query: int | None,
) -> tuple[Path, int, int, int]:
    ds_dir = saq_root / "data" / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    base = _load_vectors(base_path, vec_type, max_vectors=max_base)
    query = _load_vectors(query_path, vec_type, max_vectors=max_query)

    base_out = ds_dir / f"{dataset}_base.fvecs"
    query_out = ds_dir / f"{dataset}_query.fvecs"
    gt_out = ds_dir / f"{dataset}_groundtruth.ivecs"

    if force or not base_out.exists():
        _write_fvecs(base_out, base)
    if force or not query_out.exists():
        _write_fvecs(query_out, query)
    if gt_ivecs_path is not None and (force or not gt_out.exists()):
        shutil.copyfile(gt_ivecs_path, gt_out)

    return ds_dir, int(base.shape[0]), int(query.shape[0]), int(base.shape[1])


def _build_ivf_and_pca(ds_dir: Path, dataset: str, clusters: int, force: bool, pca_train_size: int | None) -> float:
    import faiss

    base_fvec = ds_dir / f"{dataset}_base.fvecs"
    query_fvec = ds_dir / f"{dataset}_query.fvecs"
    centroid_fvec = ds_dir / f"{dataset}_centroid_{clusters}.fvecs"
    cid_ivecs = ds_dir / f"{dataset}_cluster_id_{clusters}.ivecs"
    base_pca_fvec = ds_dir / f"{dataset}_base_pca.fvecs"
    query_pca_fvec = ds_dir / f"{dataset}_query_pca.fvecs"
    centroid_pca_fvec = ds_dir / f"{dataset}_centroid_{clusters}_pca.fvecs"
    vars_fvec = ds_dir / f"{dataset}_base_pca.vars.fvecs"

    outputs = [centroid_fvec, cid_ivecs, base_pca_fvec, query_pca_fvec, centroid_pca_fvec, vars_fvec]
    if not force and all(p.exists() for p in outputs):
        return 0.0

    t0 = time.time()
    base = _read_fvecs(base_fvec)
    query = _read_fvecs(query_fvec)
    dim = base.shape[1]

    index = faiss.index_factory(dim, f"IVF{clusters},Flat")
    index.verbose = True
    index.train(base)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    _, cluster_id = index.quantizer.search(base, 1)

    _write_fvecs(centroid_fvec, centroids.astype(np.float32))
    _write_ivecs(cid_ivecs, cluster_id.astype(np.int32))

    pca_train = base if pca_train_size is None else base[: min(pca_train_size, len(base))]
    pca = faiss.PCAMatrix(dim, dim)
    pca.train(pca_train)
    base_pca = pca.apply(base)
    query_pca = pca.apply(query)
    centroid_pca = pca.apply(centroids.astype(np.float32))

    _write_fvecs(base_pca_fvec, base_pca)
    _write_fvecs(query_pca_fvec, query_pca)
    _write_fvecs(centroid_pca_fvec, centroid_pca)
    _write_fvecs(vars_fvec, np.var(base_pca, axis=0, keepdims=True).astype(np.float32))
    return time.time() - t0


def _read_first_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="") as f:
        return next(csv.DictReader(f))


def _latest_matching_csv(results_dir: Path, dataset: str, clusters: int, bits: float, bound_m: float) -> Path:
    bit_token = f"{bits:g}"
    bound_token = f"{bound_m:g}"
    patterns = [
        f"{dataset}_ivf{clusters}_b{bit_token}_caq*_sm{bound_token}.csv",
        f"{dataset}_ivf{clusters}_b{bits}_caq*_sm{bound_m}.csv",
        f"{dataset}_ivf{clusters}_b*_caq*_sm*.csv",
    ]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(results_dir.glob(pat))
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"No SAQ relerr CSV found in {results_dir} for bits={bits}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_matching_index_csv(results_dir: Path, dataset: str, clusters: int, bits: float) -> Path:
    bit_token = f"{bits:g}"
    patterns = [
        f"{dataset}_ivf{clusters}_b{bit_token}_caq*.index.csv",
        f"{dataset}_ivf{clusters}_b{bits}_caq*.index.csv",
        f"{dataset}_ivf{clusters}_b*_caq*.index.csv",
    ]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(results_dir.glob(pat))
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"No SAQ index CSV found in {results_dir} for bits={bits}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _write_project_csv(
    out_csv: Path,
    dataset: str,
    experiment_folder: Path,
    base_path: Path,
    nq: int,
    nb: int,
    dim: int,
    rows: list[dict[str, object]],
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
        "relerr_min_real_dist",
        "relerr_count",
        "relerr_skipped_small_dist_count",
        "nq_sample",
        "sample_mode",
        "train_path",
        "seed",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "dataset": dataset,
                    "experiment_folder": str(experiment_folder),
                    "nq": nq,
                    "nb": nb,
                    "nb_sample": row["nb_sample"],
                    "dim": dim,
                    "n_subquantizers": row["n_subquantizers"],
                    "nbits": row["nbits"],
                    "bits_per_vector": row["bits_per_vector"],
                    "train_size": nb,
                    "train_time_s": row["train_time_s"],
                    "encoding_time_s": row["encoding_time_s"],
                    "distance_table_time_s": np.nan,
                    "cdist_time_s": np.nan,
                    "adc_time_s": row["adc_time_s"],
                    "rel_error_mean": row["rel_error_mean"],
                    "rel_error_std": row["rel_error_std"],
                    "relerr_min_real_dist": row["relerr_min_real_dist"],
                    "relerr_count": row["relerr_count"],
                    "relerr_skipped_small_dist_count": row["relerr_skipped_small_dist_count"],
                    "nq_sample": nq,
                    "sample_mode": row["sample_mode"],
                    "train_path": str(base_path),
                    "seed": np.nan,
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Run SAQ and export native approximate-distance relerr CSV.")
    p.add_argument("--saq_root", default="/home/cpanourg/projects/2-hdvc/lib/SAQ")
    p.add_argument("--dataset", required=True)
    p.add_argument("--base", required=True, help="Input base vectors (.bvecs or .fvecs)")
    p.add_argument("--query", required=True, help="Input query vectors (.bvecs or .fvecs)")
    p.add_argument("--groundtruth", default=None, help="Optional input gt .ivecs copied into lib/SAQ/data")
    p.add_argument("--vec_type", choices=["bvecs", "fvecs"], default="bvecs")
    p.add_argument("--clusters", type=int, default=4096)
    p.add_argument("--bits", type=float, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num_threads", type=int, default=0)
    p.add_argument("--searcher_vars_bound_m", type=float, default=4)
    p.add_argument(
        "--relerr_min_real_dist",
        type=float,
        default=1e-6,
        help="Skip exact distances <= this threshold when computing relative error.",
    )
    p.add_argument("--skip_build", action="store_true")
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--force_ivf_pca", action="store_true")
    p.add_argument("--no_segmentation", action="store_true", help="Run CAQ only by disabling SAQ segmentation.")
    p.add_argument("--rand_rotate", choices=["true", "false"], default="true")
    p.add_argument("--use_fastscan", choices=["true", "false"], default="true")
    p.add_argument("--caq_adj_rd_lmt", type=int, default=6)
    p.add_argument("--caq_adj_eps", type=float, default=1e-8)
    p.add_argument("--max_base", type=int, default=None)
    p.add_argument("--max_query", type=int, default=None)
    p.add_argument(
        "--pca_train_size",
        type=int,
        default=None,
        help="Train PCA on the first N prepared base vectors, then transform all prepared vectors.",
    )
    p.add_argument(
        "--project_out_csv",
        default=None,
        help="Output CSV path; default {results}/saq/<dataset>_SAQ_adc_vs_exact_eval.csv",
    )
    args = p.parse_args()

    saq_root = Path(args.saq_root).resolve()
    base_path = Path(args.base).resolve()
    query_path = Path(args.query).resolve()
    gt_path = Path(args.groundtruth).resolve() if args.groundtruth else None

    ds_dir, nb, nq, dim = _prepare_raw_files(
        saq_root,
        args.dataset,
        base_path,
        query_path,
        gt_path,
        args.vec_type,
        args.force_prepare,
        args.max_base,
        args.max_query,
    )
    prep_time = _build_ivf_and_pca(ds_dir, args.dataset, args.clusters, args.force_ivf_pca, args.pca_train_size)

    build_dir = saq_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        cmake_configure = ["cmake", "-DBUILD_UNIT_TESTS=OFF"]
        conda_prefix_path = _conda_cmake_prefix_path()
        if conda_prefix_path:
            cmake_configure.append(f"-DCMAKE_PREFIX_PATH={conda_prefix_path}")
        cmake_configure.append("..")
        _run(cmake_configure, cwd=build_dir)
        _run(["cmake", "--build", ".", "-j"], cwd=build_dir)

    create_index = saq_root / "bin" / "create_index"
    test_relative_error = saq_root / "bin" / "test_relative_error"
    if not create_index.exists() or not test_relative_error.exists():
        raise FileNotFoundError("Expected SAQ binaries missing in lib/SAQ/bin.")

    saq_results_dir = saq_root / "results" / "saq"
    saq_results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    method = "CAQ" if args.no_segmentation else "SAQ"

    for bits in sorted(set(float(b) for b in args.bits)):
        common_flags = [
            f"-dataset={args.dataset}",
            f"-K={args.clusters}",
            f"-B={bits:g}",
            "-enable_PCA=true",
            f"-enable_segmentation={'false' if args.no_segmentation else 'true'}",
            f"-rand_rotate={args.rand_rotate}",
            f"-use_fastscan={args.use_fastscan}",
            f"-caq_adj_rd_lmt={args.caq_adj_rd_lmt}",
            f"-caq_adj_eps={args.caq_adj_eps}",
            f"-searcher_vars_bound_m={args.searcher_vars_bound_m:g}",
        ]
        if args.num_threads:
            common_flags.append(f"-num_threads={args.num_threads}")

        t0 = time.time()
        _run([str(create_index), *common_flags], cwd=saq_root)
        encoding_time = time.time() - t0
        index_csv = _latest_matching_index_csv(saq_results_dir, args.dataset, args.clusters, bits)
        try:
            encoding_time = float(_read_first_csv_row(index_csv).get("index_time_s", encoding_time))
        except Exception:
            pass

        relerr_flags = [*common_flags, f"-relerr_min_real_dist={args.relerr_min_real_dist:g}"]
        t0 = time.time()
        _run([str(test_relative_error), *relerr_flags], cwd=saq_root)
        adc_time = time.time() - t0
        relerr_csv = _latest_matching_csv(saq_results_dir, args.dataset, args.clusters, bits, args.searcher_vars_bound_m)
        rr = _read_first_csv_row(relerr_csv)

        rows.append(
            {
                "method": method,
                "nb_sample": nb,
                "n_subquantizers": 0,
                "nbits": bits,
                "bits_per_vector": dim * bits,
                "train_time_s": prep_time,
                "encoding_time_s": encoding_time,
                "adc_time_s": adc_time,
                "rel_error_mean": float(rr["err_tot_avg"]),
                "rel_error_std": float(rr.get("err_tot_std", "nan")),
                "relerr_min_real_dist": float(rr.get("relerr_min_real_dist", args.relerr_min_real_dist)),
                "relerr_count": int(float(rr.get("relerr_count", "0"))),
                "relerr_skipped_small_dist_count": int(float(rr.get("skipped_small_dist_count", "0"))),
                "sample_mode": f"saq_native_nprobe={rr.get('nprobe', '')}",
                "use_fastscan": args.use_fastscan == "true",
            }
        )

    out_csv = (
        Path(args.project_out_csv).resolve()
        if args.project_out_csv
        else get_results_root() / "saq" / f"{args.dataset}_{method}_adc_vs_exact_eval.csv"
    )
    _write_project_csv(out_csv, args.dataset, saq_results_dir, base_path, nq, nb, dim, rows)
    print(out_csv)


if __name__ == "__main__":
    main()
