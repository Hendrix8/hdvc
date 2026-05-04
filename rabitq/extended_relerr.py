#!/usr/bin/env python3
"""Extended-RaBitQ pipeline: build codes, compute distances, export relerr CSV.

This script automates:
1) dataset preparation under `lib/Extended-RaBitQ/data/<dataset>`
2) IVF training + cluster assignments
3) index construction (`create_index`) for each requested bitwidth
4) approximate-vs-exact distance relative error (`compute_relerr`)
5) export to project CSV schema consumed by plotting scripts
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

for _anc in Path(__file__).resolve().parents:
    if (_anc / "hdvc_paths.py").is_file():
        if str(_anc) not in sys.path:
            sys.path.insert(0, str(_anc))
        break
from hdvc_paths import get_results_root  # noqa: E402

import numpy as np


SUPPORTED_BITS = (2, 3, 4, 5, 7, 8, 9)
NATIVE_BITS = (3, 4, 5, 7, 8, 9)


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_ivecs(path: Path) -> np.ndarray:
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:]


def _write_ivecs(path: Path, x: np.ndarray) -> None:
    x = np.ascontiguousarray(x.astype(np.int32))
    n, d = x.shape
    with path.open("wb") as f:
        for i in range(n):
            np.array([d], dtype=np.int32).tofile(f)
            x[i].tofile(f)


def _read_bvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    dim = np.fromfile(path, dtype=np.int32, count=1)[0]
    stride = dim + 4
    n = raw.size // stride
    arr = raw[: n * stride].reshape(n, stride)[:, 4:]
    return arr.astype(np.float32)


def _read_fvecs(path: Path) -> np.ndarray:
    return _read_ivecs(path).view(np.float32)


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
    ext_root: Path,
    dataset: str,
    base_path: Path,
    query_path: Path,
    gt_ivecs_path: Path,
    vec_type: str,
    force: bool,
) -> tuple[Path, int, int, int]:
    ds_dir = ext_root / "data" / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    base = _load_vectors(base_path, vec_type)
    query = _load_vectors(query_path, vec_type)
    gt = _read_ivecs(gt_ivecs_path)

    base_out = ds_dir / f"{dataset}_base.fvecs"
    query_out = ds_dir / f"{dataset}_query.fvecs"
    gt_out = ds_dir / f"{dataset}_groundtruth.ivecs"

    if force or not base_out.exists():
        _write_fvecs(base_out, base)
    if force or not query_out.exists():
        _write_fvecs(query_out, query)
    if force or not gt_out.exists():
        shutil.copyfile(gt_ivecs_path, gt_out)

    return ds_dir, int(base.shape[0]), int(query.shape[0]), int(base.shape[1])


def _build_centroids_assignments(
    ds_dir: Path,
    dataset: str,
    clusters: int,
    force: bool,
) -> None:
    import faiss

    base_fvecs = ds_dir / f"{dataset}_base.fvecs"
    centroid_out = ds_dir / f"{dataset}_centroid_{clusters}.fvecs"
    cluster_id_out = ds_dir / f"{dataset}_cluster_id_{clusters}.ivecs"
    dist_out = ds_dir / f"{dataset}_dist_to_centroid_{clusters}.fvecs"

    if not force and centroid_out.exists() and cluster_id_out.exists() and dist_out.exists():
        return

    x = _read_fvecs(base_fvecs)
    dim = x.shape[1]

    index = faiss.index_factory(dim, f"IVF{clusters},Flat")
    index.verbose = True
    index.train(x)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    dist_to_centroid, cluster_id = index.quantizer.search(x, 1)

    _write_fvecs(centroid_out, centroids.astype(np.float32))
    _write_ivecs(cluster_id_out, cluster_id.astype(np.int32))
    _write_fvecs(dist_out, np.sqrt(dist_to_centroid.astype(np.float32)))


def _parse_relerr_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return row


def _export_project_csv(
    out_csv: Path,
    dataset: str,
    ext_root: Path,
    base_path: Path,
    nq: int,
    nb: int,
    dim: int,
    bits: list[int],
    relerr_rows: list[dict[str, str]],
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

    by_bits = {int(r["bits"]): r for r in relerr_rows}
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in sorted(bits):
            rr = by_bits[b]
            writer.writerow(
                {
                    "method": "RaBitQ",
                    "dataset": dataset,
                    "experiment_folder": str(ext_root / "results" / "exrabitq"),
                    "nq": nq,
                    "nb": nb,
                    "nb_sample": 100,
                    "dim": dim,
                    "n_subquantizers": 0,
                    "nbits": b,
                    "bits_per_vector": dim * b,
                    "train_size": nb,
                    "train_time_s": np.nan,
                    "encoding_time_s": np.nan,
                    "distance_table_time_s": np.nan,
                    "cdist_time_s": np.nan,
                    "adc_time_s": np.nan,
                    "rel_error_mean": float(rr["rel_error_mean"]),
                    "rel_error_std": float(rr["rel_error_std"]),
                    "nq_sample": nq,
                    "sample_mode": f"extended_relerr_topk={rr['topk']}_nprobe={rr['nprobe']}",
                    "train_path": str(base_path),
                    "seed": np.nan,
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Extended-RaBitQ codes + distance relerr pipeline.")
    p.add_argument("--ext_root", default="/home/cpanourg/projects/2-hdvc/lib/Extended-RaBitQ")
    p.add_argument("--dataset", required=True)
    p.add_argument("--base", required=True, help="Input base vectors (.bvecs or .fvecs)")
    p.add_argument("--query", required=True, help="Input query vectors (.bvecs or .fvecs)")
    p.add_argument("--groundtruth", required=True, help="Input gt .ivecs")
    p.add_argument("--vec_type", choices=["bvecs", "fvecs"], default="bvecs")
    p.add_argument("--clusters", type=int, default=4096)
    p.add_argument("--bits", type=int, nargs="+", default=list(SUPPORTED_BITS))
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--nprobe", type=int, default=10)
    p.add_argument("--skip_build", action="store_true")
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--force_ivf", action="store_true")
    p.add_argument(
        "--export_only",
        action="store_true",
        help="Skip index/recompute and export from existing *_relerr.csv files.",
    )
    p.add_argument(
        "--project_out_csv",
        default=None,
        help="Output CSV in project schema (default: {results}/rabitq/<dataset>_RaBitQ_adc_vs_exact_eval.csv)",
    )
    args = p.parse_args()

    bits = sorted(set(args.bits))
    for b in bits:
        if b not in SUPPORTED_BITS:
            raise ValueError(f"Unsupported bits {b}; supported: {SUPPORTED_BITS}")

    ext_root = Path(args.ext_root).resolve()
    base_path = Path(args.base).resolve()
    query_path = Path(args.query).resolve()
    gt_path = Path(args.groundtruth).resolve()

    if args.export_only:
        base = _load_vectors(base_path, args.vec_type)
        query = _load_vectors(query_path, args.vec_type)
        nb, nq, dim = int(base.shape[0]), int(query.shape[0]), int(base.shape[1])
    else:
        ds_dir, nb, nq, dim = _prepare_dataset(
            ext_root=ext_root,
            dataset=args.dataset,
            base_path=base_path,
            query_path=query_path,
            gt_ivecs_path=gt_path,
            vec_type=args.vec_type,
            force=args.force_prepare,
        )
        _build_centroids_assignments(
            ds_dir=ds_dir,
            dataset=args.dataset,
            clusters=args.clusters,
            force=args.force_ivf,
        )

        build_dir = ext_root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_build:
            _run(["cmake", ".."], cwd=build_dir)
            _run(["cmake", "--build", ".", "-j"], cwd=build_dir)

        bin_dir = ext_root / "bin"
        create_index = bin_dir / "create_index"
        compute_relerr = bin_dir / "compute_relerr"
        if not create_index.exists() or not compute_relerr.exists():
            raise FileNotFoundError("Expected binaries missing in Extended-RaBitQ/bin.")

    relerr_rows: list[dict[str, str]] = []
    for b in bits:
        run_b = b if b in NATIVE_BITS else 3
        if b != run_b:
            print(
                f"[extended_relerr] requested bits={b} is not native in Extended-RaBitQ; "
                f"running bits={run_b} as nearest supported setting."
            )
        if not args.export_only:
            _run([str(create_index), args.dataset, str(args.clusters), str(run_b)], cwd=bin_dir)
            _run(
                [str(compute_relerr), args.dataset, str(run_b), str(args.topk), str(args.nprobe)],
                cwd=bin_dir,
            )
        relerr_path = ext_root / "results" / "exrabitq" / f"{args.dataset}_exhaf{run_b}_relerr.csv"
        if not relerr_path.exists():
            raise FileNotFoundError(f"Missing relerr csv: {relerr_path}")
        rr = _parse_relerr_row(relerr_path)
        rr["bits"] = str(b)
        relerr_rows.append(rr)

    out_csv = (
        Path(args.project_out_csv).resolve()
        if args.project_out_csv
        else get_results_root() / "rabitq" / f"{args.dataset}_RaBitQ_adc_vs_exact_eval.csv"
    )
    _export_project_csv(
        out_csv=out_csv,
        dataset=args.dataset,
        ext_root=ext_root,
        base_path=base_path,
        nq=nq,
        nb=nb,
        dim=dim,
        bits=bits,
        relerr_rows=relerr_rows,
    )
    print(out_csv)


if __name__ == "__main__":
    main()
