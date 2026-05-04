#!/usr/bin/env python3
"""Run Extended-RaBitQ (lib/Extended-RaBitQ) for BIGANN/SIFT1M only."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np


def _read_bvecs(path: Path, max_rows: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    dim = np.fromfile(path, dtype=np.int32, count=1)[0]
    stride = dim + 4
    n = raw.size // stride
    if max_rows is not None:
        n = min(n, max_rows)
    arr = raw[: n * stride].reshape(n, stride)[:, 4:]
    return arr.astype(np.float32)


def _write_fvecs(path: Path, x: np.ndarray) -> None:
    x = np.ascontiguousarray(x.astype(np.float32))
    n, d = x.shape
    with open(path, "wb") as f:
        for i in range(n):
            np.array([d], dtype=np.int32).tofile(f)
            x[i].view(np.int32).tofile(f)


def _prepare_bigann_dataset(ext_root: Path, base_bvecs: Path, query_bvecs: Path, gt_ivecs: Path) -> Path:
    ds_dir = ext_root / "data" / "bigann"
    ds_dir.mkdir(parents=True, exist_ok=True)
    base_fvecs = ds_dir / "bigann_base.fvecs"
    query_fvecs = ds_dir / "bigann_query.fvecs"
    gt_out = ds_dir / "bigann_groundtruth.ivecs"

    if not base_fvecs.exists():
        _write_fvecs(base_fvecs, _read_bvecs(base_bvecs, max_rows=1_000_000))
    if not query_fvecs.exists():
        _write_fvecs(query_fvecs, _read_bvecs(query_bvecs))
    if not gt_out.exists():
        shutil.copyfile(gt_ivecs, gt_out)
    return ds_dir


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Extended-RaBitQ BIGANN runner (no reference backend).")
    p.add_argument("--ext_root", type=str, default="/home/cpanourg/projects/2-hdvc/lib/Extended-RaBitQ")
    p.add_argument("--base_bvecs", type=str, default="/data/cpanourg/2-hdvc/data/bigann/SIFT1M/bigann_base.bvecs")
    p.add_argument("--query_bvecs", type=str, default="/data/cpanourg/2-hdvc/data/bigann/SIFT1M/bigann_query.bvecs")
    p.add_argument("--gt_ivecs", type=str, default="/data/cpanourg/2-hdvc/data/sift/sift1m/sift_groundtruth.ivecs")
    p.add_argument("--clusters", type=int, default=4096)
    p.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 7, 8, 9])
    p.add_argument("--skip_build", action="store_true")
    args = p.parse_args()

    ext_root = Path(args.ext_root).resolve()
    if not ext_root.exists():
        raise FileNotFoundError(f"Extended-RaBitQ root not found: {ext_root}")

    for b in args.bits:
        if b not in (3, 4, 5, 7, 8, 9):
            raise ValueError("Extended-RaBitQ in this repo supports bits in {3,4,5,7,8,9}.")

    _prepare_bigann_dataset(
        ext_root=ext_root,
        base_bvecs=Path(args.base_bvecs),
        query_bvecs=Path(args.query_bvecs),
        gt_ivecs=Path(args.gt_ivecs),
    )

    if not args.skip_build:
        build_dir = ext_root / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        _run(["cmake", ".."], cwd=build_dir)
        _run(["cmake", "--build", ".", "-j"], cwd=build_dir)

    # Build IVF centroids + assignments with the project helper.
    # Keep this command simple and explicit.
    ivf_py = ext_root / "python" / "ivf.py"
    text = ivf_py.read_text()
    text = text.replace('DATASET = "openai1536"', 'DATASET = "bigann"')
    text = text.replace("K = 4096", f"K = {args.clusters}")
    ivf_py.write_text(text)
    _run(["python", "python/ivf.py"], cwd=ext_root)

    bin_dir = ext_root / "bin"
    create_index = bin_dir / "create_index"
    test_search = bin_dir / "test_search"
    if not create_index.exists() or not test_search.exists():
        raise FileNotFoundError("Extended-RaBitQ binaries not found in bin/.")

    for b in args.bits:
        _run([str(create_index), "bigann", str(args.clusters), str(b)], cwd=bin_dir)
        _run([str(test_search), "bigann", str(b)], cwd=bin_dir)
        print(ext_root / "results" / "exrabitq" / f"bigann_exhaf{b}.csv")


if __name__ == "__main__":
    main()
