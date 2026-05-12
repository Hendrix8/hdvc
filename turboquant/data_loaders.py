"""
Loaders for the ANN benchmark datasets under /data/cpanourg/2-hdvc/data/.

Supports three file formats:
  - .fvecs   per-row layout [dim:int32, dim*float32]
  - .bin     header [n:int32, d:int32], then n*d float32
  - .u8bin   header [n:int32, d:int32], then n*d uint8

All loaders accept an optional ``count`` to read only the first N vectors.
"""

from pathlib import Path

import numpy as np

DATA_ROOT = Path("/data/cpanourg/2-hdvc/data")


def read_fvecs(filename, count=None):
    fv = np.memmap(filename, dtype="float32", mode="r")
    if fv.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = int(fv.view(np.int32)[0])
    assert dim > 0
    n_total = fv.size // (dim + 1)
    if count is None or count > n_total:
        count = n_total
    head = np.memmap(filename, dtype="float32", mode="r",
                     shape=(count * (dim + 1),))
    arr = head.reshape(count, dim + 1)[:, 1:].copy()
    return np.ascontiguousarray(arr)


def read_fbin(filename, count=None):
    head = np.fromfile(filename, dtype=np.int32, count=2)
    n, d = int(head[0]), int(head[1])
    if count is None or count > n:
        count = n
    arr = np.memmap(filename, dtype=np.float32, mode="r",
                    offset=8, shape=(count * d,))
    return np.ascontiguousarray(arr.reshape(count, d).copy())


def read_u8bin(filename, count=None):
    head = np.fromfile(filename, dtype=np.int32, count=2)
    n, d = int(head[0]), int(head[1])
    if count is None or count > n:
        count = n
    arr = np.memmap(filename, dtype=np.uint8, mode="r",
                    offset=8, shape=(count * d,))
    return np.ascontiguousarray(arr.reshape(count, d).astype(np.float32))


def read_raw_f32(filename, dim, count=None):
    """Read a header-less raw float32 file with known ``dim``.

    Used for the Deep1B-Skoltech ``.bin`` files which are simply n*d float32
    laid out row-major without any header.
    """
    size = Path(filename).stat().st_size
    n_total = size // (dim * 4)
    if count is None or count > n_total:
        count = n_total
    arr = np.memmap(filename, dtype=np.float32, mode="r",
                    shape=(count * dim,))
    return np.ascontiguousarray(arr.reshape(count, dim).copy())


# ----------------------------------------------------------------------------
# Dataset registry. Each entry has paths for learn/base/query and a reader id.
# Note: TurboQuant-MSE is data-oblivious, so the learn set is unused; we still
# record the path in case future variants need it.
# ----------------------------------------------------------------------------

DATASET_PATHS = {
    "deep": {  # deep1b 96-dim, z-normalized, raw float32 (no header)
        "learn": DATA_ROOT / "deep1b/dataset/learn_100m.bin",
        "base":  DATA_ROOT / "deep1b/dataset/test_1m.bin",
        "query": DATA_ROOT / "deep1b/dataset/query_10k.bin",
        "reader": "raw_f32",
        "dim": 96,
    },
    "bigann": {  # 128-dim uint8 (SIFT-like)
        "learn": DATA_ROOT / "bigann/base.1B.u8bin",
        "base":  DATA_ROOT / "bigann/base.1B.u8bin",
        "query": DATA_ROOT / "bigann/query.public.10K.u8bin",
        "reader": "u8bin",
    },
    "gist": {  # 960-dim float32
        "learn": DATA_ROOT / "gist/gist_learn.fvecs",
        "base":  DATA_ROOT / "gist/gist_base.fvecs",
        "query": DATA_ROOT / "gist/gist_query.fvecs",
        "reader": "fvecs",
    },
    "msmarco": {  # float32 .fvecs
        "learn": DATA_ROOT / "msmarco/train1m.fvecs",
        "base":  DATA_ROOT / "msmarco/base1m.fvecs",
        "query": DATA_ROOT / "msmarco/query10k.fvecs",
        "reader": "fvecs",
    },
    "openai": {  # 1536-dim float32
        "learn": DATA_ROOT / "openai/openai_train1m.fvecs",
        "base":  DATA_ROOT / "openai/openai_base1m.fvecs",
        "query": DATA_ROOT / "openai/openai_query10k.fvecs",
        "reader": "fvecs",
    },
}

READERS = {"fvecs": read_fvecs, "fbin": read_fbin, "u8bin": read_u8bin,
           "raw_f32": read_raw_f32}


def _read(paths_entry, key, count):
    reader_name = paths_entry["reader"]
    reader = READERS[reader_name]
    if reader_name == "raw_f32":
        return reader(paths_entry[key], dim=paths_entry["dim"], count=count)
    return reader(paths_entry[key], count=count)


def load_eval_set(name, n_base=10_000, n_query=1_000):
    """Load (X_base, X_query) for evaluation. Returns float32 ndarrays."""
    if name not in DATASET_PATHS:
        raise ValueError(f"unknown dataset: {name!r}; available: {list(DATASET_PATHS)}")
    paths = DATASET_PATHS[name]
    X_base = _read(paths, "base", n_base)
    X_query = _read(paths, "query", n_query)
    if X_base.shape[1] != X_query.shape[1]:
        raise RuntimeError(
            f"dim mismatch for {name}: base={X_base.shape}, query={X_query.shape}"
        )
    return X_base.astype(np.float32, copy=False), X_query.astype(np.float32, copy=False)


def load_learn_set(name, n_learn=1_000_000):
    """Optional: load the learn set (unused by TurboQuant-MSE itself)."""
    paths = DATASET_PATHS[name]
    arr = _read(paths, "learn", n_learn)
    return arr.astype(np.float32, copy=False)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "gist"
    base, query = load_eval_set(name, n_base=100, n_query=10)
    print(f"[{name}] base={base.shape} dtype={base.dtype}  query={query.shape}")
    print(f"  base norm range: [{np.linalg.norm(base, axis=1).min():.3f}, "
          f"{np.linalg.norm(base, axis=1).max():.3f}]")
    print(f"  base[0, :5] = {base[0, :5]}")
