"""Dataset I/O for LSQ++ eval (.bin / .fvecs / .fbin / .bvecs; numpy only)."""

from pathlib import Path

import numpy as np


def bin_num_vectors(path: str, dim: int) -> int:
    nbytes = Path(path).stat().st_size
    return nbytes // (dim * 4)


def read_bin_prefix(
    path: str,
    dim: int,
    max_rows: int | None = None,
    start_row: int = 0,
) -> np.ndarray:
    n_total = bin_num_vectors(path, dim)
    if start_row >= n_total:
        return np.zeros((0, dim), dtype=np.float32)
    n_avail = n_total - start_row
    n_read = n_avail if max_rows is None else min(max_rows, n_avail)
    if n_read <= 0:
        return np.zeros((0, dim), dtype=np.float32)
    row_bytes = dim * 4
    offset = start_row * row_bytes
    mm = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        offset=offset,
        shape=(n_read, dim),
    )
    return np.asarray(mm, dtype=np.float32)


def read_fvecs(filename: str, c_contiguous: bool = True) -> np.ndarray:
    print(f"Reading File - {filename}", end=":")
    fv = np.memmap(filename, dtype="float32", mode="r")
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    print(fv.shape)
    return fv


def read_fbin(filename: str, start_idx: int = 0, chunk_size: int | None = None) -> np.ndarray:
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(
            f, count=nvecs * dim, dtype=np.float32, offset=start_idx * 4 * dim
        )
    return arr.reshape(nvecs, dim)


def load_dataset(
    dataset_path: str,
    query_path: str | None = None,
    dim: int | None = None,
    start_idx: int = 0,
    db_chunk_size: int | None = None,
    qr_chunk_size: int | None = None,
):
    if dataset_path.endswith(".fvecs"):
        db = np.array(read_fvecs(dataset_path))
    elif dataset_path.endswith(".bin"):
        if dim is None:
            raise ValueError("dim parameter is required for .bin files")
        db = np.fromfile(dataset_path, dtype=np.float32).reshape(-1, dim)
    elif dataset_path.endswith(".fbin"):
        db = read_fbin(dataset_path, start_idx=start_idx, chunk_size=db_chunk_size)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    if query_path:
        if query_path.endswith(".fvecs"):
            qr = np.array(read_fvecs(query_path))
        elif query_path.endswith(".bin"):
            if dim is None:
                raise ValueError("dim parameter is required for .bin files")
            qr = np.fromfile(query_path, dtype=np.float32).reshape(-1, dim)
        elif query_path.endswith(".fbin"):
            qr = read_fbin(query_path, start_idx=start_idx, chunk_size=qr_chunk_size)
        else:
            raise ValueError(f"Unsupported query format: {query_path}")
    else:
        qr = db.copy()

    return db.astype(np.float32), qr.astype(np.float32)


def read_bvecs_prefix(
    path: str,
    max_rows: int | None = None,
    start_row: int = 0,
) -> np.ndarray:
    with open(path, "rb") as f:
        dim = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    vec_size = 4 + dim
    nbytes = Path(path).stat().st_size
    n_total = nbytes // vec_size
    if start_row >= n_total:
        return np.zeros((0, dim), dtype=np.float32)
    n_avail = n_total - start_row
    n_read = n_avail if max_rows is None else min(max_rows, n_avail)
    if n_read <= 0:
        return np.zeros((0, dim), dtype=np.float32)
    offset = start_row * vec_size
    mm = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=offset,
        shape=(n_read, vec_size),
    )
    return np.asarray(mm[:, 4:], dtype=np.float32)


def infer_dim_from_path(path: str) -> int:
    if path.endswith(".bin"):
        raise ValueError(
            "Cannot infer dim for .bin files; pass --dim explicitly."
        )
    if path.endswith(".fvecs"):
        t = read_fvecs_prefix(path, 1)
        if t.shape[0] == 0:
            raise ValueError(f"Empty or invalid fvecs: {path}")
        return int(t.shape[1])
    if path.endswith(".bvecs"):
        with open(path, "rb") as f:
            return int(np.frombuffer(f.read(4), dtype=np.int32)[0])
    if path.endswith(".fbin"):
        with open(path, "rb") as f:
            _, d = np.fromfile(f, count=2, dtype=np.int32)
            return int(d)
    raise ValueError(f"Unsupported format for dim inference: {path}")


def read_fvecs_prefix(
    filename: str, max_rows: int | None = None, start_row: int = 0
) -> np.ndarray:
    fv = np.memmap(filename, dtype=np.float32, mode="r")
    if fv.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = int(fv.view(np.int32)[0])
    stride = dim + 1
    n_total = fv.size // stride
    if start_row >= n_total:
        return np.zeros((0, dim), dtype=np.float32)
    n_avail = n_total - start_row
    n_read = n_avail if max_rows is None else min(max_rows, n_avail)
    if n_read <= 0:
        return np.zeros((0, dim), dtype=np.float32)
    off = start_row * stride
    block = fv[off : off + n_read * stride].reshape(n_read, stride)
    return np.asarray(block[:, 1:], dtype=np.float32)


def load_vectors_prefix(
    path: str,
    dim: int | None,
    max_rows: int | None,
    start_row: int = 0,
) -> np.ndarray:
    if path.endswith(".bin"):
        if dim is None:
            raise ValueError("dim is required for .bin")
        return read_bin_prefix(path, dim, max_rows, start_row)
    if path.endswith(".fvecs"):
        return read_fvecs_prefix(path, max_rows, start_row)
    if path.endswith(".bvecs"):
        return read_bvecs_prefix(path, max_rows, start_row)
    if path.endswith(".fbin"):
        return read_fbin(path, start_idx=start_row, chunk_size=max_rows)
    raise ValueError(f"Unsupported format: {path}")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
