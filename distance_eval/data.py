"""Load database + query vectors (fvecs / fbin / bvecs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Project root on sys.path when running as script
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def read_bvecs(path: Path, max_vectors: int | None = None) -> np.ndarray:
    """Read .bvecs (SIFT/BIGANN): per vector, 4-byte dim + dim bytes uint8."""
    with open(path, "rb") as f:
        dim_b = f.read(4)
        if len(dim_b) < 4:
            return np.zeros((0, 0), dtype=np.float32)
        dim = int(np.frombuffer(dim_b, dtype=np.int32)[0])
        f.seek(0)
        data = []
        n_read = 0
        vec_size = 4 + dim
        while True:
            if max_vectors is not None and n_read >= max_vectors:
                break
            chunk = f.read(vec_size * 10000)
            if len(chunk) < vec_size:
                break
            n = len(chunk) // vec_size
            arr = np.frombuffer(chunk[: n * vec_size], dtype=np.uint8)
            arr = arr.reshape(n, vec_size)[:, 4:].astype(np.float32)
            data.append(arr)
            n_read += n
        if not data:
            return np.zeros((0, dim), dtype=np.float32)
        out = np.vstack(data)
        if max_vectors is not None and len(out) > max_vectors:
            out = out[:max_vectors]
        return out


def load_db_and_queries(
    dataset_path: str,
    query_path: str,
    dim: int,
    max_db: int = 1_000_000,
    max_queries: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Load database and query vectors. Supports .fvecs, .fbin, .bvecs."""
    dataset_path = Path(dataset_path)
    query_path = Path(query_path)

    if dataset_path.suffix == ".bvecs":
        db = read_bvecs(dataset_path, max_vectors=max_db)
        qr = read_bvecs(query_path, max_vectors=max_queries)
    else:
        # Lazy import: src.utils pulls faiss at module level
        from src.utils import load_dataset  # noqa: WPS433

        db, qr = load_dataset(
            str(dataset_path),
            str(query_path),
            dim=dim,
            db_chunk_size=max_db,
            qr_chunk_size=max_queries,
        )
    return db.astype(np.float32), qr.astype(np.float32)
