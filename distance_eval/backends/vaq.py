"""VAQ: per-subspace IP LUT + int16 centroid indices (same geometry as vaq/eval.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit, prange

from distance_eval.base import ADCBackend


def _read_vaq_codes_centroids(root: Path) -> tuple[np.ndarray, list[np.ndarray], int]:
    codes_fp = root / "codes.fvecs"
    centroids_fp = root / "centroids.fvecs"
    if not codes_fp.is_file():
        raise FileNotFoundError(f"Missing {codes_fp}")
    if not centroids_fp.is_file():
        raise FileNotFoundError(f"Missing {centroids_fp}")

    with open(codes_fp, "rb") as f:
        nrows = int(np.fromfile(f, dtype=np.int64, count=1)[0])
        ncols = int(np.fromfile(f, dtype=np.int64, count=1)[0])
        codes = np.fromfile(f, dtype=np.int16, count=int(nrows * ncols)).reshape(
            int(nrows), int(ncols)
        )

    centroids_per_subs: list[np.ndarray] = []
    with open(centroids_fp, "rb") as f:
        n_subs = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
        for _ in range(n_subs):
            n_centroids = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            subs_dim = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            centroids_data = np.fromfile(
                f, dtype=np.float32, count=int(n_centroids * subs_dim)
            )
            centroids_per_subs.append(
                centroids_data.reshape(int(n_centroids), int(subs_dim)).astype(np.float32)
            )

    dim_meta = None
    meta_path = root / "metadata.json"
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = json.load(f)
        dim_meta = int(meta.get("summary", {}).get("dim", 0)) or None

    return codes, centroids_per_subs, dim_meta or 0


@njit(parallel=True)
def _vaq_ip_sum(
    dot_tables: np.ndarray,
    codes: np.ndarray,
    n_subspaces: int,
    offsets: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """dot_tables: (nq, total_k) float32 flattened LUTs; codes (nb, n_subspaces)."""
    nq = dot_tables.shape[0]
    nb = codes.shape[0]
    out = np.zeros((nq, nb), dtype=np.float32)
    for qq in prange(nq):
        for i in range(nb):
            s = 0.0
            for j in range(n_subspaces):
                off = int(offsets[j])
                w = int(widths[j])
                cij = int(codes[i, j])
                if 0 <= cij < w:
                    s += dot_tables[qq, off + cij]
            out[qq, i] = s
    return out


class VAQBackend(ADCBackend):
    method_name = "VAQ"

    def __init__(self, artifact_dir: str | Path, dim: int):
        self.root = Path(artifact_dir)
        self.codes_full, self._centroids_per_sub, dim_from_meta = _read_vaq_codes_centroids(
            self.root
        )
        self.n_subspaces = len(self._centroids_per_sub)
        if dim_from_meta > 0:
            self.dim = dim_from_meta
        else:
            self.dim = int(dim)
        subs_len = self.dim // self.n_subspaces
        if self.dim % self.n_subspaces > 0:
            subs_len += 1
        self._subs_len = subs_len
        # per-subspace LUT width (max centroid index + 1)
        self._widths = np.array([c.shape[0] for c in self._centroids_per_sub], dtype=np.int64)
        self._offsets = np.zeros(self.n_subspaces, dtype=np.int64)
        for j in range(1, self.n_subspaces):
            self._offsets[j] = self._offsets[j - 1] + self._widths[j - 1]

    def encode(self, db: np.ndarray) -> Any:
        nb = int(db.shape[0])
        if nb > self.codes_full.shape[0]:
            raise ValueError(
                f"db has {nb} rows but VAQ codes only has {self.codes_full.shape[0]}"
            )
        return self.codes_full[:nb].copy()

    def db_norms_sq(self, codes: Any) -> np.ndarray:
        c = np.asarray(codes, dtype=np.int64)
        nb = c.shape[0]
        total = np.zeros(nb, dtype=np.float32)
        for j in range(self.n_subspaces):
            cent = self._centroids_per_sub[j]
            idx = c[:, j]
            valid = (idx >= 0) & (idx < cent.shape[0])
            sel = cent[np.clip(idx, 0, cent.shape[0] - 1)]
            sq = np.sum(sel * sel, axis=1).astype(np.float32)
            total += np.where(valid, sq, 0.0)
        return total

    def prepare_query(self, q: np.ndarray) -> Any:
        q = np.ascontiguousarray(q, dtype=np.float32)
        nq = q.shape[0]
        total_w = int(self._offsets[-1] + self._widths[-1])
        flat = np.zeros((nq, total_w), dtype=np.float32)
        for j in range(self.n_subspaces):
            start_dim = j * self._subs_len
            end_dim = min(start_dim + self._subs_len, self.dim)
            cent = self._centroids_per_sub[j]
            sd = cent.shape[1]
            q_sub = q[:, start_dim:end_dim]
            if q_sub.shape[1] != sd:
                if q_sub.shape[1] < sd:
                    padded = np.zeros((nq, sd), dtype=np.float32)
                    padded[:, : q_sub.shape[1]] = q_sub
                    q_sub = padded
                else:
                    q_sub = q_sub[:, :sd]
            off = int(self._offsets[j])
            w = int(self._widths[j])
            # IP table: (nq, n_centroids) = q_sub @ cent.T
            flat[:, off : off + w] = q_sub @ cent.astype(np.float32).T
        return flat

    def ip_estimate(self, qstate: Any, codes: Any) -> np.ndarray:
        dot_flat = np.ascontiguousarray(qstate, dtype=np.float32)
        c = np.ascontiguousarray(codes, dtype=np.int64)
        return _vaq_ip_sum(
            dot_flat,
            c,
            self.n_subspaces,
            self._offsets,
            self._widths,
        )
