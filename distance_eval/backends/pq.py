"""PQ: FAISS inner-product LUT + packed-code IP accumulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numba import njit, prange

from distance_eval.base import ADCBackend
from distance_eval.codecs import decode_pq_codes


def _extract_pq(index: faiss.Index) -> faiss.ProductQuantizer:
    if hasattr(index, "pq"):
        return index.pq
    if hasattr(index, "index"):
        sub = faiss.downcast_index(index.index)
        if hasattr(sub, "pq"):
            return sub.pq
    raise ValueError("Cannot extract ProductQuantizer from index")


@njit(parallel=True)
def _pq_ip_batch_numba(ip_tables: np.ndarray, codes_idx: np.ndarray, M: int) -> np.ndarray:
    """ip_tables (nq, M, ksub), codes_idx (nb, M) int32 → (nq, nb) IP sum."""
    nq = ip_tables.shape[0]
    nb = codes_idx.shape[0]
    out = np.empty((nq, nb), dtype=np.float32)
    for qq in prange(nq):
        for i in range(nb):
            s = 0.0
            for j in range(M):
                s += ip_tables[qq, j, codes_idx[i, j]]
            out[qq, i] = s
    return out


def _pq_db_norms_sq(centroids: np.ndarray, codes_idx: np.ndarray) -> np.ndarray:
    """centroids (M, ksub, dsub), codes_idx (nb, M) → (nb,) ||x̂||²."""
    M = centroids.shape[0]
    nb = codes_idx.shape[0]
    total = np.zeros(nb, dtype=np.float32)
    for j in range(M):
        sel = centroids[j][codes_idx[:, j].astype(np.int64, copy=False)]
        total += np.sum(sel.astype(np.float32) * sel.astype(np.float32), axis=1).astype(
            np.float32
        )
    return total


class PQBackend(ADCBackend):
    method_name = "PQ"

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        index = faiss.read_index(str(self.index_path))
        self.pq = _extract_pq(index)
        self.M = int(self.pq.M)
        self.nbits = int(self.pq.nbits)
        self.dim = int(self.pq.d)
        self.ksub = 1 << self.nbits
        cent_flat = faiss.vector_to_array(self.pq.centroids)
        self._centroids = cent_flat.reshape(self.M, self.ksub, self.dim // self.M).astype(
            np.float32, copy=False
        )

    def encode(self, db: np.ndarray) -> Any:
        db = np.ascontiguousarray(db, dtype=np.float32)
        codes_flat = self.pq.compute_codes(db).ravel()
        nb = db.shape[0]
        return decode_pq_codes(codes_flat, nb, self.M, self.nbits)

    def db_norms_sq(self, codes: Any) -> np.ndarray:
        codes_idx = np.asarray(codes, dtype=np.int32)
        return _pq_db_norms_sq(self._centroids, codes_idx)

    def prepare_query(self, q: np.ndarray) -> Any:
        q = np.ascontiguousarray(q, dtype=np.float32)
        nq = q.shape[0]
        ip_flat = np.empty((nq, self.M * self.ksub), dtype=np.float32)
        self.pq.compute_inner_prod_tables(nq, faiss.swig_ptr(q), faiss.swig_ptr(ip_flat))
        return ip_flat.reshape(nq, self.M, self.ksub).astype(np.float32, copy=False)

    def ip_estimate(self, qstate: Any, codes: Any) -> np.ndarray:
        ip_tables = np.ascontiguousarray(qstate, dtype=np.float32)
        codes_idx = np.ascontiguousarray(codes, dtype=np.int32)
        return _pq_ip_batch_numba(ip_tables, codes_idx, self.M)
