"""OPQ: rotate query with OPQMatrix, then PQ IP LUT on rotated vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from distance_eval.backends.pq import (
    PQBackend,
    _pq_db_norms_sq,
    _pq_ip_batch_numba,
)
from distance_eval.base import ADCBackend
from distance_eval.codecs import decode_pq_codes


def _unwrap_opq_index(ipt: faiss.Index) -> tuple[faiss.VectorTransform, faiss.ProductQuantizer]:
    ipt = faiss.downcast_index(ipt)
    if type(ipt).__name__ != "IndexPreTransform":
        raise ValueError(f"Expected IndexPreTransform, got {type(ipt)}")
    sub = faiss.downcast_index(ipt.index)
    if not hasattr(sub, "pq"):
        raise ValueError("Sub-index has no PQ")
    pq = sub.pq
    chain = ipt.chain
    if hasattr(chain, "size"):
        if chain.size() < 1:
            raise ValueError("IndexPreTransform has empty chain")
        vt0 = chain.at(0)
    else:
        if len(chain) < 1:
            raise ValueError("IndexPreTransform has empty chain")
        vt0 = chain[0]
    opq = faiss.downcast_vector_transform(vt0)
    return opq, pq


class OPQBackend(ADCBackend):
    method_name = "OPQ"

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self._index = faiss.read_index(str(self.index_path))
        self._ipt = faiss.downcast_index(self._index)
        self.opq, self.pq = _unwrap_opq_index(self._ipt)
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
        codes_flat = self._ipt.sa_encode(db).ravel()
        nb = db.shape[0]
        return decode_pq_codes(codes_flat, nb, self.M, self.nbits)

    def db_norms_sq(self, codes: Any) -> np.ndarray:
        return _pq_db_norms_sq(self._centroids, np.asarray(codes, dtype=np.int32))

    def prepare_query(self, q: np.ndarray) -> Any:
        q = np.ascontiguousarray(q, dtype=np.float32)
        q_rot = np.ascontiguousarray(self.opq.apply_py(q), dtype=np.float32)
        nq = q_rot.shape[0]
        ip_flat = np.empty((nq, self.M * self.ksub), dtype=np.float32)
        self.pq.compute_inner_prod_tables(nq, faiss.swig_ptr(q_rot), faiss.swig_ptr(ip_flat))
        return ip_flat.reshape(nq, self.M, self.ksub).astype(np.float32, copy=False)

    def ip_estimate(self, qstate: Any, codes: Any) -> np.ndarray:
        ip_tables = np.ascontiguousarray(qstate, dtype=np.float32)
        codes_idx = np.ascontiguousarray(codes, dtype=np.int32)
        return _pq_ip_batch_numba(ip_tables, codes_idx, self.M)
