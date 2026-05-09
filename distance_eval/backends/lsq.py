"""LSQ++: dot-product tables + per-vector alpha (||sum c_j||²)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numba import njit, prange

from distance_eval.base import ADCBackend
from distance_eval.codecs import unpack_lsq_codes
from lsqpp.lsq_adc import lsq_alpha_from_codes, lsq_dot_tables_vectorized


@njit(parallel=True)
def _lsq_ip_sum_only(dot_tables: np.ndarray, codes: np.ndarray, M: int) -> np.ndarray:
    nq, m, ksub = dot_tables.shape
    nb = codes.shape[0]
    out = np.empty((nq, nb), dtype=np.float32)
    for q in prange(nq):
        for i in range(nb):
            s = 0.0
            for j in range(m):
                s += dot_tables[q, j, codes[i, j]]
            out[q, i] = s
    return out


class LSQBackend(ADCBackend):
    method_name = "LSQpp"

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        index = faiss.read_index(str(self.index_path))
        idx = faiss.downcast_index(index)
        if type(idx).__name__ != "IndexLocalSearchQuantizer":
            raise ValueError(
                f"Expected IndexLocalSearchQuantizer, got {type(idx).__name__}"
            )
        self._index = idx
        self.lsq = idx.lsq
        self.M = int(self.lsq.M)
        self.nbits = int(self.lsq.nbits)
        self.dim = int(self.lsq.d)
        self.ksub = 1 << self.nbits
        table = np.ascontiguousarray(
            faiss.vector_to_array(self.lsq.codebooks).reshape(-1, self.dim),
            dtype=np.float32,
        )
        self._codebooks = table.reshape(self.M, self.ksub, self.dim)

    def encode(self, db: np.ndarray) -> Any:
        db = np.ascontiguousarray(db, dtype=np.float32)
        codes_u8 = np.asarray(self.lsq.compute_codes(db), dtype=np.uint8)
        return unpack_lsq_codes(codes_u8, self.M, self.nbits)

    def db_norms_sq(self, codes: Any) -> np.ndarray:
        codes_ix = np.asarray(codes, dtype=np.int64)
        return lsq_alpha_from_codes(self._codebooks, codes_ix)

    def prepare_query(self, q: np.ndarray) -> Any:
        q = np.ascontiguousarray(q, dtype=np.float32)
        return lsq_dot_tables_vectorized(q, self._codebooks)

    def ip_estimate(self, qstate: Any, codes: Any) -> np.ndarray:
        dot_tables = np.ascontiguousarray(qstate, dtype=np.float32)
        codes_ix = np.ascontiguousarray(codes, dtype=np.int64)
        return _lsq_ip_sum_only(dot_tables, codes_ix, self.M)
