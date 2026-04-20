"""LSQ ADC distance helpers (from legacy src.utils; numba + numpy only)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


def lsq_dot_tables_vectorized(queries: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """
    queries:   (nq, d) float32
    codebooks: (m, ksub, d) float32
    returns:   (nq, m, ksub) float32
    """
    nq, d = queries.shape
    m, ksub, d2 = codebooks.shape
    assert d == d2
    dot_tables = np.empty((nq, m, ksub), dtype=np.float32)
    for j in range(m):
        dot_tables[:, j, :] = queries @ codebooks[j].T
    return dot_tables


def lsq_alpha_from_codes(
    codebooks: np.ndarray, codes: np.ndarray, block: int = 200_000
) -> np.ndarray:
    """
    codebooks: (m, ksub, d)
    codes:     (nb, m) integer codes
    returns:   alpha (nb,) float32 reconstruction energy ||sum_j C_j[c_ij]||^2
    """
    nb, m = codes.shape
    d = codebooks.shape[2]
    alpha = np.empty(nb, dtype=np.float32)
    for start in range(0, nb, block):
        end = min(start + block, nb)
        cb = codes[start:end]
        accum = np.zeros((end - start, d), dtype=np.float32)
        for j in range(m):
            accum += codebooks[j][cb[:, j]]
        alpha[start:end] = np.einsum("bd,bd->b", accum, accum)
    return alpha


@njit(parallel=True)
def lsq_distances_batch_numba(
    dot_tables: np.ndarray, codes: np.ndarray, alpha: np.ndarray, qnorms: np.ndarray
) -> np.ndarray:
    nq, m, ksub = dot_tables.shape
    nb = codes.shape[0]
    out = np.empty((nq, nb), dtype=np.float32)
    for q in prange(nq):
        qn = qnorms[q]
        for i in range(nb):
            s = 0.0
            for j in range(m):
                s += dot_tables[q, j, codes[i, j]]
            out[q, i] = qn + alpha[i] - 2.0 * s
    return out
