"""Decode FAISS packed PQ / product-style codes to (nb, M) int indices."""

from __future__ import annotations

import numpy as np


def decode_pq_codes(codes_flat: np.ndarray, nb: int, M: int, nbits: int) -> np.ndarray:
    """Decode FAISS packed PQ codes → (nb, M) centroid-index array."""
    codes_flat = np.asarray(codes_flat).ravel()
    if nbits == 8:
        return codes_flat.reshape(nb, M).astype(np.int32)

    code_size = (M * nbits + 7) // 8
    codes_2d = codes_flat.reshape(nb, code_size)
    mask = (1 << nbits) - 1
    result = np.zeros((nb, M), dtype=np.int32)

    for j in range(M):
        bit_start = j * nbits
        byte_idx = bit_start >> 3
        bit_off = bit_start & 7

        val = codes_2d[:, byte_idx].astype(np.int32) >> bit_off
        bits_read = 8 - bit_off

        while bits_read < nbits:
            byte_idx += 1
            if byte_idx < code_size:
                val |= codes_2d[:, byte_idx].astype(np.int32) << bits_read
            bits_read += 8

        result[:, j] = val & mask

    return result


def unpack_lsq_codes(codes_packed: np.ndarray, M: int, nbits: int) -> np.ndarray:
    """Unpack FAISS LocalSearchQuantizer packed bytes to (n, M) indices."""
    import faiss

    if not hasattr(faiss, "unpack_bitstrings"):
        raise RuntimeError(
            "faiss.unpack_bitstrings is missing; use a full faiss-cpu build."
        )
    b = np.ascontiguousarray(codes_packed, dtype=np.uint8)
    unpacked = faiss.unpack_bitstrings(b, M, nbits)
    return np.asarray(unpacked, dtype=np.int64)
