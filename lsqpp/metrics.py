"""Relative-error computation aligned with PQ/OPQ eval scripts."""

import numpy as np
from scipy.spatial.distance import cdist

REL_EPSILON = 1e-6
MAX_REL_CLIP = 1e6
MAX_SAFE_FLOAT32 = np.finfo(np.float32).max / 10.0


def compute_rel_error_pq_style(
    adc_sample: np.ndarray, exact_sample: np.ndarray
) -> tuple[np.ndarray, float, float]:
    adc_sample = np.clip(adc_sample.astype(np.float32), 0, MAX_SAFE_FLOAT32)
    exact_64 = exact_sample.astype(np.float64)
    adc_64 = adc_sample.astype(np.float64)
    denominator = np.maximum(exact_64, REL_EPSILON)
    diff_64 = np.abs(adc_64 - exact_64)
    rel_error_64 = diff_64 / denominator
    rel_error = np.clip(rel_error_64, 0, MAX_REL_CLIP).astype(np.float32)

    valid_mask = np.isfinite(rel_error) & (rel_error >= 0) & (rel_error < MAX_REL_CLIP)
    if valid_mask.sum() == 0:
        return rel_error, float("nan"), float("nan")
    rel_clean = rel_error[valid_mask].astype(np.float64)
    rel_clean = rel_clean[np.isfinite(rel_clean)]
    if rel_clean.size == 0:
        return rel_error, float("nan"), float("nan")
    return rel_error, float(rel_clean.mean()), float(rel_clean.std())


def exact_distances_sqeuclidean(
    qr_sample: np.ndarray, db_sample: np.ndarray
) -> np.ndarray:
    with np.errstate(over="ignore"):
        exact_64 = cdist(qr_sample, db_sample, metric="sqeuclidean")
    return np.clip(exact_64, 0, MAX_SAFE_FLOAT32).astype(np.float32)


def _rank_1d(values: np.ndarray) -> np.ndarray:
    """Fast ranks for distance arrays. Ties get arbitrary stable ranks; adequate for float distances."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.arange(order.shape[0], dtype=np.float64)
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(x, y) / denom)


def mean_spearman_rank(adc_sample: np.ndarray, exact_sample: np.ndarray) -> float:
    if exact_sample.shape != adc_sample.shape:
        raise ValueError(f"shape mismatch exact={exact_sample.shape}, adc={adc_sample.shape}")
    nq = exact_sample.shape[0]
    vals: list[float] = []
    for query_index in range(nq):
        r_exact = _rank_1d(exact_sample[query_index])
        r_adc = _rank_1d(adc_sample[query_index])
        corr = _pearson_corr(r_exact, r_adc)
        if np.isfinite(corr):
            vals.append(corr)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def compute_lsq_reconstruction_error(
    test_db_sample: np.ndarray, codes_ix_sample: np.ndarray, codebooks: np.ndarray
) -> float:
    """
    codebooks: (M, ksub, dim)
    codes_ix_sample: (n_sample, M)
    """
    nb, d = test_db_sample.shape
    M = codes_ix_sample.shape[1]
    recon = np.zeros_like(test_db_sample)
    for i in range(nb):
        for m in range(M):
            recon[i] += codebooks[m, codes_ix_sample[i, m]]
    distances = np.linalg.norm(test_db_sample - recon, axis=1)
    return float(np.mean(distances))
