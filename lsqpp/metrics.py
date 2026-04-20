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
