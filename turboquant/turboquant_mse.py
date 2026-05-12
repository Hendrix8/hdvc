"""
TurboQuant-MSE: per-coordinate scalar quantization on Haar-rotated vectors.

Faithful implementation of Algorithm 1 from
    Zandieh et al. "TurboQuant: Online Vector Quantization with Near-optimal
    Distortion Rate" (arXiv:2504.19874, 2025).

Pipeline (data-oblivious):
    encode(x):
        1. n  = ||x||
        2. u  = x / n
        3. y  = Pi @ u                              # Haar random rotation
        4. idx_j = argmin_k |y_j - c_k|  for j in [d]
        return (idx, n)

    decode(idx, n):
        1. y_recon_j = c_{idx_j}
        2. x_recon   = n * Pi.T @ y_recon

The 1D codebook {c_1, ..., c_{2^b}} is the optimal Lloyd-Max scalar quantizer
for the Beta-like density
        f_Y(y) ∝ (1 - y^2)^((d-3)/2),  y in [-1, 1],
which is the marginal distribution of any coordinate of a uniform point on
S^{d-1}. The codebook only depends on (d, b) and is cached on disk.

Only parameters exposed: ``d`` and ``bit_per_dim`` (per user request).
block_size / outlier handling are not supported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.special import betainc, betaincinv, betaln

CODEBOOK_DIR = Path(__file__).resolve().parent / "codebooks_tqmse"
CODEBOOK_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 1D codebook: optimal Lloyd-Max for f_Y(y) = C_d * (1-y^2)^((d-3)/2)  on [-1, 1].
# Uses the analytical formulas via the regularized incomplete Beta function so
# the result is exact (up to scipy's quadrature precision), not subject to the
# sampling noise of a Monte-Carlo Lloyd-Max which severely under-fits at high K.
#
# Setup with u = (1+y)/2 ~ Beta(alpha, alpha), alpha = (d-1)/2:
#   F_Y(y) = betainc(alpha, alpha, (1+y)/2)
#   F_Y^{-1}(p) = 2 * betaincinv(alpha, alpha, p) - 1
#   Normalization: C_d = 1 / B(1/2, (d-1)/2)
#   ∫_a^b y * f_Y(y) dy = C_d / (d-1) * [(1-a^2)^((d-1)/2) - (1-b^2)^((d-1)/2)]
#   E[Y | a<=Y<=b]      = numerator / (F_Y(b) - F_Y(a))
# ----------------------------------------------------------------------------


def _analytical_lloyd_max(d: int, K: int,
                          max_iters: int = 400,
                          tol: float = 1e-14
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Lloyd-Max for the symmetric Beta marginal of a Haar-rotated
    unit vector in R^d."""
    alpha = (d - 1) / 2.0
    log_Cd = -betaln(0.5, alpha)  # log of normalization constant of f_Y

    def F_Y(y):
        u = (1.0 + y) / 2.0
        return betainc(alpha, alpha, u)

    def F_Y_inv(p):
        return 2.0 * betaincinv(alpha, alpha, p) - 1.0

    def conditional_mean(a, b):
        """Vectorized E[Y | a<=Y<=b] using expm1 to avoid catastrophic
        cancellation when adjacent boundaries are very close (high K)."""
        v_a = np.clip(1.0 - a * a, 0.0, 1.0)
        v_b = np.clip(1.0 - b * b, 0.0, 1.0)
        with np.errstate(divide="ignore"):
            log_a = alpha * np.log(np.where(v_a > 0, v_a, 1e-300))
            log_b = alpha * np.log(np.where(v_b > 0, v_b, 1e-300))
        # diff = exp(log_a) - exp(log_b)
        #      = sign(log_a - log_b) * exp(max) * (-expm1(-|log_a - log_b|))
        max_log = np.maximum(log_a, log_b)
        abs_diff = np.abs(log_a - log_b)
        sign = np.where(log_a >= log_b, 1.0, -1.0)
        diff = sign * np.exp(max_log) * (-np.expm1(-abs_diff))
        num = np.exp(log_Cd) * diff / (d - 1.0)
        den = F_Y(b) - F_Y(a)
        out = np.where(np.abs(den) > 1e-30, num / den, 0.5 * (a + b))
        return out

    # Initialize boundaries at equal-probability quantiles
    boundaries = np.zeros(K + 1, dtype=np.float64)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    if K > 1:
        ps = np.arange(1, K, dtype=np.float64) / K
        boundaries[1:-1] = F_Y_inv(ps)

    prev_centroids = None
    centroids = None
    for _ in range(max_iters):
        centroids = conditional_mean(boundaries[:-1], boundaries[1:])
        centroids = np.sort(centroids)
        new_boundaries = boundaries.copy()
        new_boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        if prev_centroids is not None and \
                np.max(np.abs(centroids - prev_centroids)) < tol:
            boundaries = new_boundaries
            break
        prev_centroids = centroids.copy()
        boundaries = new_boundaries

    assert centroids is not None
    return centroids, boundaries


def compute_codebook(d: int, bit_per_dim: int,
                     device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the optimal 1D codebook for (d, bit_per_dim). Cached on disk."""
    K = 2 ** bit_per_dim
    path = CODEBOOK_DIR / f"cb_d{d}_b{bit_per_dim}.json"
    if path.exists():
        obj = json.load(open(path))
        centroids = torch.tensor(obj["centroids"], dtype=torch.float32)
        boundaries = torch.tensor(obj["boundaries"], dtype=torch.float32)
        return centroids, boundaries

    centroids_np, boundaries_np = _analytical_lloyd_max(d, K)
    centroids = torch.from_numpy(centroids_np.astype(np.float32))
    boundaries = torch.from_numpy(boundaries_np.astype(np.float32))

    json.dump({
        "d": d,
        "bit_per_dim": bit_per_dim,
        "K": K,
        "method": "analytical_lloyd_max_betainc",
        "centroids": centroids.tolist(),
        "boundaries": boundaries.tolist(),
    }, open(path, "w"))
    return centroids, boundaries


# ----------------------------------------------------------------------------
# Haar random rotation Pi (d x d).
# ----------------------------------------------------------------------------

def sample_haar_orthogonal(d: int, seed: int = 42,
                           device: str = "cuda",
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(d, d, generator=g, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    s = torch.sign(torch.diag(R))
    s[s == 0] = 1
    return Q * s.unsqueeze(0)


# ----------------------------------------------------------------------------
# TurboQuant-MSE quantizer.
# ----------------------------------------------------------------------------

class TurboQuantMSE:
    """Per-coordinate scalar quantizer on Haar-rotated, unit-norm vectors.

    Parameters
    ----------
    d : int
        Vector dimension.
    bit_per_dim : int
        Bits per coordinate; codebook size K = 2**bit_per_dim.
    device : str
        Torch device for compute (e.g. ``"cuda:0"``).
    seed : int
        RNG seed for the Haar rotation matrix.
    """

    def __init__(self, d: int, bit_per_dim: int,
                 device: str = "cuda", seed: int = 42):
        if bit_per_dim < 1:
            raise ValueError(f"bit_per_dim must be >= 1, got {bit_per_dim}")
        self.d = int(d)
        self.b = int(bit_per_dim)
        self.K = 1 << self.b
        self.device = device
        self.seed = seed

        centroids, boundaries = compute_codebook(self.d, self.b, device=device)
        self.centroids = centroids.to(device)
        # Inner boundaries (length K-1) are what bucketize needs.
        self._inner_boundaries = boundaries[1:-1].to(device).contiguous()
        self.Q = sample_haar_orthogonal(self.d, seed=seed, device=device)

    # --- core ops -----------------------------------------------------------

    @torch.no_grad()
    def encode(self, X: torch.Tensor, batch_size: int = 8192
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (codes [N, d] int32, norms [N] float32) on CPU."""
        if X.dim() != 2 or X.shape[1] != self.d:
            raise ValueError(
                f"X must have shape [N, {self.d}], got {tuple(X.shape)}")
        N = X.shape[0]
        out_codes = torch.empty((N, self.d), dtype=torch.int32)
        out_norms = torch.empty(N, dtype=torch.float32)
        for i in range(0, N, batch_size):
            xb = X[i:i + batch_size].to(self.device, dtype=torch.float32,
                                        non_blocking=True)
            norms = xb.norm(dim=1)
            xunit = xb / norms.unsqueeze(1).clamp(min=1e-12)
            y = xunit @ self.Q
            codes = torch.bucketize(y, self._inner_boundaries)  # [B, d]
            out_codes[i:i + xb.shape[0]] = codes.to(torch.int32, copy=False).cpu()
            out_norms[i:i + xb.shape[0]] = norms.cpu()
        return out_codes, out_norms

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, norms: torch.Tensor,
               batch_size: int = 8192) -> torch.Tensor:
        """Returns x_recon [N, d] float32 on CPU."""
        N = codes.shape[0]
        out = torch.empty((N, self.d), dtype=torch.float32)
        QT = self.Q.T.contiguous()
        for i in range(0, N, batch_size):
            cb = codes[i:i + batch_size].to(self.device, dtype=torch.long,
                                            non_blocking=True)
            nb = norms[i:i + batch_size].to(self.device, dtype=torch.float32,
                                            non_blocking=True)
            y_recon = self.centroids[cb]
            x_unit = y_recon @ QT
            out[i:i + cb.shape[0]] = (x_unit * nb.unsqueeze(1)).cpu()
        return out

    @torch.no_grad()
    def encode_decode(self, X: torch.Tensor,
                      batch_size: int = 8192) -> torch.Tensor:
        """Fused encode+decode, useful for distortion measurement."""
        if X.dim() != 2 or X.shape[1] != self.d:
            raise ValueError(
                f"X must have shape [N, {self.d}], got {tuple(X.shape)}")
        N = X.shape[0]
        out = torch.empty((N, self.d), dtype=torch.float32)
        QT = self.Q.T.contiguous()
        for i in range(0, N, batch_size):
            xb = X[i:i + batch_size].to(self.device, dtype=torch.float32,
                                        non_blocking=True)
            norms = xb.norm(dim=1)
            xunit = xb / norms.unsqueeze(1).clamp(min=1e-12)
            y = xunit @ self.Q
            codes = torch.bucketize(y, self._inner_boundaries)
            y_recon = self.centroids[codes]
            x_unit_recon = y_recon @ QT
            out[i:i + xb.shape[0]] = (x_unit_recon * norms.unsqueeze(1)).cpu()
        return out


if __name__ == "__main__":
    # Quick sanity test: pseudo-Beta with d=128, b=2 should give centroids near
    # ±sqrt(2/(pi*d)) ≈ ±0.0705 for the K=2 case (paper eq. for large d).
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    for d in [96, 128, 1536]:
        for b in [1, 2, 4]:
            c, _ = compute_codebook(d, b, device=device)
            print(f"  d={d:4d}  b={b}  K={2**b:4d}  centroids[:4]={c[:4].tolist()}")

    # Round-trip distortion on synthetic Gaussian data
    d = 128
    X = torch.randn(2000, d)
    for b in [1, 2, 4, 8]:
        q = TurboQuantMSE(d=d, bit_per_dim=b, device=device)
        x_recon = q.encode_decode(X)
        diff = X - x_recon
        mse = (diff * diff).sum(dim=1).mean().item()
        x_norms = X.norm(dim=1)
        rel = ((diff * diff).sum(dim=1).sqrt() / x_norms).mean().item()
        # Paper Theorem 1: for unit-norm x, MSE per-vector ≈ 0.36, 0.117, 0.03, 0.009
        # (for b=1..4). For non-unit x we expect MSE ≈ ||x||^2 * those values.
        expected = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}.get(b, None)
        print(f"  b={b}: per-vec MSE={mse:.5f}, rel_err={rel:.4f}"
              + (f"  (paper ≈ ||x||^2 * {expected})" if expected else ""))
