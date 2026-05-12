"""
Evaluate TurboQuant-MSE across bit_per_dim ∈ {1..10} on multiple datasets.

For each (dataset, bit_per_dim) we compute three measures on the first
``n_base`` (default 10 000) database vectors and the first ``n_query``
(default 1 000) query vectors:

  - MSE distortion        : mean_i ||x_i - x_hat_i||^2
  - L2 relative error     : mean_i ||x_i - x_hat_i|| / ||x_i||
  - Spearman rho (L2)     : per-query Spearman rank correlation between
                             approximate L2 distances (q to x_hat) and exact
                             L2 distances (q to x), then averaged over queries.

CSV output columns are stable for downstream merge with cpanourg's results.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.stats import spearmanr

from data_loaders import load_eval_set
from turboquant_mse import TurboQuantMSE


# ----------------------------------------------------------------------------
# Distance / rank utilities (GPU-side).
# ----------------------------------------------------------------------------

@torch.no_grad()
def pairwise_l2_sq(Q: torch.Tensor, X: torch.Tensor,
                   chunk_q: int = 256) -> torch.Tensor:
    """Returns ||Q_i - X_j||^2 with shape [Nq, Nx], chunked along Q for memory."""
    x2 = (X * X).sum(dim=1)  # [Nx]
    out = torch.empty((Q.shape[0], X.shape[0]),
                      device=Q.device, dtype=torch.float32)
    for i in range(0, Q.shape[0], chunk_q):
        qb = Q[i:i + chunk_q]
        q2 = (qb * qb).sum(dim=1, keepdim=True)  # [b, 1]
        out[i:i + qb.shape[0]] = (q2 + x2.unsqueeze(0) - 2.0 * (qb @ X.T)).clamp(min=0)
    return out


def spearman_per_query_mean(approx: torch.Tensor,
                            exact: torch.Tensor) -> Dict[str, float]:
    """Compute Spearman rho per row, then return mean / std / quantiles.

    Uses scipy.stats.spearmanr per row (handles ties properly).
    """
    approx_np = approx.detach().cpu().numpy()
    exact_np = exact.detach().cpu().numpy()
    Nq = approx_np.shape[0]
    rhos = np.empty(Nq, dtype=np.float64)
    for i in range(Nq):
        # spearmanr returns SpearmanrResult(correlation, pvalue) or a single scalar
        # for newer scipy. Use .statistic for compatibility.
        res = spearmanr(approx_np[i], exact_np[i])
        rho = res.correlation if hasattr(res, "correlation") else res[0]
        if not np.isfinite(rho):
            rho = 0.0
        rhos[i] = rho
    return {
        "spearman_mean": float(rhos.mean()),
        "spearman_std":  float(rhos.std()),
        "spearman_p10":  float(np.percentile(rhos, 10)),
        "spearman_p50":  float(np.percentile(rhos, 50)),
        "spearman_p90":  float(np.percentile(rhos, 90)),
    }


# ----------------------------------------------------------------------------
# Per-config run.
# ----------------------------------------------------------------------------

@torch.no_grad()
def run_one_config(X_base: torch.Tensor, X_query: torch.Tensor,
                   d: int, bit_per_dim: int,
                   device: str, seed: int = 42) -> Dict:
    t0 = time.perf_counter()
    q = TurboQuantMSE(d=d, bit_per_dim=bit_per_dim, device=device, seed=seed)
    setup_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_recon = q.encode_decode(X_base, batch_size=8192)
    encode_time = time.perf_counter() - t0

    diff = X_base - X_recon
    sq_err = (diff * diff).sum(dim=1)
    mse_distortion = sq_err.mean().item()

    x_norms = X_base.norm(dim=1).clamp(min=1e-12)
    l2_rel = (sq_err.sqrt() / x_norms).mean().item()

    # ranking measure: query vs exact base vs query vs reconstructed base
    t0 = time.perf_counter()
    Xb_dev = X_base.to(device, non_blocking=True)
    Xr_dev = X_recon.to(device, non_blocking=True)
    Xq_dev = X_query.to(device, non_blocking=True)
    exact = pairwise_l2_sq(Xq_dev, Xb_dev)
    approx = pairwise_l2_sq(Xq_dev, Xr_dev)
    rank_stats = spearman_per_query_mean(approx, exact)
    spearman_time = time.perf_counter() - t0

    out = {
        "bit_per_dim": bit_per_dim,
        "K": 2 ** bit_per_dim,
        "d": d,
        "mse_distortion": mse_distortion,
        "l2_rel_err_mean": l2_rel,
        **rank_stats,
        "setup_time_s": setup_time,
        "encode_time_s": encode_time,
        "spearman_time_s": spearman_time,
    }
    return out


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------

CSV_FIELDS = [
    "method", "dataset", "d", "bit_per_dim", "K",
    "n_base", "n_query",
    "mse_distortion", "l2_rel_err_mean",
    "spearman_mean", "spearman_std",
    "spearman_p10", "spearman_p50", "spearman_p90",
    "setup_time_s", "encode_time_s", "spearman_time_s",
    "total_time_s",
]


def append_row(csv_path: Path, row: Dict) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["deep", "bigann", "gist", "msmarco", "openai"])
    parser.add_argument("--bits", type=int, nargs="+",
                        default=list(range(1, 11)))
    parser.add_argument("--n_base", type=int, default=10_000)
    parser.add_argument("--n_query", type=int, default=1_000)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, falling back to CPU (will be slow).")
    print(f"[load] dataset={args.dataset}  n_base={args.n_base}  n_query={args.n_query}")
    t0 = time.perf_counter()
    X_base_np, X_query_np = load_eval_set(args.dataset,
                                          n_base=args.n_base,
                                          n_query=args.n_query)
    load_time = time.perf_counter() - t0
    print(f"  loaded in {load_time:.2f}s. base={X_base_np.shape}, query={X_query_np.shape}")

    d = X_base_np.shape[1]
    X_base = torch.from_numpy(X_base_np).to(torch.float32)
    X_query = torch.from_numpy(X_query_np).to(torch.float32)

    if args.output_csv is None:
        out_path = Path("/data/qwang/q/turboquant/results") / f"tqmse_{args.dataset}_eval.csv"
    else:
        out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run] device={device}  bits={args.bits}  -> {out_path}")
    print(f"{'b':>3} {'K':>5} {'mse':>10} {'rel_err':>9} {'spearman':>9} "
          f"{'time':>7}")
    print("-" * 56)
    for b in args.bits:
        t0 = time.perf_counter()
        res = run_one_config(X_base, X_query, d=d, bit_per_dim=b,
                             device=device, seed=args.seed)
        res["dataset"] = args.dataset
        res["method"] = "TQMSE"
        res["n_base"] = args.n_base
        res["n_query"] = args.n_query
        res["total_time_s"] = time.perf_counter() - t0
        print(f"{b:>3} {res['K']:>5} {res['mse_distortion']:>10.5f} "
              f"{res['l2_rel_err_mean']:>9.4f} {res['spearman_mean']:>9.4f} "
              f"{res['total_time_s']:>6.1f}s")
        append_row(out_path, res)
    print(f"\nDone. Wrote {out_path}")


if __name__ == "__main__":
    main()
