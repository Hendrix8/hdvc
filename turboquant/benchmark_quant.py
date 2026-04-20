#!/usr/bin/env python3
"""Quantization benchmark: TurboQuant (MSE / Prod) and RaBitQ on ANN vector datasets."""

import numpy as np
import time
import argparse
import csv
import functools
import os
from collections import defaultdict
from scipy import stats

DATA_ROOT = "/data/cpanourg/2-hdvc/data"

DATASETS = {
    "sift": {
        "base": f"{DATA_ROOT}/sift/sift1m/sift_base.fvecs",
        "query": f"{DATA_ROOT}/sift/sift1m/sift_query.fvecs",
        "dim": 128, "format": "fvecs",
    },
    "gist": {
        "base": f"{DATA_ROOT}/gist/gist_base.fvecs",
        "query": f"{DATA_ROOT}/gist/gist_query.fvecs",
        "dim": 960, "format": "fvecs",
    },
    "deep": {
        "base": f"{DATA_ROOT}/deep1b/dataset/fvecs/test_1m.fvecs",
        "query": f"{DATA_ROOT}/deep1b/dataset/fvecs/query_10k.fvecs",
        "dim": 96, "format": "fvecs",
    },
    "bigann": {
        "base": f"{DATA_ROOT}/bigann/SIFT1M/bigann_base.bvecs",
        "query": f"{DATA_ROOT}/bigann/SIFT1M/bigann_query.bvecs",
        "dim": 128, "format": "bvecs",
    },
    "msmarco": {
        "base": f"{DATA_ROOT}/msmarco/base1m.fvecs",
        "query": f"{DATA_ROOT}/msmarco/query10k.fvecs",
        "dim": 1024, "format": "fvecs",
    },
    "openai": {
        "base": f"{DATA_ROOT}/openai/openai_base1m.fvecs",
        "query": f"{DATA_ROOT}/openai/openai_query10k.fvecs",
        "dim": 1536, "format": "fvecs",
    },
}


# --- I/O ---

def read_fvecs(filename, max_n=None):
    fv = np.memmap(filename, dtype='float32', mode='r')
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, 1 + dim)[:, 1:]
    if max_n is not None:
        fv = fv[:max_n]
    return fv.copy().astype(np.float32)


def read_bvecs(filename, max_n=None):
    with open(filename, 'rb') as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
    record_size = 4 + dim
    fv = np.memmap(filename, dtype=np.uint8, mode='r')
    n = len(fv) // record_size
    fv = fv.reshape(n, record_size)[:, 4:]
    if max_n is not None:
        fv = fv[:max_n]
    return fv.copy().astype(np.float32)


def load_dataset(name, nb=10000, nq=1000):
    ds = DATASETS[name]
    reader = read_bvecs if ds["format"] == "bvecs" else read_fvecs
    print(f"Loading dataset '{name}' (d={ds['dim']})...")
    base = reader(ds["base"], nb)
    queries = reader(ds["query"], nq)
    print(f"  Base:  {base.shape}  Query: {queries.shape}")
    return base, queries, ds["dim"]


# --- Utilities ---

def generate_rotation(d, seed=None):
    """Haar-distributed random orthogonal matrix via QR of a Gaussian matrix."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)).astype(np.float32)
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return (Q * signs[np.newaxis, :]).astype(np.float32)


def normalize(x):
    """Decompose x = ||x|| * x_hat. Returns (x_hat, norms)."""
    if x.ndim == 1:
        norm = np.float32(max(np.linalg.norm(x), 1e-10))
        return (x / norm).astype(np.float32), norm
    norms = np.maximum(np.linalg.norm(x, axis=1), 1e-10).astype(np.float32)
    return (x / norms[:, None]).astype(np.float32), norms


@functools.lru_cache(maxsize=128)
def compute_centroids(d, b, max_iter=200, tol=1e-10):
    """Lloyd-Max optimal centroids under Gaussian N(0, 1/d) approximation."""
    k = 2 ** b
    sigma = 1.0 / np.sqrt(d)

    quantile_points = np.linspace(0.5 / k, 1 - 0.5 / k, k)
    centroids = stats.norm.ppf(quantile_points, loc=0, scale=sigma)

    for _ in range(max_iter):
        boundaries = np.empty(k + 1)
        boundaries[0] = centroids[0] - 10 * sigma
        boundaries[-1] = centroids[-1] + 10 * sigma
        boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2

        a_std = boundaries[:-1] / sigma
        b_std = boundaries[1:] / sigma
        denom = np.maximum(stats.norm.cdf(b_std) - stats.norm.cdf(a_std), 1e-15)
        new_centroids = sigma * (stats.norm.pdf(a_std) - stats.norm.pdf(b_std)) / denom

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = np.empty(k + 1)
    boundaries[0] = centroids[0] - 10 * sigma
    boundaries[-1] = centroids[-1] + 10 * sigma
    boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2
    return centroids.astype(np.float32), boundaries.astype(np.float32)


# --- Quantizers ---

class TurboQuantMSE:
    """b-bit MSE-optimal: rotation + Lloyd-Max scalar quantization.

    IP estimated via full reconstruction: <q, x> ≈ ||x|| * <Πq, ŷ>
    """
    def __init__(self, d, b, seed=42):
        self.d, self.b = d, b
        self.rotation = generate_rotation(d, seed)
        print(f"  Centroids d={d}, b={b}...", end="", flush=True)
        self.centroids, self.boundaries = compute_centroids(d, b)
        print(" ok")
        self.name = f"TQ-MSE-{b}bit"

    def encode(self, x):
        x_hat, norms = normalize(x)
        y = x_hat @ self.rotation.T
        dtype = np.uint8 if self.b <= 8 else np.uint16
        indices = np.searchsorted(self.boundaries[1:-1], y).astype(dtype)
        return {"indices": indices, "norms": norms}

    def decode(self, code):
        y_hat = self.centroids[code["indices"]]
        return (y_hat @ self.rotation) * code["norms"][:, None]

    def estimate_ip(self, code, queries):
        return self.decode(code) @ queries.T

    def quant_bits_per_dim(self):
        return self.b

    def bits_per_vector(self):
        return self.b * self.d + 32


class TurboQuantProd:
    """b-bit IP-unbiased: (b-1)-bit MSE + 1-bit QJL residual correction.

    IP estimated as: <q, x> ≈ ||x|| * (<Πq, ŷ> + sqrt(π/2)/d * ||r|| * <Sq, sign(Sr)>)
    where r = x̂ - x̃_mse is the quantization residual.
    """
    def __init__(self, d, b, seed=42):
        assert b >= 2
        self.d, self.b = d, b
        self.mse = TurboQuantMSE(d, b - 1, seed)
        self.S = np.random.default_rng(seed + 1000).standard_normal((d, d)).astype(np.float32)
        self.name = f"TQ-Prod-{b}bit"

    def encode(self, x):
        x_hat, norms = normalize(x)
        mse_code = self.mse.encode(x_hat)
        x_mse = self.mse.decode(mse_code)
        residual = x_hat - x_mse
        res_norms = np.linalg.norm(residual, axis=1).astype(np.float32)
        signs = np.sign(residual @ self.S.T).astype(np.int8)
        signs[signs == 0] = 1
        return {"mse_indices": mse_code["indices"], "qjl_signs": signs,
                "res_norms": res_norms, "norms": norms}

    def decode(self, code):
        x_mse = self.mse.centroids[code["mse_indices"]] @ self.mse.rotation
        scale = np.sqrt(np.pi / 2) / self.d
        x_qjl = scale * code["res_norms"][:, None] * (code["qjl_signs"].astype(np.float32) @ self.S)
        return (x_mse + x_qjl) * code["norms"][:, None]

    def estimate_ip(self, code, queries):
        x_mse = self.mse.centroids[code["mse_indices"]] @ self.mse.rotation
        ip_mse = x_mse @ queries.T
        scale = np.sqrt(np.pi / 2) / self.d
        ip_qjl = scale * code["res_norms"][:, None] * (code["qjl_signs"].astype(np.float32) @ self.S @ queries.T)
        return code["norms"][:, None] * (ip_mse + ip_qjl)

    def quant_bits_per_dim(self):
        return self.b

    def bits_per_vector(self):
        return (self.b - 1) * self.d + self.d + 64


class RaBitQ:
    """1-bit binary quantization with per-vector correction (Gao et al., SIGMOD 2024).

    IP estimated as: <q, x> ≈ ||x|| * c * <Πq, sign(Πx̂)>, where c = mean(|Πx̂|).
    """
    def __init__(self, d, seed=42):
        self.d = d
        self.rotation = generate_rotation(d, seed)
        self.name = "RaBitQ-1bit"

    def encode(self, x):
        x_hat, norms = normalize(x)
        y = x_hat @ self.rotation.T
        signs = np.sign(y).astype(np.int8)
        signs[signs == 0] = 1
        correction = np.mean(np.abs(y), axis=1).astype(np.float32)
        return {"signs": signs, "norms": norms, "correction": correction}

    def decode(self, code):
        y_hat = code["correction"][:, None] * code["signs"].astype(np.float32)
        return (y_hat @ self.rotation) * code["norms"][:, None]

    def estimate_ip(self, code, queries):
        q_rot = queries @ self.rotation.T
        ip_binary = code["signs"].astype(np.float32) @ q_rot.T
        return code["norms"][:, None] * code["correction"][:, None] * ip_binary

    def quant_bits_per_dim(self):
        return 1

    def bits_per_vector(self):
        return self.d + 64


# --- Evaluation ---

def eval_method(method, base, queries, exact_ip, exact_l2sq, x_sq, q_sq):
    nb, d = base.shape
    nq = queries.shape[0]

    t0 = time.time()
    code = method.encode(base)
    encode_time = time.time() - t0

    approx_ip = method.estimate_ip(code, queries)
    approx_l2sq = x_sq[:, None] + q_sq[None, :] - 2 * approx_ip

    if hasattr(method, "decode"):
        x_recon = method.decode(code)
        distortion = float(np.mean(np.sum((base - x_recon) ** 2, axis=1)))
    else:
        distortion = float("nan")

    ip_rel_err = np.abs(approx_ip - exact_ip) / np.maximum(np.abs(exact_ip), 1e-10)
    l2_rel_err = np.abs(approx_l2sq - exact_l2sq) / np.maximum(exact_l2sq, 1e-10)

    def recall_at_k(true_nn, ranking, ks):
        out = {}
        for k in ks:
            if k > nb:
                out[k] = float("nan")
            else:
                out[k] = float(np.mean(np.any(ranking[:k, :] == true_nn[None, :], axis=0)))
        return out

    l2_recalls = recall_at_k(np.argmin(exact_l2sq, axis=0), np.argsort(approx_l2sq, axis=0), [1, 10, 100])
    ip_recalls = recall_at_k(np.argmax(exact_ip, axis=0), np.argsort(-approx_ip, axis=0), [1, 10, 100])

    return {
        "name": method.name, "quant_bpd": method.quant_bits_per_dim(),
        "total_bits": method.bits_per_vector(), "_nb": nb, "_nq": nq,
        "encode_time": encode_time, "distortion": distortion,
        "ip_rel_err_mean": float(np.mean(ip_rel_err)),
        "ip_rel_err_median": float(np.median(ip_rel_err)),
        "l2_rel_err_mean": float(np.mean(l2_rel_err)),
        "l2_rel_err_median": float(np.median(l2_rel_err)),
        "l2_recall@1": l2_recalls[1], "l2_recall@10": l2_recalls[10], "l2_recall@100": l2_recalls[100],
        "ip_recall@1": ip_recalls[1], "ip_recall@10": ip_recalls[10], "ip_recall@100": ip_recalls[100],
    }


def print_results(results, d):
    nb, nq = results[0]["_nb"], results[0]["_nq"]
    by_bit = defaultdict(list)
    for r in results:
        by_bit[r["quant_bpd"]].append(r)

    metrics = [
        ("Distortion", "distortion",      True,  ".2f"),
        ("L2-RelErr",  "l2_rel_err_mean", True,  ".6f"),
        ("IP-RelErr",  "ip_rel_err_mean", True,  ".6f"),
        ("L2-R@1",     "l2_recall@1",     False, ".4f"),
        ("L2-R@10",    "l2_recall@10",    False, ".4f"),
        ("L2-R@100",   "l2_recall@100",   False, ".4f"),
        ("IP-R@1",     "ip_recall@1",     False, ".4f"),
        ("IP-R@10",    "ip_recall@10",    False, ".4f"),
        ("IP-R@100",   "ip_recall@100",   False, ".4f"),
    ]

    W = 120
    print(f"\n{'=' * W}")
    print(f"  Results: d={d}, nb={nb}, nq={nq}")
    print(f"{'=' * W}")

    for bpd in sorted(by_bit):
        group = sorted(by_bit[bpd], key=lambda r: r["name"])
        names = [r["name"] for r in group]
        cw = max(14, max(len(n) for n in names) + 2)

        print(f"\n{'─' * W}\n  {bpd} bit/dim\n{'─' * W}")
        hdr = f"  {'Metric':<16s}" + "".join(f" {n:>{cw}s}" for n in names)
        if len(group) >= 2:
            hdr += f"  {'Best':>{cw}s}"
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")

        for label, key, lower_better, fmt in metrics:
            vals = [r[key] for r in group]
            row = f"  {label:<16s}" + "".join(f" {v:{cw}{fmt}}" for v in vals)
            if len(group) >= 2:
                best = names[int(np.argmin(vals) if lower_better else np.argmax(vals))]
                row += f"  {best:>{cw}s}"
            print(row)

    print(f"\n{'=' * W}\n  Summary\n{'=' * W}")
    hdr = f"{'Method':<20s} {'b/d':>4s} {'Distort':>10s} {'L2-Err':>8s} {'IP-Err':>8s}" \
          f" {'L2-R@1':>7s} {'L2-R@10':>8s} {'L2-R@100':>8s}" \
          f" {'IP-R@1':>7s} {'IP-R@10':>8s} {'IP-R@100':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<20s} {r['quant_bpd']:>4d} {r['distortion']:>10.2f}"
              f" {r['l2_rel_err_mean']:>8.6f} {r['ip_rel_err_mean']:>8.6f}"
              f" {r['l2_recall@1']:>7.4f} {r['l2_recall@10']:>8.4f} {r['l2_recall@100']:>8.4f}"
              f" {r['ip_recall@1']:>7.4f} {r['ip_recall@10']:>8.4f} {r['ip_recall@100']:>8.4f}")
    print("=" * len(hdr))


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Quantization Benchmark")
    parser.add_argument("--dataset", default="sift", choices=list(DATASETS.keys()))
    parser.add_argument("--nb", type=int, default=10000)
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-prod", action="store_true")
    parser.add_argument("--no-rabitq", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    base, queries, d = load_dataset(args.dataset, args.nb, args.nq)

    print("\nComputing exact distances...")
    t0 = time.time()
    exact_ip = base @ queries.T
    x_sq = np.sum(base ** 2, axis=1)
    q_sq = np.sum(queries ** 2, axis=1)
    exact_l2sq = np.maximum(x_sq[:, None] + q_sq[None, :] - 2 * exact_ip, 0)
    print(f"  {time.time() - t0:.2f}s  ({base.shape[0]}x{queries.shape[0]} pairs)")

    print(f"\nInitializing quantizers (seed={args.seed})...")
    methods = []
    if not args.no_rabitq:
        methods.append(RaBitQ(d, seed=args.seed))
    for b in sorted(args.bits):
        methods.append(TurboQuantMSE(d, b, seed=args.seed))
    if not args.no_prod:
        for b in sorted(args.bits):
            if b >= 2:
                methods.append(TurboQuantProd(d, b, seed=args.seed))

    print(f"\nEvaluating {len(methods)} methods...")
    results = []
    for method in methods:
        print(f"  {method.name}...", end="", flush=True)
        t0 = time.time()
        r = eval_method(method, base, queries, exact_ip, exact_l2sq, x_sq, q_sq)
        results.append(r)
        print(f" {time.time() - t0:.1f}s")

    results.sort(key=lambda r: (r["quant_bpd"], r["name"]))
    print_results(results, d)

    # CSV output
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = args.output or os.path.join(out_dir, f"quant_benchmark_{args.dataset}_nb{args.nb}_nq{args.nq}.csv")

    columns = [
        "method", "dataset", "dim", "nb", "nq", "bits_per_dim",
        "distortion", "l2_rel_err_mean", "l2_rel_err_median",
        "ip_rel_err_mean", "ip_rel_err_median",
        "l2_recall@1", "l2_recall@10", "l2_recall@100",
        "ip_recall@1", "ip_recall@10", "ip_recall@100",
    ]
    r3 = lambda v: round(v, 3)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in results:
            w.writerow({
                "method": r["name"], "dataset": args.dataset, "dim": d,
                "nb": r["_nb"], "nq": r["_nq"], "bits_per_dim": r["quant_bpd"],
                "distortion": r3(r["distortion"]),
                "l2_rel_err_mean": r3(r["l2_rel_err_mean"]),
                "l2_rel_err_median": r3(r["l2_rel_err_median"]),
                "ip_rel_err_mean": r3(r["ip_rel_err_mean"]),
                "ip_rel_err_median": r3(r["ip_rel_err_median"]),
                "l2_recall@1": r3(r["l2_recall@1"]),
                "l2_recall@10": r3(r["l2_recall@10"]),
                "l2_recall@100": r3(r["l2_recall@100"]),
                "ip_recall@1": r3(r["ip_recall@1"]),
                "ip_recall@10": r3(r["ip_recall@10"]),
                "ip_recall@100": r3(r["ip_recall@100"]),
            })
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
