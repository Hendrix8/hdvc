#!/usr/bin/env python3
"""VAQ: train/encode via lib/VAQ run_vaq, then ADC vs exact L2² (PQ-aligned loading)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vaq.data_io import (
    ensure_dir,
    infer_dim_from_path,
    load_dataset,
    load_vectors_prefix,
    write_fvecs,
)
from vaq.metrics import (
    TEST_CAP,
    compute_rel_error_pq_style,
    exact_distances_sqeuclidean,
)


def _default_vaq_binary() -> Path:
    p = _REPO / "lib" / "VAQ" / "build" / "examples" / "run_vaq"
    return p


def _parse_method_params(method: str) -> tuple[int, int, int, int, float]:
    m = re.match(
        r"VAQ(\d+)m(\d+)min(\d+)max(\d+)var([\d.]+)", method.split(",")[0]
    )
    if not m:
        raise ValueError(f"Could not parse VAQ method string: {method}")
    total_bits, n_subspaces, min_bits, max_bits, variance = m.groups()
    return (
        int(total_bits),
        int(n_subspaces),
        int(min_bits),
        int(max_bits),
        float(variance),
    )


def _build_method_string(
    base_method: str,
    total_bits: int | None,
    n_subspaces: int | None,
    min_bits: int | None,
    max_bits: int | None,
    variance: float | None,
) -> str:
    if not any(
        x is not None
        for x in (total_bits, n_subspaces, min_bits, max_bits, variance)
    ):
        return base_method
    match = re.match(
        r"(VAQ)(\d+)(m)(\d+)(min)(\d+)(max)(\d+)(var)([\d.]+)(.*)",
        base_method,
    )
    if match:
        (
            prefix,
            old_tb,
            m,
            old_m,
            min_p,
            old_min,
            max_p,
            old_max,
            var_p,
            old_var,
            suffix,
        ) = match.groups()
        tb = int(total_bits) if total_bits is not None else int(old_tb)
        ns = int(n_subspaces) if n_subspaces is not None else int(old_m)
        mn = int(min_bits) if min_bits is not None else int(old_min)
        mx = int(max_bits) if max_bits is not None else int(old_max)
        vr = float(variance) if variance is not None else float(old_var)
        return f"VAQ{tb}m{ns}min{mn}max{mx}var{vr}{suffix}"
    if all(
        x is not None
        for x in (total_bits, n_subspaces, min_bits, max_bits, variance)
    ):
        search = base_method.split(",")[-1] if "," in base_method else "HEAP"
        return f"VAQ{int(total_bits)}m{int(n_subspaces)}min{int(min_bits)}max{int(max_bits)}var{float(variance)},{search}"
    return base_method


def run_vaq_eval(
    dataset_path: str,
    query_path: str | None = None,
    train_path: str | None = None,
    dim: int | None = None,
    dataset_name: str = "custom",
    data_root: str = "/data/cpanourg/2-hdvc/",
    method: str = "VAQ256m32min7max8var1,HEAP",
    total_bits: int | None = None,
    n_subspaces: int | None = None,
    min_bits: int | None = None,
    max_bits: int | None = None,
    variance: float | None = None,
    train_size: int = 100_000,
    sample_db: int = 10_000,
    sample_queries: int = 1_000,
    results_dir: str = "results/vaq",
    vaq_binary: str | None = None,
    refine: str = "100,200",
    k: int = 100,
    learn_ratio: float = 0.05,
    seed: int = 123,
    sample_mode: str = "first",
    skip_search: bool = False,
    reuse_artifact_dir: str | None = None,
    train_time_override: float | None = None,
    encoding_time_override: float | None = None,
):
    _ = seed
    if sample_mode not in ("first", "random"):
        raise ValueError("sample_mode must be 'first' or 'random'")

    method = _build_method_string(
        method, total_bits, n_subspaces, min_bits, max_bits, variance
    )
    vb = Path(vaq_binary) if vaq_binary else _default_vaq_binary()
    if not vb.is_file():
        raise FileNotFoundError(
            f"VAQ binary not found: {vb}. Build lib/VAQ (see lib/VAQ/README or CMake)."
        )

    data_root_p = Path(data_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^\w.\-]+", "_", method.split(",")[0])[:96]

    # --- Load data (PQ-style or legacy) ---
    if train_path:
        if query_path is None:
            raise ValueError("train_path requires query_path")
        if dim is None:
            dim = infer_dim_from_path(train_path)
        tp, dp = Path(train_path).resolve(), Path(dataset_path).resolve()
        same_file = tp == dp
        print(
            f"📂 PQ-style load: train={train_path}"
            + (" (same file as base)" if same_file else "")
        )
        if same_file:
            train_db = load_vectors_prefix(dataset_path, dim, train_size, 0)
            test_db = load_vectors_prefix(dataset_path, dim, TEST_CAP, train_size)
        else:
            train_db = load_vectors_prefix(train_path, dim, train_size, 0)
            test_db = load_vectors_prefix(dataset_path, dim, TEST_CAP, 0)
        if len(train_db) < train_size:
            print(f"⚠️  Training set has only {len(train_db)} vectors; using all")
            train_size = len(train_db)
        qr = load_vectors_prefix(query_path, dim, sample_queries, 0)
    else:
        print(f"📂 Loading dataset from {dataset_path}")
        db, qr = load_dataset(dataset_path, query_path, dim)
        test_size = min(TEST_CAP, len(db))
        n_need = train_size + test_size
        if n_need > len(db):
            train_size = max(1, min(train_size, len(db) - 1))
            test_size = min(test_size, len(db) - train_size)
            print(f"⚠️  Shrinking split: train_size={train_size}, test_size={test_size}")
        idxs = np.random.choice(db.shape[0], train_size + test_size, replace=False)
        train_db, test_db = db[idxs[:train_size]], db[idxs[train_size:]]

    nb, nq, dim = test_db.shape[0], qr.shape[0], test_db.shape[1]
    print(
        f"train_db={train_db.shape}, test_db={test_db.shape}, qr={qr.shape}, dim={dim}"
    )

    out_dir = (
        Path(reuse_artifact_dir).resolve()
        if reuse_artifact_dir
        else data_root_p / results_dir / dataset_name / f"vaq_{slug}_{ts}"
    )
    ensure_dir(out_dir)
    codes_fp = str(out_dir / "codes.fvecs")
    centroids_fp = str(out_dir / "centroids.fvecs")
    result_csv = str(out_dir / "vaq_results.csv")

    train_time = None
    encoding_time = None
    vaq_cpp_log: list[str] = []
    cmd: list[str] = []
    if reuse_artifact_dir:
        metadata_fp = out_dir / "metadata.json"
        if metadata_fp.exists():
            try:
                old_summary = json.loads(metadata_fp.read_text()).get("summary", {})
                train_time = float(old_summary.get("train_time_s", 0.0))
                encoding_time = float(old_summary.get("encoding_time_s", 0.0))
            except Exception:
                train_time = encoding_time = None
        print(f"♻️  Reusing VAQ artifacts from {out_dir}")
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="vaq_"))
        train_fp = str(temp_dir / "trainset.fvecs")
        dataset_fp = str(temp_dir / "dataset.fvecs")
        queries_fp = str(temp_dir / "queries.fvecs")
        write_fvecs(train_fp, train_db)
        write_fvecs(dataset_fp, test_db)
        if not skip_search:
            write_fvecs(queries_fp, qr)

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            f"{os.path.expanduser('~')}/local/glpk/lib:"
            f"{os.path.expanduser('~')}/local/armadillo/lib:"
            f"{env.get('CONDA_PREFIX', '')}/lib:"
            f"{env.get('LD_LIBRARY_PATH', '')}"
        )

        cmd = [
            str(vb),
            "--dataset",
            dataset_fp,
            "--trainset",
            train_fp,
            "--file-format-ori",
            "fvecs",
            "--timeseries-size",
            str(dim),
            "--dataset-size",
            str(nb),
            "--trainset-size",
            str(train_size),
            "--method",
            method,
            "--save-enc",
            codes_fp,
            "--save",
            centroids_fp,
            "--learn-ratio",
            str(learn_ratio),
        ]
        if skip_search:
            cmd.extend(["--skip-query", "1"])
        else:
            cmd.extend(
                [
                    "--queries",
                    queries_fp,
                    "--queries-size",
                    str(nq),
                    "--result",
                    result_csv,
                    "--k",
                    str(k),
                    "--refine",
                    refine,
                ]
            )

        t0 = time.time()
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            vaq_cpp_log.append(line)
            print(line, end="")
            if "Training time:" in line:
                try:
                    train_time = float(
                        line.split("Training time:")[1].split("s")[0].strip()
                    )
                except Exception:
                    pass
            if "Encoding time:" in line:
                try:
                    encoding_time = float(
                        line.split("Encoding time:")[1].split("s")[0].strip()
                    )
                except Exception:
                    pass
        proc.wait()
        total_cpp = time.time() - t0
        if train_time is None:
            train_time = total_cpp * 0.3
        if encoding_time is None:
            encoding_time = total_cpp * 0.7

        shutil.rmtree(temp_dir)

    if train_time is None:
        train_time = 0.0
    if encoding_time is None:
        encoding_time = 0.0
    if train_time_override is not None:
        train_time = train_time_override
    if encoding_time_override is not None:
        encoding_time = encoding_time_override

    if not os.path.exists(codes_fp):
        log = "".join(vaq_cpp_log)
        if skip_search and "unrecognized option" in log and "skip-query" in log:
            raise RuntimeError(
                f"VAQ did not write codes: {codes_fp}\n"
                "Your lib/VAQ/build/examples/run_vaq is older than the repo: it does not "
                "support --skip-query. Rebuild from source (with your conda env active if "
                "you use conda BLAS), e.g.\n"
                "  cd lib/VAQ/build && cmake .. && cmake --build . --target run_vaq -j\n"
                "Or run without --skip_search until the binary is rebuilt."
            ) from None
        raise RuntimeError(f"VAQ did not write codes: {codes_fp}")

    with open(codes_fp, "rb") as f:
        nrows = np.fromfile(f, dtype=np.int64, count=1)[0]
        ncols = np.fromfile(f, dtype=np.int64, count=1)[0]
        codes = np.fromfile(f, dtype=np.int16, count=int(nrows * ncols)).reshape(
            int(nrows), int(ncols)
        )

    centroids_per_subs: list[np.ndarray] = []
    with open(centroids_fp, "rb") as f:
        n_subs = np.fromfile(f, dtype=np.uint64, count=1)[0]
        for _ in range(int(n_subs)):
            n_centroids = np.fromfile(f, dtype=np.uint64, count=1)[0]
            subs_dim = np.fromfile(f, dtype=np.uint64, count=1)[0]
            centroids_data = np.fromfile(
                f, dtype=np.float32, count=int(n_centroids * subs_dim)
            )
            centroids_per_subs.append(
                centroids_data.reshape(int(n_centroids), int(subs_dim))
            )

    total_b, n_subspaces, min_b, max_b, var_f = _parse_method_params(method)
    subs_len = dim // n_subspaces
    if dim % n_subspaces > 0:
        subs_len += 1

    n_sample_q = min(sample_queries, nq)
    n_sample_db = min(sample_db, nb)
    if sample_mode == "first":
        db_idx = np.arange(n_sample_db)
        q_idx = np.arange(n_sample_q)
        test_db_sample = test_db[:n_sample_db]
        qr_sample = qr[:n_sample_q]
        codes_sample = codes[:n_sample_db]
    else:
        db_idx = np.random.choice(nb, n_sample_db, replace=False)
        q_idx = np.random.choice(nq, n_sample_q, replace=False)
        test_db_sample = test_db[db_idx]
        qr_sample = qr[q_idx]
        codes_sample = codes[db_idx]

    nq_sample, nb_sample = len(q_idx), len(db_idx)
    t0 = time.time()
    exact_sample = exact_distances_sqeuclidean(qr_sample, test_db_sample)
    cdist_time = time.time() - t0

    adc_sample = np.zeros((nq_sample, nb_sample), dtype=np.float32)
    distance_table_time = 0.0
    adc_time = 0.0
    for subs in range(n_subspaces):
        start_dim = subs * subs_len
        end_dim = min(start_dim + subs_len, dim)
        centroids_subs = centroids_per_subs[subs]
        query_subvecs = qr_sample[:, start_dim:end_dim]
        sd = centroids_subs.shape[1]
        if query_subvecs.shape[1] != sd:
            if query_subvecs.shape[1] < sd:
                padded = np.zeros((nq_sample, sd), dtype=np.float32)
                padded[:, : query_subvecs.shape[1]] = query_subvecs
                query_subvecs = padded
            else:
                query_subvecs = query_subvecs[:, :sd]

        t_lut = time.time()
        diff = query_subvecs[:, np.newaxis, :] - centroids_subs[np.newaxis, :, :]
        lut = np.sum(diff * diff, axis=2).astype(np.float32)
        distance_table_time += time.time() - t_lut

        t_adc = time.time()
        codes_sub = codes_sample[:, subs].astype(np.int64, copy=False)
        valid = (codes_sub >= 0) & (codes_sub < lut.shape[1])
        if np.any(valid):
            adc_sample[:, valid] += lut[:, codes_sub[valid]]
        adc_time += time.time() - t_adc

    rel_error, mean_rel, std_rel = compute_rel_error_pq_style(adc_sample, exact_sample)
    print(f"Mean rel. error: {mean_rel:.4f}, std: {std_rel:.4f}")

    safe_method = method.replace(",", "_").replace("/", "_")
    out_bin = out_dir / f"rel_error_{safe_method}_db{n_sample_db}_qr{n_sample_q}.bin"
    rel_error.astype(np.float32).tofile(out_bin)

    nbits_avg = int(round((min_b + max_b) / 2))
    summary = {
        "method": "VAQ",
        "dataset": dataset_name,
        "experiment_folder": str(out_dir),
        "nq": nq,
        "nb": nb,
        "nb_sample": int(n_sample_db),
        "nq_sample": int(n_sample_q),
        "dim": dim,
        "n_subquantizers": int(n_subspaces),
        "nbits": nbits_avg,
        "bits_per_vector": int(total_b),
        "min_bits": int(min_b),
        "max_bits": int(max_b),
        "variance": float(var_f),
        "vaq_method": method,
        "train_size": train_size,
        "train_time_s": float(train_time),
        "encoding_time_s": float(encoding_time),
        "distance_table_time_s": float(distance_table_time),
        "cdist_time_s": float(cdist_time),
        "adc_time_s": float(adc_time),
        "rel_error_mean": mean_rel,
        "rel_error_std": std_rel,
        "sample_mode": sample_mode,
        "train_path": train_path or "",
        "seed": seed,
        "skip_search": skip_search,
    }

    run_csv = out_dir / "summary.csv"
    with open(run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    csv_agg = data_root_p / results_dir / f"{dataset_name}_VAQ_adc_vs_exact_eval.csv"
    need_header = not csv_agg.exists() or csv_agg.stat().st_size == 0
    with open(csv_agg, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if need_header:
            w.writeheader()
        w.writerow(summary)

    metadata = {
        "summary": summary,
        "vaq_method_params": {
            "total_bits": int(total_b),
            "n_subspaces": int(n_subspaces),
            "min_bits": int(min_b),
            "max_bits": int(max_b),
            "variance": float(var_f),
            "search": method.split(",", 1)[1] if "," in method else "",
        },
        "inputs": {
            "dataset_path": dataset_path,
            "train_path": train_path or "",
            "query_path": query_path or "",
            "dim": int(dim),
        },
        "eval_config": {
            "train_size": int(train_size),
            "sample_db": int(sample_db),
            "sample_queries": int(sample_queries),
            "refine": refine,
            "k": int(k),
            "learn_ratio": float(learn_ratio),
            "seed": int(seed),
            "sample_mode": sample_mode,
            "skip_search": bool(skip_search),
            "reuse_artifact_dir": reuse_artifact_dir or "",
            "adc_eval_mode": "streaming_vectorized",
        },
        "artifacts": {
            "codes": codes_fp,
            "centroids": centroids_fp,
            "rel_error_bin": str(out_bin),
            "summary_csv": str(run_csv),
            "aggregate_csv": str(csv_agg),
        },
        "run_vaq_command": cmd,
    }
    metadata_json = out_dir / "metadata.json"
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"✅ Run summary: {run_csv}")
    print(f"✅ Run metadata: {metadata_json}")
    print(f"✅ Appended: {csv_agg}")
    print(f"✅ Rel. error bin: {out_bin}")


def main():
    p = argparse.ArgumentParser(description="VAQ ADC vs exact (PQ-aligned).")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--query_path", type=str, default=None)
    p.add_argument("--train_path", type=str, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--dataset_name", type=str, default="custom")
    p.add_argument("--data_root", type=str, default="/data/cpanourg/2-hdvc/")
    p.add_argument("--method", type=str, default="VAQ256m32min7max8var1,HEAP")
    p.add_argument("--total_bits", type=int, default=None)
    p.add_argument("--n_subspaces", type=int, default=None)
    p.add_argument("--min_bits", type=int, default=None)
    p.add_argument("--max_bits", type=int, default=None)
    p.add_argument("--variance", type=float, default=None)
    p.add_argument("--train_size", type=int, default=100_000)
    p.add_argument("--sample_db", type=int, default=10_000)
    p.add_argument("--sample_queries", type=int, default=1_000)
    p.add_argument("--results_dir", type=str, default="results/vaq")
    p.add_argument(
        "--vaq_binary",
        type=str,
        default=None,
        help="Path to run_vaq (default: lib/VAQ/build/examples/run_vaq under repo root)",
    )
    p.add_argument("--refine", type=str, default="100,200")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--learn_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sample_mode", type=str, default="first", choices=("first", "random"))
    p.add_argument(
        "--skip_search",
        action="store_true",
        help="Train/encode only: pass --skip-query to run_vaq (no C++ ANN/query). "
        "Python still loads queries for ADC vs exact rel. error and CSV output.",
    )
    p.add_argument(
        "--reuse_artifact_dir",
        type=str,
        default=None,
        help="Reuse an existing VAQ run directory containing codes.fvecs and centroids.fvecs; "
        "skips C++ train/encode and recomputes ADC/relative-error outputs.",
    )
    p.add_argument("--train_time_override", type=float, default=None)
    p.add_argument("--encoding_time_override", type=float, default=None)
    args = p.parse_args()
    run_vaq_eval(**vars(args))


if __name__ == "__main__":
    main()
