#!/usr/bin/env python3
"""
Run QINCo2 training + evaluation for relative-error experiments.

Workflow:
    1. Load dataset / queries via src.utils.load_dataset (supports fbin/fvecs/etc).
    2. Clean NaN/Inf rows by reloading chunks (same strategy as other methods).
    3. Split database into train/test, write .fvecs files required by QINCo2.
    4. Launch QINCo2 training in-process via lib.Qinco tasks (records training time).
    5. Load trained model in inference mode, encode/decode a database sample,
       compute distances against query sample, and obtain relative-error stats.
    6. Persist codes + statistics and append experiment row to CSV (aligned with other methods).

The script is designed so run_qinco2.sh can simply vary hyperparameters/paths.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from hdvc_paths import get_results_root  # noqa: E402

from src.utils import (  # noqa: E402
    append_or_create_csv,
    ensure_dir,
    load_dataset,
    read_fbin,
    write_fvecs,
)

# QINCo2 tasks (instantiate directly instead of spawning subprocesses)
from lib.Qinco.qinco.qinco_tasks import (  # noqa: E402
    QincoConvertTask,
    QincoEvalTask,
    QincoTrainTask,
)
from lib.Qinco.qinco.search.search_tasks import (  # noqa: E402
    BuildIndexTask,
    EncodeDBTask,
    IVFTrainTask,
    SearchTask,
    TrainPairwiseDecoderTask,
)


EXPERIMENTS = {
    "train": QincoTrainTask,
    "eval_valset": QincoTrainTask,
    "eval": QincoEvalTask,
    "eval_time": QincoEvalTask,
    "convert": QincoConvertTask,
    "ivf_centroids": IVFTrainTask,
    "encode": EncodeDBTask,
    "build_index": BuildIndexTask,
    "train_pairwise_decoder": TrainPairwiseDecoderTask,
    "search": SearchTask,
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def clean_data_by_reloading(data: np.ndarray, filepath: str, initial_size: int) -> np.ndarray:
    """
    Remove rows with NaN/Inf by loading additional clean data from file.
    Mirrors the logic used in PQ/OPQ/LSQpp scripts to ensure consistency.
    """
    data_flat = data.view(np.float32).reshape(-1)
    has_nan = np.isnan(data_flat).any()
    has_inf = np.isinf(data_flat).any()
    if not has_nan and not has_inf:
        return data

    nan_rows = np.isnan(data).sum(axis=1) > 0
    inf_rows = np.isinf(data).sum(axis=1) > 0
    has_invalid = nan_rows | inf_rows
    invalid_count = has_invalid.sum()

    if invalid_count == 0:
        return data

    print(
        f"⚠️  Warning: Found {invalid_count} rows with NaN/Inf "
        f"({invalid_count / len(data) * 100:.2f}%). Loading clean replacements..."
    )

    valid_mask = ~has_invalid
    valid_data = data[valid_mask]
    remaining_invalid = invalid_count

    if filepath.endswith(".fbin"):
        replacement_start = initial_size
        chunk_size = max(remaining_invalid * 5, 100_000)
        max_attempts = 50
        attempt = 0
        clean_replacement_list = []

        while remaining_invalid > 0 and attempt < max_attempts:
            try:
                replacement = read_fbin(filepath, start_idx=replacement_start, chunk_size=chunk_size)
                if len(replacement) == 0:
                    break

                replacement_nan = np.isnan(replacement).sum(axis=1) == 0
                replacement_inf = np.isinf(replacement).sum(axis=1) == 0
                replacement_valid = replacement_nan & replacement_inf
                clean_replacements = replacement[replacement_valid]

                if len(clean_replacements) > 0:
                    n_to_add = min(remaining_invalid, len(clean_replacements))
                    clean_replacement_list.append(clean_replacements[:n_to_add])
                    remaining_invalid -= n_to_add
                    if remaining_invalid == 0:
                        print(f"✅ Replaced all {invalid_count} invalid rows with clean data")
                        break
                    replacement_start += len(replacement)
                else:
                    replacement_start += chunk_size

                attempt += 1
            except (ValueError, IndexError, IOError):
                print(f"⚠️  Reached end of file or error loading more data (attempt {attempt + 1}/{max_attempts})")
                break

        if clean_replacement_list:
            all_replacements = np.vstack(clean_replacement_list)
            valid_data = np.vstack([valid_data, all_replacements])

        if remaining_invalid > 0:
            print(f"⚠️  Warning: Could not replace {remaining_invalid} invalid rows. Removing them...")
            print(f"   Final data size: {len(valid_data)} (removed {remaining_invalid} invalid rows)")
    else:
        print(
            f"⚠️  Removing {invalid_count} invalid rows "
            "(file format doesn't support chunked reloading)"
        )

    return valid_data


def split_train_val_test(
    db: np.ndarray,
    train_size: int,
    val_ratio: float,
    test_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split database into train / val / test subsets."""
    if train_size >= len(db):
        raise ValueError(f"train_size {train_size} must be < total database size {len(db)}")

    max_test = len(db) - train_size
    test_size = min(test_size, max_test)
    if test_size <= 0:
        raise ValueError("test_size must be > 0 after accounting for train_size.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(db))

    train_idx = perm[:train_size]
    remaining = perm[train_size:]
    test_idx = remaining[:test_size]

    train_set = db[train_idx]
    test_set = db[test_idx]

    val_size = max(1, int(train_size * val_ratio))
    val_size = min(val_size, train_size // 2)  # cap so val <= 50% of train

    if val_size > 0:
        val_idx = rng.choice(train_size, size=val_size, replace=False)
        val_set = train_set[val_idx]
        train_mask = np.ones(train_size, dtype=bool)
        train_mask[val_idx] = False
        train_set = train_set[train_mask]
    else:
        val_set = np.empty((0, db.shape[1]), dtype=np.float32)

    return train_set, val_set, test_set


def clip_relative_error(adc_sample: np.ndarray, exact_sample: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Compute numerically stable relative error statistics."""
    epsilon = 1e-6
    max_safe_value = np.finfo(np.float32).max / 10.0

    exact_clip = np.clip(exact_sample, 0, max_safe_value).astype(np.float32)
    adc_clip = np.clip(adc_sample, 0, max_safe_value).astype(np.float32)

    diff_64 = np.abs(adc_clip.astype(np.float64) - exact_clip.astype(np.float64))
    denominator_64 = np.maximum(exact_clip.astype(np.float64), epsilon)
    rel_error = diff_64 / denominator_64
    rel_error = np.clip(rel_error, 0, 1e6)

    valid_mask = np.isfinite(rel_error) & (rel_error >= 0) & (rel_error < 1e6)
    if not valid_mask.any():
        return float("nan"), float("nan"), rel_error.astype(np.float32)

    rel_valid = rel_error[valid_mask].astype(np.float64)
    rel_valid = rel_valid[np.isfinite(rel_valid)]

    if len(rel_valid) == 0:
        return float("nan"), float("nan"), rel_error.astype(np.float32)

    mean_rel = float(rel_valid.mean())
    std_rel = float(rel_valid.std())
    return mean_rel, std_rel, rel_error.astype(np.float32)


def torch_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_experiment(args: argparse.Namespace) -> None:
    """Main orchestration."""
    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    query_path = Path(args.query_path).resolve() if args.query_path else None

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("=" * 80)
    print("📂 Loading dataset")
    print("=" * 80)

    db, qr = load_dataset(str(dataset_path), str(query_path) if query_path else None, dim=args.dim)
    db = db.astype(np.float32, copy=False)
    if qr is not None:
        qr = qr.astype(np.float32, copy=False)

    db = clean_data_by_reloading(db, str(dataset_path), initial_size=len(db))
    if qr is not None:
        source_for_queries = str(query_path) if query_path else str(dataset_path)
        qr = clean_data_by_reloading(qr, source_for_queries, initial_size=len(qr))

    print(f"Database shape: {db.shape}")
    if qr is not None:
        print(f"Query shape: {qr.shape}")

    if len(db) <= args.train_size:
        raise ValueError("Dataset must contain more rows than train_size.")

    # ------------------------------------------------------------------
    # Split & persist to fvecs
    # ------------------------------------------------------------------
    train_set, val_set, test_db = split_train_val_test(
        db,
        train_size=args.train_size,
        val_ratio=args.val_ratio,
        test_size=args.test_size,
        seed=args.seed,
    )

    val_size = len(val_set)
    print(f"Train size: {len(train_set)}, Val size: {val_size}, Test size: {len(test_db)}")

    temp_dir = Path(args.temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate train+val (QINCo2 expects single training file; val length provided via cfg.ds.valset)
    train_val = np.concatenate([train_set, val_set], axis=0) if val_size > 0 else train_set
    train_fvecs = temp_dir / f"{args.dataset_name}_trainval.fvecs"
    db_fvecs = temp_dir / f"{args.dataset_name}_db_eval.fvecs"
    write_fvecs(train_fvecs, train_val)
    write_fvecs(db_fvecs, test_db)
    print(f"✅ Wrote training data to {train_fvecs}")
    print(f"✅ Wrote evaluation DB to {db_fvecs}")

    if qr is not None:
        qr_fvecs = temp_dir / f"{args.dataset_name}_queries.fvecs"
        write_fvecs(qr_fvecs, qr)
        print(f"✅ Wrote queries to {qr_fvecs}")
    else:
        qr_fvecs = None

    # ------------------------------------------------------------------
    # Train QINCo2
    # ------------------------------------------------------------------
    cfg_path = Path(args.config_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot locate QINCo2 config: {cfg_path}")

    base_cfg = OmegaConf.load(str(cfg_path))

    model_output = Path(args.model_output_dir).resolve()
    model_output.mkdir(parents=True, exist_ok=True)
    model_path = model_output / f"{args.dataset_name}_M{args.M}_K{args.K}_{int(time.time())}.pt"

    train_cfg = OmegaConf.create(base_cfg)
    train_cfg.task = "train"
    train_cfg.output = str(model_path)
    train_cfg.db = str(db_fvecs)
    train_cfg.trainset = str(train_fvecs)
    train_cfg.ds.valset = val_size
    train_cfg.ds.trainset = len(train_val)
    if args.ds_loop is not None:
        train_cfg.ds.loop = args.ds_loop
    train_cfg.seed = args.seed
    train_cfg.resume = args.resume
    if args.verbose is not None:
        train_cfg.verbose = args.verbose
    train_cfg.cpu = not args.use_gpu

    # Hyperparameters
    train_cfg.M = args.M
    train_cfg.K = args.K
    train_cfg.L = args.L
    train_cfg.dh = args.dh
    train_cfg.de = args.de
    train_cfg.A = args.A
    train_cfg.B = args.B
    train_cfg.ivf_K = args.ivf_K
    train_cfg.epochs = args.epochs
    train_cfg.optimizer = args.optimizer
    train_cfg.lr = args.lr
    train_cfg.wd = args.weight_decay
    train_cfg.grad_clip = args.grad_clip
    train_cfg.batch = args.batch

    print("=" * 80)
    print("🚀 Training QINCo2")
    print("=" * 80)
    train_task = EXPERIENCE_to_task("train", train_cfg)
    train_start = time.perf_counter()
    train_task.run()
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_time = time.perf_counter() - train_start
    print(f"✅ QINCo2 training finished in {train_time:.2f}s")
    train_task.accelerator.end_training()

    # ------------------------------------------------------------------
    # Load trained model in inference mode
    # ------------------------------------------------------------------
    eval_cfg = OmegaConf.create(base_cfg)
    eval_cfg.task = "eval"
    eval_cfg.model = str(model_path)
    eval_cfg.db = str(db_fvecs)
    eval_cfg.seed = args.seed
    eval_cfg.cpu = not args.use_gpu
    eval_cfg.inference = True
    eval_cfg.ds.db = min(args.sample_db, len(test_db))
    eval_cfg.batch = args.batch
    # Keep hyperparameters consistent
    eval_cfg.M = args.M
    eval_cfg.K = args.K
    eval_cfg.L = args.L
    eval_cfg.dh = args.dh
    eval_cfg.de = args.de
    eval_cfg.A = args.A
    eval_cfg.B = args.B

    print("=" * 80)
    print("🧠 Loading trained model for inference")
    print("=" * 80)
    eval_task = EXPERIENCE_to_task("eval", eval_cfg)
    model = eval_task.qinco_model
    accelerator = eval_task.accelerator
    device = accelerator.device

    # ------------------------------------------------------------------
    # Prepare samples and compute codes
    # ------------------------------------------------------------------
    sample_db = min(args.sample_db, len(test_db))
    db_sample = test_db[:sample_db].astype(np.float32, copy=False)

    if qr is None:
        raise ValueError("Query set is required for relative-error computation.")

    sample_queries = min(args.sample_queries, len(qr))
    rng = np.random.default_rng(args.seed)
    qr_indices = rng.choice(len(qr), size=sample_queries, replace=False)
    qr_sample = qr[qr_indices].astype(np.float32, copy=False)

    db_tensor = torch.from_numpy(db_sample).to(device, dtype=torch.float32)
    encode_start = time.perf_counter()
    codes = model(db_tensor, step="encode")
    torch_sync_if_needed(device)
    encode_time = time.perf_counter() - encode_start

    decode_start = time.perf_counter()
    recon = model.decode(codes)
    torch_sync_if_needed(device)
    decode_time = time.perf_counter() - decode_start
    recon_np = recon.cpu().numpy().astype(np.float32)
    codes_np = codes.cpu().numpy()

    # ------------------------------------------------------------------
    # Distance computations
    # ------------------------------------------------------------------
    print("📐 Computing distances...")
    exact_start = time.perf_counter()
    exact_dist = cdist(qr_sample, db_sample, metric="sqeuclidean").astype(np.float32)
    exact_time = time.perf_counter() - exact_start

    approx_start = time.perf_counter()
    adc_dist = cdist(qr_sample, recon_np, metric="sqeuclidean").astype(np.float32)
    adc_time = time.perf_counter() - approx_start

    mean_rel, std_rel, rel_matrix = clip_relative_error(adc_dist, exact_dist)
    print(f"Mean relative error: {mean_rel:.4e}, std: {std_rel:.4e}")

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    results_dir = Path(args.data_root).resolve() / args.results_dir
    ensure_dir(results_dir)

    codes_dir = results_dir / "qinco2_codes"
    codes_dir.mkdir(parents=True, exist_ok=True)
    codes_fp = codes_dir / f"{args.dataset_name}_M{args.M}_K{args.K}_codes.npz"
    np.savez_compressed(codes_fp, codes=codes_np)
    print(f"💾 Codes saved to {codes_fp}")

    relerr_dir = results_dir / "qinco2_relerr"
    relerr_dir.mkdir(parents=True, exist_ok=True)
    relerr_fp = relerr_dir / (
        f"{args.dataset_name}_M{args.M}_K{args.K}_db{sample_db}_qr{sample_queries}.f32"
    )
    rel_matrix.astype(np.float32).tofile(relerr_fp)
    print(f"💾 Relative error matrix saved to {relerr_fp}")

    csv_fp = results_dir / f"{args.dataset_name}_QINCo2_adc_vs_exact_eval.csv"
    header = [
        "method",
        "dataset",
        "nq",
        "nb",
        "nb_sample",
        "dim",
        "n_subquantizers",
        "nbits",
        "bits_per_vector",
        "train_size",
        "sample_db",
        "sample_queries",
        "train_time_s",
        "encoding_time_s",
        "decoding_time_s",
        "cdist_time_s",
        "adc_time_s",
        "rel_error_mean",
        "rel_error_std",
        "model_path",
    ]
    rows = [[
        "QINCo2",
        args.dataset_name,
        len(qr),
        len(test_db),
        sample_db,
        db_sample.shape[1],
        args.M,
        int(round(math.log2(args.K))),
        args.M * int(round(math.log2(args.K))),
        args.train_size,
        sample_db,
        sample_queries,
        train_time,
        encode_time,
        decode_time,
        exact_time,
        adc_time,
        mean_rel,
        std_rel,
        str(model_path),
    ]]
    append_or_create_csv(str(csv_fp), header, rows)
    print(f"✅ Results appended to {csv_fp}")

    # Cleanup accelerator from eval task
    eval_task.accelerator.end_training()


def EXPERIENCE_to_task(task_name: str, cfg) -> QincoTrainTask:
    """Helper to instantiate QINCo2 Hydra task from config."""
    task_cls = EXPERIMENTS[task_name]
    return task_cls(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QINCo2 training and evaluation pipeline.")

    # Data paths
    parser.add_argument("--dataset_path", required=True, help="Path to database file (.fvecs/.fbin/.bin).")
    parser.add_argument("--query_path", default=None, help="Optional path to query file.")
    parser.add_argument("--dim", type=int, default=None, help="Dimensionality override if needed.")
    parser.add_argument("--dataset_name", default="custom", help="Dataset label for outputs.")
    parser.add_argument("--data_root", default="/data/cpanourg/2-hdvc", help="Root directory for results.")
    parser.add_argument("--results_dir", default="results/relerr", help="Subdirectory for CSV / artifacts.")
    parser.add_argument("--temp_dir", default="/data/cpanourg/2-hdvc/temp/qinco2", help="Where to store intermediate fvecs.")
    parser.add_argument(
        "--model_output_dir",
        default=str(get_results_root() / "qinco2" / "models"),
        help="Where to store trained checkpoints.",
    )

    # Experiment sizing
    parser.add_argument("--train_size", type=int, default=1_000_000, help="Training set size for QINCo2.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio (fraction of train).")
    parser.add_argument("--test_size", type=int, default=1_000_000, help="Size of evaluation DB (subset of remaining data).")
    parser.add_argument("--sample_db", type=int, default=10_000, help="Database sample size used for distance comparisons.")
    parser.add_argument("--sample_queries", type=int, default=1_000, help="Query sample size for distance comparisons.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for sampling.")

    # Hyperparameters (defaults match QINCo2 large config)
    parser.add_argument("--M", type=int, default=8, help="Number of codebooks.")
    parser.add_argument("--K", type=int, default=256, help="Centroids per codebook.")
    parser.add_argument("--L", type=int, default=16, help="Number of residual blocks per step.")
    parser.add_argument("--dh", type=int, default=384, help="Hidden dimension.")
    parser.add_argument("--de", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--A", type=int, default=16, help="Pre-selected candidates.")
    parser.add_argument("--B", type=int, default=32, help="Beam width.")
    parser.add_argument("--ivf_K", type=int, default=1048576, help="IVF centroids (if used).")
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size per accelerator process.")
    parser.add_argument("--optimizer", default="adamw", help="Optimizer (adam / adamw).")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--grad_clip", type=float, default=0.1, help="Gradient clipping value.")
    parser.add_argument("--ds_loop", type=int, default=None, help="Override for cfg.ds.loop (epoch size).")

    # Execution control
    parser.add_argument("--config_path", default="/home/cpanourg/projects/2-hdvc/lib/Qinco/config/qinco_cfg.yaml", help="Path to qinco_cfg.yaml.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU (default False).")
    parser.add_argument("--resume", action="store_true", help="Resume training if output checkpoint exists.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Force verbose logging during training.")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Silence batch-level logging from QINCo2.")
    parser.set_defaults(verbose=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

