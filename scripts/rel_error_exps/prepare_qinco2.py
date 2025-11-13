#!/usr/bin/env python3
"""
Utility script to prepare custom datasets and launch QINCo2 experiments.

The script:
  1. Loads a database (and optional query) file using src.utils.load_dataset.
  2. Optionally downsamples the database to a requested train size.
  3. Writes the database and training samples to .fvecs files so that QINCo2 can
     consume them without additional preprocessing.
  4. Prints (and optionally executes) the QINCo2 training command with the
     chosen hyper-parameters.

Typical usage:
    python prepare_qinco2.py \
        --dataset_path /data/cpanourg/2-hdvc/data/deep1b/base.1B.fbin \
        --train_size 1000000 \
        --dataset_name deep_custom \
        --output_dir /data/cpanourg/2-hdvc/temp/qinco2 \
        --run

By default the script only prepares the data and prints the command. Pass
`--run` to actually execute QINCo2's `run.py task=train ...`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# Make project utilities available
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import load_dataset, write_fvecs  # noqa: E402


def sanitize_matrix(name: str, mat: np.ndarray) -> np.ndarray:
    """Remove rows containing NaN/Inf values and warn the user."""
    if mat.size == 0:
        raise ValueError(f"{name} is empty after loading.")

    mask = np.isfinite(mat).all(axis=1)
    invalid = (~mask).sum()
    if invalid > 0:
        print(
            f"⚠️  {name}: removing {invalid}/{len(mat)} rows containing NaN/Inf values "
            f"({invalid / len(mat) * 100:.3f}%)."
        )
        mat = mat[mask]

    return mat.astype(np.float32, copy=False)


def sample_training_set(
    db: np.ndarray, train_size: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split database into train subset and the remainder for evaluation."""
    if train_size <= 0:
        raise ValueError("train_size must be positive.")
    if train_size > len(db):
        raise ValueError(
            f"train_size={train_size} but dataset only has {len(db)} vectors."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(db))
    train_idx = perm[:train_size]
    remain_idx = perm[train_size:]

    train_set = db[train_idx]
    db_remainder = db[remain_idx] if len(remain_idx) else train_set.copy()

    return train_set, db_remainder


def build_qinco_command(
    qinco_root: Path,
    output_model: Path,
    train_fvecs: Path,
    db_fvecs: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Construct the Hydra command line for QINCo2 training."""
    hydra_args = [
        "python",
        str(qinco_root / "run.py"),
        f"task=train",
        f"output={output_model}",
        f"trainset={train_fvecs}",
        f"db={db_fvecs}",
        f"M={args.M}",
        f"K={args.K}",
        f"L={args.L}",
        f"dh={args.dh}",
        f"de={args.de}",
        f"A={args.A}",
        f"B={args.B}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"lr={args.lr}",
        f"wd={args.weight_decay}",
        f"grad_clip={args.grad_clip}",
    ]

    if args.ivf_K is not None:
        hydra_args.append(f"ivf_K={args.ivf_K}")
    if args.model_args:
        hydra_args.append(f"model_args={args.model_args}")
    if args.resume:
        hydra_args.append("resume=true")
    if args.tensorboard:
        hydra_args.append(f"tensorboard={args.tensorboard}")
    if args.ds_valset is not None:
        hydra_args.append(f"ds.valset={args.ds_valset}")
    if args.ds_loop is not None:
        hydra_args.append(f"ds.loop={args.ds_loop}")

    return hydra_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare datasets and launch QINCo2 training."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the database file (.fvecs/.fbin/.bin etc.).",
    )
    parser.add_argument(
        "--query_path",
        default=None,
        help="Optional path to a query file; will be converted to .fvecs if provided.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Dimensionality override if it cannot be inferred from the file header.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=1_000_000,
        help="Number of vectors to use for QINCo2 training (sampled from the dataset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for sampling the training subset.",
    )
    parser.add_argument(
        "--dataset_name",
        default="custom_dataset",
        help="Label used to name the generated files and results.",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/cpanourg/2-hdvc/temp/qinco2_prepared",
        help="Directory where the .fvecs files and model checkpoint will be stored.",
    )
    parser.add_argument(
        "--qinco_root",
        default="/home/cpanourg/projects/2-hdvc/lib/Qinco",
        help="Path to the QINCo2 repository containing run.py.",
    )
    parser.add_argument(
        "--model_args",
        default=None,
        help="Optional preset (e.g. qinco2-L) defined by the QINCo2 project.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training if the output checkpoint already exists.",
    )
    parser.add_argument(
        "--tensorboard",
        default=None,
        help="Optional path to a directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the QINCo2 training command after preparing the data.",
    )

    # Hyper-parameters with defaults aligned to the QINCo2-L configuration
    parser.add_argument("--M", type=int, default=8, help="Number of codebooks.")
    parser.add_argument("--K", type=int, default=256, help="Centroids per codebook.")
    parser.add_argument("--L", type=int, default=16, help="Residual blocks per step.")
    parser.add_argument("--dh", type=int, default=384, help="Hidden dimension.")
    parser.add_argument("--de", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--A", type=int, default=16, help="Pre-selected candidates.")
    parser.add_argument("--B", type=int, default=32, help="Beam width.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
        help="Number of epochs (paper default is 70 for large models).",
    )
    parser.add_argument("--batch", type=int, default=1024, help="Batch size per GPU.")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay coefficient."
    )
    parser.add_argument(
        "--grad_clip", type=float, default=0.1, help="Gradient clipping threshold."
    )
    parser.add_argument(
        "--ivf_K",
        type=int,
        default=None,
        help="Optional IVF coarse centroids (e.g. 1048576 for IVF-QINCo2).",
    )
    parser.add_argument(
        "--ds_valset",
        type=int,
        default=10_000,
        help="Size of the validation split extracted from the training set.",
    )
    parser.add_argument(
        "--ds_loop",
        type=int,
        default=None,
        help="Override epoch size (defaults to full training set if None).",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    query_path = Path(args.query_path).resolve() if args.query_path else None

    print("=" * 80)
    print("📦  Loading dataset")
    print("=" * 80)
    db, queries = load_dataset(
        str(dataset_path),
        str(query_path) if query_path else None,
        dim=args.dim,
    )

    db = sanitize_matrix("database", db)
    if queries is not None:
        queries = sanitize_matrix("queries", queries)

    print(f"Loaded database shape: {db.shape}")
    if queries is not None:
        print(f"Loaded queries shape: {queries.shape}")

    # Sample training subset
    train_set, db_eval = sample_training_set(db, args.train_size, args.seed)
    print(f"Training subset size: {len(train_set)}")
    print(f"Evaluation database size: {len(db_eval)}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_fvecs = output_dir / f"{args.dataset_name}_train.fvecs"
    db_fvecs = output_dir / f"{args.dataset_name}_db.fvecs"
    queries_fvecs: Optional[Path] = (
        output_dir / f"{args.dataset_name}_queries.fvecs" if queries is not None else None
    )

    print("=" * 80)
    print("💾  Writing .fvecs files")
    print("=" * 80)
    write_fvecs(train_fvecs, train_set)
    write_fvecs(db_fvecs, db_eval)
    print(f"✅ Wrote training set to {train_fvecs}")
    print(f"✅ Wrote evaluation DB to {db_fvecs}")

    if queries is not None and queries_fvecs is not None:
        write_fvecs(queries_fvecs, queries)
        print(f"✅ Wrote queries to {queries_fvecs}")

    qinco_root = Path(args.qinco_root).resolve()
    if not (qinco_root / "run.py").exists():
        raise FileNotFoundError(f"Could not find run.py under {qinco_root}")

    output_model = output_dir / f"{args.dataset_name}_qinco2.pt"
    command = build_qinco_command(qinco_root, output_model, train_fvecs, db_fvecs, args)

    print("=" * 80)
    print("🛠️  QINCo2 training command")
    print("=" * 80)
    command_str = " \\\n    ".join(command)
    print(command_str)

    if args.run:
        print("=" * 80)
        print("🚀  Launching QINCo2 training")
        print("=" * 80)
        subprocess.run(command, check=True)
    else:
        print("ℹ️  Pass --run to execute the command automatically.")


if __name__ == "__main__":
    main()

