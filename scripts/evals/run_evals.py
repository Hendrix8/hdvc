#!/usr/bin/env python3
"""
Run evaluation metrics on PQ/OPQ models from adc_vs_exact_eval CSV files.

Reads CSV with experiment_folder column, loads models from those paths,
loads database and query vectors (from config for each dataset),
and computes:
  - compression_rate: bits per compressed vector / bits per original vector
  - reconstruction_error: mean Euclidean distance between original and reconstructed
  - spearman: Spearman rank correlation between exact and ADC distance rankings
  - recall: Recall@1, Recall@10, Recall@100 (queries vs database)

Outputs separate CSV files per metric, e.g.:
  deep_PQ_compression_rate.csv
  deep_PQ_reconstruction_error.csv
  deep_PQ_spearman.csv
  deep_PQ_recall.csv

Usage:
  python run_evals.py --input_csv /path/to/deep_PQ_adc_vs_exact_eval.csv --output_dir /path/to/output
  python run_evals.py --input_csv /path/to/deep_PQ_adc_vs_exact_eval.csv --data_root /data/cpanourg/2-hdvc/data
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_dataset
from scripts.evals.config import DATASET_CONFIG, DATA_ROOT

# Temp dir for exact distance cache (datetime-stamped to avoid false loading)
TEMP_DIR = Path("/data/cpanourg/99-temp")


# =============================================================================
# Data loading (support fvecs, fbin, bvecs)
# =============================================================================

def read_bvecs(path: Path, max_vectors: int = None) -> np.ndarray:
    """Read .bvecs file (SIFT/BIGANN format: per vector, 4-byte dim + dim bytes uint8)."""
    with open(path, "rb") as f:
        # First vector gives us dim
        dim_b = f.read(4)
        if len(dim_b) < 4:
            return np.zeros((0, 0), dtype=np.float32)
        dim = int(np.frombuffer(dim_b, dtype=np.int32)[0])
        f.seek(0)
        data = []
        n_read = 0
        vec_size = 4 + dim  # 4 bytes dim + dim bytes
        while True:
            if max_vectors and n_read >= max_vectors:
                break
            chunk = f.read(vec_size * 10000)
            if len(chunk) < vec_size:
                break
            n = len(chunk) // vec_size
            arr = np.frombuffer(chunk[: n * vec_size], dtype=np.uint8)
            arr = arr.reshape(n, vec_size)[:, 4:].astype(np.float32)  # skip 4-byte dim per vector
            data.append(arr)
            n_read += n
        if not data:
            return np.zeros((0, dim), dtype=np.float32)
        out = np.vstack(data)
        if max_vectors and len(out) > max_vectors:
            out = out[:max_vectors]
        return out


def load_db_and_queries(dataset_path: str, query_path: str, dim: int, max_db: int = 1_000_000, max_queries: int = 10_000) -> tuple:
    """Load database and query vectors. Supports .fvecs, .fbin, .bvecs."""
    dataset_path = Path(dataset_path)
    query_path = Path(query_path)

    if dataset_path.suffix == ".bvecs":
        db = read_bvecs(dataset_path, max_vectors=max_db)
        qr = read_bvecs(query_path, max_vectors=max_queries)
    else:
        db, qr = load_dataset(
            str(dataset_path),
            str(query_path),
            dim=dim,
            db_chunk_size=max_db,
            qr_chunk_size=max_queries,
        )
    return db.astype(np.float32), qr.astype(np.float32)


def get_or_compute_exact_dists(
    dataset_name: str,
    db: np.ndarray,
    qr: np.ndarray,
    max_db: int,
    max_queries: int,
    run_timestamp: str,
    cache: dict,
) -> np.ndarray:
    """
    Get exact squared L2 distances (nq x nb) from cache or compute once and save.
    Uses first max_db from db and first max_queries from qr - same for all evals on this dataset.
    Saves to /data/cpanourg/99-temp with datetime stamp to avoid false loading.
    """
    key = (dataset_name, max_db, max_queries)
    if key in cache:
        return cache[key]

    nb = len(db)
    nq = min(len(qr), max_queries)
    cache_path = TEMP_DIR / f"exact_dists_{dataset_name}_{max_db}_{max_queries}_{run_timestamp}.npy"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        exact_sq = np.load(cache_path)
        tqdm.write(f"  Loaded exact dists from cache: {cache_path.name}")
    else:
        qr_sub = qr[:nq].astype(np.float32)
        db_f32 = db.astype(np.float32)
        exact_sq = np.zeros((nq, nb), dtype=np.float32)
        for i in tqdm(range(nq), desc="Exact dists (compute)", leave=False, unit="query"):
            exact_sq[i] = np.sum((db_f32 - qr_sub[i]) ** 2, axis=1)
        np.save(cache_path, exact_sq)
        tqdm.write(f"  Saved exact dists to cache: {cache_path.name}")

    cache[key] = exact_sq
    return exact_sq


# =============================================================================
# Model loading (PQ and OPQ)
# =============================================================================

def resolve_model_folder(
    experiment_folder: Path,
    n_subquantizers: int = None,
    nbits: int = None,
    train_size: int = None,
) -> Path:
    """
    Resolve experiment folder to one that contains pq_model.index.
    If the exact folder has no model, search parent for subq{M}_nbits{B}_train{tr}_*
    (same params, any timestamp) and use the first match with a model.
    """
    model_path = experiment_folder / "pq_model.index"
    if model_path.exists():
        return experiment_folder

    has_fallback = (
        n_subquantizers is not None and n_subquantizers > 0
        and nbits is not None and nbits > 0
        and train_size is not None and train_size > 0
    )
    if not has_fallback:
        raise FileNotFoundError(f"Model not found: {model_path} (no fallback params)")

    parent = experiment_folder.parent
    pattern = f"subq{n_subquantizers}_nbits{nbits}_train{train_size}_*"
    for folder in sorted(parent.glob(pattern)):
        if (folder / "pq_model.index").exists():
            return folder

    raise FileNotFoundError(f"Model not found: {model_path} (no match for {pattern} in {parent})")


def load_pq_model(
    experiment_folder: Path,
    n_subquantizers: int = None,
    nbits: int = None,
    train_size: int = None,
):
    """Load full IndexPQ from experiment folder. Returns the index (use index.pq for ProductQuantizer)."""
    folder = resolve_model_folder(experiment_folder, n_subquantizers, nbits, train_size)
    return faiss.read_index(str(folder / "pq_model.index"))


def load_opq_model(
    experiment_folder: Path,
    n_subquantizers: int = None,
    nbits: int = None,
    train_size: int = None,
):
    """Load OPQ model: returns full IndexPQ (use index.pq for ProductQuantizer)."""
    folder = resolve_model_folder(experiment_folder, n_subquantizers, nbits, train_size)
    model_path = folder / "pq_model.index"
    if not model_path.exists():
        for p in folder.rglob("pq_model.index"):
            model_path = p
            break
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {folder}")
    return faiss.read_index(str(model_path))


# =============================================================================
# Metrics
# =============================================================================

def compression_rate(row: pd.Series) -> float:
    """bits_compressed / bits_original = (M*nbits) / (dim*32)."""
    bits_compressed = row["n_subquantizers"] * row["nbits"]
    dim = row.get("dim", 0)
    if dim == 0:
        return np.nan
    bits_original = dim * 32
    return float(bits_compressed / bits_original)


def reconstruction_error(index, db: np.ndarray, max_samples: int = 10_000, chunk_size: int = 2000) -> float:
    """Mean Euclidean distance between original and reconstructed vectors."""
    n = min(max_samples, len(db))
    x = db[:n].astype(np.float32)
    pq = index.pq
    all_dists = []
    for start in tqdm(range(0, n, chunk_size), desc="Reconstruction", leave=False, unit="chunk"):
        end = min(start + chunk_size, n)
        chunk = x[start:end]
        codes = pq.compute_codes(chunk)
        recon = index.sa_decode(codes)
        all_dists.append(np.linalg.norm(chunk - recon, axis=1))
    dists = np.concatenate(all_dists)
    return float(np.mean(dists))


def spearman_faiss(index, db: np.ndarray, qr: np.ndarray, max_queries: int = 1000, exact_sq: np.ndarray = None) -> float:
    """Spearman: use IndexPQ.search to get ADC distances, compare to exact. Index must already have db added.
    If exact_sq is provided (nq x nb), use it instead of recomputing."""
    from scipy.stats import spearmanr

    nb = len(db)
    nq = min(len(qr), max_queries)
    qr = qr[:nq].astype(np.float32)

    if exact_sq is None:
        exact_sq = np.zeros((nq, nb), dtype=np.float32)
        for i in tqdm(range(nq), desc="Spearman (exact dists)", leave=False, unit="query"):
            exact_sq[i] = np.sum((db.astype(np.float32) - qr[i]) ** 2, axis=1)
    else:
        exact_sq = exact_sq[:nq]  # ensure we only use nq rows

    # ADC distances from FAISS (returns squared L2)
    adc_sq, _ = index.search(qr, nb)

    correlations = []
    for i in tqdm(range(nq), desc="Spearman (correl)", leave=False, unit="query"):
        r, _ = spearmanr(exact_sq[i], adc_sq[i])
        if not np.isnan(r):
            correlations.append(r)
    return float(np.mean(correlations)) if correlations else np.nan


def recall_at_k(index, db: np.ndarray, qr: np.ndarray, k_values: list = [1, 10, 100], max_queries: int = 1000, exact_sq: np.ndarray = None) -> dict:
    """Recall@k: for each query, |true_k ∩ pred_k| / k. Index must already have db added.
    If exact_sq is provided (nq x nb), derive exact_nn from it instead of recomputing."""
    nb = len(db)
    nq = min(len(qr), max_queries)
    qr = qr[:nq].astype(np.float32)

    K = max(k_values)

    if exact_sq is None:
        exact_sq = np.zeros((nq, nb), dtype=np.float32)
        for i in tqdm(range(nq), desc="Recall (exact)", leave=False, unit="query"):
            exact_sq[i] = np.sum((db.astype(np.float32) - qr[i]) ** 2, axis=1)
    else:
        exact_sq = exact_sq[:nq]

    # Exact k-NN from exact_sq
    exact_nn = np.zeros((nq, K), dtype=np.int64)
    for i in tqdm(range(nq), desc="Recall (exact nn)", leave=False, unit="query"):
        exact_nn[i] = np.argsort(exact_sq[i])[:K]

    # ADC k-NN
    _, adc_nn = index.search(qr, K)

    result = {}
    for k in k_values:
        hits = 0
        for i in range(nq):
            true_k = set(exact_nn[i, :k])
            pred_k = set(adc_nn[i, :k])
            hits += len(true_k & pred_k) / k
        result[f"recall_{k}"] = hits / nq
    return result


# =============================================================================
# Main processing
# =============================================================================

VALID_EVAL_MEASURES = ["compression_rate", "reconstruction_error", "spearman", "recall"]


def run_evals(
    input_csv: Path,
    output_dir: Path,
    data_root: Path = None,
    max_db: int = 1_000_000,
    max_queries: int = 10_000,
    max_rec_samples: int = 10_000,
    eval_measures: list = None,
):
    """Process CSV, run evals, write per-metric output CSVs.

    eval_measures: list of measure names to compute (e.g. ["compression_rate", "recall"]).
        None = all measures. Valid: compression_rate, reconstruction_error, spearman, recall.
    """
    if data_root is None:
        data_root = Path(DATA_ROOT)

    if eval_measures is None:
        eval_measures = VALID_EVAL_MEASURES
    else:
        invalid = [m for m in eval_measures if m not in VALID_EVAL_MEASURES]
        if invalid:
            raise ValueError(f"Invalid eval_measures: {invalid}. Valid: {VALID_EVAL_MEASURES}")

    df = pd.read_csv(input_csv)
    if "experiment_folder" not in df.columns:
        raise ValueError("CSV must have 'experiment_folder' column")
    if "dataset" not in df.columns:
        raise ValueError("CSV must have 'dataset' column")

    # Infer method and base name from input filename: deep_PQ_adc_vs_exact_eval.csv -> deep_PQ
    stem = input_csv.stem  # e.g. deep_PQ_adc_vs_exact_eval
    parts = stem.split("_")
    if len(parts) >= 2:
        base_name = f"{parts[0]}_{parts[1]}"
    else:
        base_name = stem.replace("_adc_vs_exact_eval", "")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exact_dists_cache = {}

    # Initialize output dataframes (copy input, add metric columns and time columns)
    df_comp = df.copy()
    if "compression_rate" in eval_measures:
        df_comp["compression_rate"] = np.nan
        df_comp["compression_rate_time_s"] = np.nan

    df_rec = df.copy()
    if "reconstruction_error" in eval_measures:
        df_rec["reconstruction_error"] = np.nan
        df_rec["reconstruction_error_time_s"] = np.nan

    df_spear = df.copy()
    if "spearman" in eval_measures:
        df_spear["spearman"] = np.nan
        df_spear["spearman_time_s"] = np.nan

    df_recall = df.copy()
    if "recall" in eval_measures:
        for k in [1, 10, 100]:
            df_recall[f"recall_{k}"] = np.nan
        df_recall["recall_time_s"] = np.nan

    needs_model = any(
        m in eval_measures for m in ("reconstruction_error", "spearman", "recall")
    )

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Evaluating",
        unit="row",
        dynamic_ncols=True,
    ):
        exp_folder = Path(row["experiment_folder"])
        dataset_name = row["dataset"]
        method = row.get("method", "PQ")
        tqdm.write(f"{dataset_name} {method} {exp_folder.name}")

        if dataset_name not in DATASET_CONFIG:
            tqdm.write(f"  ⚠️  Unknown dataset: {dataset_name}, skipping")
            continue

        dataset_path, query_path = DATASET_CONFIG[dataset_name]
        dataset_path = Path(dataset_path)
        query_path = Path(query_path)

        # Load model (fallback to first subq_M_nbits_B_train_T_* folder if exact path has no model)
        n_subq = int(row.get("n_subquantizers", 0))
        nbits_v = int(row.get("nbits", 0))
        tr_size = int(row.get("train_size", 0))

        index = None
        db = None
        qr = None
        if needs_model:
            try:
                if method == "OPQ":
                    index = load_opq_model(exp_folder, n_subq, nbits_v, tr_size)
                else:
                    index = load_pq_model(exp_folder, n_subq, nbits_v, tr_size)
            except FileNotFoundError as e:
                tqdm.write(f"  ⚠️  {e}")
                continue

            dim = index.d
            try:
                db, qr = load_db_and_queries(str(dataset_path), str(query_path), dim, max_db, max_queries)
            except Exception as e:
                tqdm.write(f"  ❌ Load data: {e}")
                continue
            tqdm.write(f"  db={len(db)}, qr={len(qr)}, dim={dim}")

        # 1. Compression rate (no data needed)
        if "compression_rate" in eval_measures:
            t0 = time.perf_counter()
            df_comp.at[idx, "compression_rate"] = compression_rate(row)
            df_comp.at[idx, "compression_rate_time_s"] = time.perf_counter() - t0

        # 2. Reconstruction error
        if "reconstruction_error" in eval_measures:
            try:
                t0 = time.perf_counter()
                rec_err = reconstruction_error(index, db, max_samples=max_rec_samples)
                df_rec.at[idx, "reconstruction_error"] = rec_err
                df_rec.at[idx, "reconstruction_error_time_s"] = time.perf_counter() - t0
            except Exception as e:
                tqdm.write(f"  ❌ Reconstruction error: {e}")

        # 3. Spearman & 4. Recall: add db once (index is trained), then run both
        if "spearman" in eval_measures or "recall" in eval_measures:
            db_f32 = db.astype(np.float32)
            index.add(db_f32)

        # Get or compute exact distances once per dataset (cached to temp file)
        exact_sq = None
        if "spearman" in eval_measures or "recall" in eval_measures:
            exact_sq = get_or_compute_exact_dists(
                dataset_name, db, qr, max_db, max_queries, run_timestamp, exact_dists_cache
            )

        if "spearman" in eval_measures:
            try:
                t0 = time.perf_counter()
                sp = spearman_faiss(index, db, qr, max_queries=min(500, len(qr)), exact_sq=exact_sq)
                df_spear.at[idx, "spearman"] = sp
                df_spear.at[idx, "spearman_time_s"] = time.perf_counter() - t0
            except Exception as e:
                tqdm.write(f"  ❌ Spearman: {e}")

        if "recall" in eval_measures:
            try:
                t0 = time.perf_counter()
                rec = recall_at_k(index, db, qr, k_values=[1, 10, 100], max_queries=min(1000, len(qr)), exact_sq=exact_sq)
                df_recall.at[idx, "recall_time_s"] = time.perf_counter() - t0
                for k, v in rec.items():
                    df_recall.at[idx, k] = v
            except Exception as e:
                tqdm.write(f"  ❌ Recall: {e}")

    # Save outputs (only for requested measures)
    saved = []
    if "compression_rate" in eval_measures:
        df_comp.to_csv(output_dir / f"{base_name}_compression_rate.csv", index=False)
        saved.append(f"{base_name}_compression_rate.csv")
    if "reconstruction_error" in eval_measures:
        df_rec.to_csv(output_dir / f"{base_name}_reconstruction_error.csv", index=False)
        saved.append(f"{base_name}_reconstruction_error.csv")
    if "spearman" in eval_measures:
        df_spear.to_csv(output_dir / f"{base_name}_spearman.csv", index=False)
        saved.append(f"{base_name}_spearman.csv")
    if "recall" in eval_measures:
        df_recall.to_csv(output_dir / f"{base_name}_recall.csv", index=False)
        saved.append(f"{base_name}_recall.csv")

    print(f"\n✅ Saved to {output_dir}/")
    for f in saved:
        print(f"   - {f}")


def main():
    parser = argparse.ArgumentParser(description="Run evals on PQ/OPQ models from CSV")
    parser.add_argument("--input_csv", type=Path, default="/data/cpanourg/2-hdvc/results/relerr_cpp/bigann_PQ_adc_vs_exact_eval.csv", help="Path to adc_vs_exact_eval CSV")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory (default: same as input)")
    parser.add_argument("--data_root", type=Path, default='/data/cpanourg/2-hdvc/data', help="Data root for dataset paths")
    parser.add_argument("--max_db", type=int, default=1_000_000, help="Max database vectors to load")
    parser.add_argument("--max_queries", type=int, default=1_000, help="Max query vectors")
    parser.add_argument("--max_rec_samples", type=int, default=10_000, help="Max samples for reconstruction error")
    parser.add_argument(
        "--eval_measures", type=str, nargs="*", default=['compression_rate', 'reconstruction_error', 'spearman', 'recall'], # ['compression_rate', 'reconstruction_error', 'spearman', 'recall']
        help=f"Measures to compute (default: all). Options: {', '.join(VALID_EVAL_MEASURES)}. Example: --eval_measures compression_rate recall")
    
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_csv.parent
    run_evals(
        input_csv=args.input_csv,
        output_dir=output_dir,
        data_root=args.data_root,
        max_db=args.max_db,
        max_queries=args.max_queries,
        max_rec_samples=args.max_rec_samples,
        eval_measures=args.eval_measures,
    )


if __name__ == "__main__":
    main()
