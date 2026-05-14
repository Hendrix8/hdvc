#!/usr/bin/env python3
"""LSQ++ ADC vs exact eval over a *pre-trained* grid (purpose=train_only_save_models).

Walks a grid directory like:

    <grid_root>/<dataset>/M{M}_nbits{nbits}_train{train_size}/
        ├── COMPLETE
        ├── codebooks.npz
        ├── index_lsqpp.faiss     (NOT used — see note below)
        └── metadata.json

and, for each combo, rebuilds a fresh `LocalSearchQuantizer` from the saved
codebooks and uses it to encode the test set. We deliberately bypass
`faiss.read_index(index_lsqpp.faiss)` because indexes serialized with faiss 1.8
trigger glibc heap corruption when re-loaded in newer faiss builds (1.12 / 1.13).
The codebooks tensor alone fully determines the LSQ encoding, so this gives
results equivalent to `sa_encode`/`compute_codes` on the original index.

For each combo we compute:
  - relative error (ADC vs exact L2^2, PQ-style),
  - Spearman rank correlation,
  - reconstruction error,
  - LUT/ADC timings (CPU process time, same spirit as ``time.process_time`` /
    FAISS ``pq_adc_timing`` / ``scripts/evals/measure_adc_cpu_time.py`` — not wall clock).
    Median over ``--adc_timing_repeats`` rounds: ``distance_table_time_s`` = alpha +
    dot tables; ``adc_time_s`` = numba ADC scan.
appending one row per combo to:
  - <out_dir>/<dataset>_LSQpp_adc_vs_exact_eval.csv
  - <out_dir>/<dataset>_LSQpp_reconstruction_error.csv

These are the CSVs consumed by `scripts/figure_generators/lsqpp_relerr_plots.ipynb`.

Recommended environment:
    /home/cpanourg/.miniconda3/envs/tsfm/bin/python
(faiss 1.13.x + numba 0.61.x + numpy 1.26.x + scipy 1.15.x).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import statistics
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lsqpp.data_io import load_vectors_prefix
from lsqpp.lsq_adc import (
    lsq_alpha_from_codes,
    lsq_distances_batch_numba,
    lsq_dot_tables_vectorized,
)
from lsqpp.metrics import (
    compute_lsq_reconstruction_error,
    compute_rel_error_pq_style,
    exact_distances_sqeuclidean,
    mean_spearman_rank,
)


DEFAULT_DATASETS = ("deep", "bigann", "gist", "msmarco", "openai")
COMBO_DIR_RE = re.compile(r"^M(?P<M>\d+)_nbits(?P<nbits>\d+)_train(?P<train>\d+)$")

EVAL_COLUMNS = [
    "method",
    "dataset",
    "experiment_folder",
    "nq",
    "nb",
    "nb_sample",
    "nq_sample",
    "pair_count",
    "dim",
    "n_subquantizers",
    "nbits",
    "bits_per_vector",
    "train_size",
    "train_time_s",
    "encoding_time_s",
    "distance_table_time_s",
    "cdist_time_s",
    "adc_time_s",
    "per_pair_adc_time_ns",
    "rel_error_mean",
    "rel_error_std",
    "spearman",
    "reconstruction_error",
    "sample_mode",
    "train_path",
    "seed",
    "evaluated_at",
]

RECON_COLUMNS = [
    "method",
    "dataset",
    "n_subquantizers",
    "nbits",
    "bits_per_vector",
    "train_size",
    "reconstruction_error",
    "nb_sample",
    "evaluated_at",
]


def dataset_paths(dataset: str, data_root: Path) -> dict:
    """Filesystem layout for each benchmark; mirrors the notebook's _dataset_paths,
    but rooted at the location where the data actually lives on this machine."""
    if dataset == "deep":
        return {
            "dataset_path": data_root / "deep1b/dataset/test_1m.bin",
            "train_path": data_root / "deep1b/dataset/learn_100m.bin",
            "query_path": data_root / "deep1b/dataset/query_10k.bin",
            "dim": 96,
        }
    if dataset == "bigann":
        return {
            "dataset_path": data_root / "bigann/SIFT1M/bigann_base.bvecs",
            "train_path": data_root / "bigann/SIFT1M/bigann_learn.bvecs",
            "query_path": data_root / "bigann/SIFT1M/bigann_query.bvecs",
            "dim": 128,
        }
    if dataset == "gist":
        return {
            "dataset_path": data_root / "gist/gist_base.fvecs",
            "train_path": data_root / "gist/gist_learn.fvecs",
            "query_path": data_root / "gist/gist_query.fvecs",
            "dim": 960,
        }
    if dataset == "msmarco":
        return {
            "dataset_path": data_root / "msmarco/fvecs/base1m.fvecs",
            "train_path": data_root / "msmarco/fvecs/train1m.fvecs",
            "query_path": data_root / "msmarco/fvecs/query10k.fvecs",
            "dim": 1024,
        }
    if dataset == "openai":
        return {
            "dataset_path": data_root / "openai/fvecs/openai_base1m.fvecs",
            "train_path": data_root / "openai/fvecs/openai_train1m.fvecs",
            "query_path": data_root / "openai/fvecs/openai_query10k.fvecs",
            "dim": 1536,
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def list_combos(grid_root: Path, dataset: str) -> list[tuple[int, int, int, Path]]:
    """Return [(M, nbits, train_size, combo_dir), ...] sorted by (M, nbits, train_size)."""
    ds_root = grid_root / dataset
    if not ds_root.is_dir():
        return []
    out: list[tuple[int, int, int, Path]] = []
    for sub in ds_root.iterdir():
        if not sub.is_dir():
            continue
        m = COMBO_DIR_RE.match(sub.name)
        if not m:
            continue
        out.append((int(m["M"]), int(m["nbits"]), int(m["train"]), sub))
    out.sort()
    return out


def unpack_lsq_codes(codes_packed: np.ndarray, M: int, nbits: int) -> np.ndarray:
    if not hasattr(faiss, "unpack_bitstrings"):
        raise RuntimeError(
            "faiss.unpack_bitstrings is missing; use a full faiss-cpu build with extra_wrappers."
        )
    b = np.ascontiguousarray(codes_packed, dtype=np.uint8)
    unpacked = faiss.unpack_bitstrings(b, M, nbits)
    return np.asarray(unpacked, dtype=np.int64)


def build_lsq_from_codebooks(codebooks: np.ndarray, search_type: int) -> "faiss.LocalSearchQuantizer":
    """Build a ready-to-encode LSQ from saved codebooks.

    Avoids `faiss.read_index` which corrupts the heap when re-loading models
    serialized by older faiss versions.
    """
    M, ksub, d = codebooks.shape
    if (ksub & (ksub - 1)) != 0:
        raise ValueError(f"ksub={ksub} is not a power of two")
    nbits = int(np.log2(ksub))
    lsq = faiss.LocalSearchQuantizer(int(d), int(M), nbits, search_type)
    # Default `nperts` can exceed M for small M (e.g. M=2); clamp to satisfy faiss assert.
    lsq.nperts = min(int(lsq.nperts), int(M))
    flat = np.ascontiguousarray(codebooks.reshape(-1), dtype=np.float32)
    faiss.copy_array_to_vector(flat, lsq.codebooks)
    lsq.is_trained = True
    lsq.compute_codebook_tables()
    return lsq


def measure_lut_adc_cpu_repeated(
    *,
    codebooks: np.ndarray,
    qr_sample: np.ndarray,
    codes_ix: np.ndarray,
    qnorms: np.ndarray,
    repeats: int,
    warmup_round: bool,
) -> tuple[float, float, np.ndarray]:
    """Return (distance_table_cpu_s, adc_scan_cpu_s, adc_distances) using process CPU time.

    ``distance_table`` here matches the eval split: ``lsq_alpha_from_codes`` +
    ``lsq_dot_tables_vectorized``. ``adc_scan`` is ``lsq_distances_batch_numba``.

    ``repeats`` timed rounds; each round rebuilds LUTs then runs ADC (PQ-style
    separation). Warmup (if True) runs one untimed full pass so JIT and caches
    are not inside the first sample. Reported values are the *median* CPU
    seconds over timed rounds for each phase independently. ``adc_distances``
    is taken from the last timed round (numerically identical across rounds).
    """
    r = max(1, int(repeats))
    lut_list: list[float] = []
    adc_list: list[float] = []

    def one_round() -> tuple[float, float, np.ndarray]:
        t0 = time.process_time()
        alpha = lsq_alpha_from_codes(codebooks, codes_ix)
        dot_tables = lsq_dot_tables_vectorized(qr_sample, codebooks)
        t1 = time.process_time()
        lut = t1 - t0
        t0 = time.process_time()
        adc_sample = lsq_distances_batch_numba(dot_tables, codes_ix, alpha, qnorms)
        t1 = time.process_time()
        adc = t1 - t0
        return lut, adc, adc_sample

    if warmup_round:
        _, _, _ = one_round()
    last_adc: np.ndarray | None = None
    for _ in range(r):
        lut, adc, adc_mat = one_round()
        lut_list.append(lut)
        adc_list.append(adc)
        last_adc = adc_mat

    med_lut = float(statistics.median(lut_list))
    med_adc = float(statistics.median(adc_list))
    assert last_adc is not None
    return med_lut, med_adc, last_adc


def numba_warmup(M: int = 2, nbits: int = 8, dim: int = 8) -> None:
    """Run a tiny ADC call so the numba JIT cost is paid up-front (not inside the timed region)."""
    ksub = 1 << nbits
    codebooks = np.random.RandomState(0).randn(M, ksub, dim).astype(np.float32)
    queries = np.random.RandomState(1).randn(2, dim).astype(np.float32)
    codes = np.random.RandomState(2).randint(0, ksub, size=(3, M)).astype(np.int64)
    alpha = lsq_alpha_from_codes(codebooks, codes)
    dot_tables = lsq_dot_tables_vectorized(queries, codebooks)
    qnorms = np.einsum("ij,ij->i", queries, queries).astype(np.float32)
    _ = lsq_distances_batch_numba(dot_tables, codes, alpha, qnorms)


def load_existing_keys(csv_path: Path) -> set[tuple[int, int, int]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    keys: set[tuple[int, int, int]] = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                keys.add(
                    (
                        int(float(row["n_subquantizers"])),
                        int(float(row["nbits"])),
                        int(float(row["train_size"])),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return keys


def append_row(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def evaluate_dataset(
    *,
    dataset: str,
    grid_root: Path,
    data_root: Path,
    out_dir: Path,
    sample_db: int,
    sample_queries: int,
    seed: int,
    skip_existing: bool,
    only_combos: set[tuple[int, int, int]] | None,
    adc_timing_repeats: int,
    adc_timing_warmup: bool,
    overwrite_csv: bool,
) -> None:
    combos = list_combos(grid_root, dataset)
    if not combos:
        print(f"[{dataset}] no combo directories under {grid_root / dataset}")
        return

    paths = dataset_paths(dataset, data_root)
    dim_expected = int(paths["dim"])
    eval_csv = out_dir / f"{dataset}_LSQpp_adc_vs_exact_eval.csv"
    recon_csv = out_dir / f"{dataset}_LSQpp_reconstruction_error.csv"
    if overwrite_csv:
        for p in (eval_csv, recon_csv):
            if p.is_file():
                p.unlink()
                print(f"[{dataset}] removed {p.name} (--overwrite_csv)")
    existing_eval = load_existing_keys(eval_csv) if skip_existing else set()
    existing_recon = load_existing_keys(recon_csv) if skip_existing else set()

    print(
        f"[{dataset}] {len(combos)} combos; "
        f"already in eval CSV: {len(existing_eval)}; already in recon CSV: {len(existing_recon)}"
    )

    np.random.seed(seed)
    test_db_sample = load_vectors_prefix(
        str(paths["dataset_path"]), dim_expected, sample_db, 0
    ).astype(np.float32, copy=False)
    qr_sample = load_vectors_prefix(
        str(paths["query_path"]), dim_expected, sample_queries, 0
    ).astype(np.float32, copy=False)
    n_sample_db = int(test_db_sample.shape[0])
    n_sample_q = int(qr_sample.shape[0])
    if n_sample_db == 0 or n_sample_q == 0:
        print(f"[{dataset}] empty sample(s) (db={n_sample_db}, q={n_sample_q}); skipping")
        return
    pair_count = max(1, n_sample_db * n_sample_q)

    qnorms = np.einsum("ij,ij->i", qr_sample, qr_sample).astype(np.float32)
    t0 = time.process_time()
    exact_sample = exact_distances_sqeuclidean(qr_sample, test_db_sample)
    cdist_time = time.process_time() - t0
    print(
        f"[{dataset}] loaded test_db_sample={test_db_sample.shape}, qr_sample={qr_sample.shape}; "
        f"cdist precomputed in {cdist_time:.2f}s"
    )

    for M, nbits, train_size, combo_dir in combos:
        key = (M, nbits, train_size)
        if only_combos is not None and key not in only_combos:
            continue
        if not (combo_dir / "COMPLETE").exists():
            print(f"[{dataset}] skip {combo_dir.name} (no COMPLETE marker)")
            continue
        eval_needed = key not in existing_eval
        recon_needed = key not in existing_recon
        if not eval_needed and not recon_needed:
            print(f"[{dataset}] skip {combo_dir.name} (already in both CSVs)")
            continue

        try:
            with open(combo_dir / "metadata.json") as f:
                meta = json.load(f)
        except FileNotFoundError:
            print(f"[{dataset}] skip {combo_dir.name} (missing metadata.json)")
            continue

        dim_meta = int(meta.get("dim", dim_expected))
        if dim_meta != dim_expected:
            print(
                f"[{dataset}] skip {combo_dir.name}: metadata dim={dim_meta} "
                f"!= dataset dim={dim_expected}"
            )
            continue
        train_time_meta = float(meta.get("train_time_s", float("nan")))
        train_path_meta = str(meta.get("train_path", ""))

        cb_path = combo_dir / "codebooks.npz"
        if not cb_path.exists():
            print(f"[{dataset}] skip {combo_dir.name} (missing codebooks.npz)")
            continue
        with np.load(cb_path) as data:
            codebooks = np.ascontiguousarray(data["codebooks"], dtype=np.float32)
            search_type = int(data["search_type"]) if "search_type" in data.files else \
                int(faiss.LocalSearchQuantizer.ST_norm_qint8)
        if codebooks.shape != (M, 1 << nbits, dim_expected):
            print(
                f"[{dataset}] skip {combo_dir.name}: codebooks shape {codebooks.shape} "
                f"!= ({M},{1<<nbits},{dim_expected})"
            )
            continue

        try:
            lsq = build_lsq_from_codebooks(codebooks, search_type)
        except Exception as exc:
            print(f"[{dataset}] FAIL build_lsq {combo_dir.name}: {exc}")
            continue

        t0 = time.process_time()
        codes_u8 = np.asarray(lsq.compute_codes(test_db_sample), dtype=np.uint8)
        encoding_time = time.process_time() - t0
        try:
            codes_ix = unpack_lsq_codes(codes_u8, M, nbits)
        except Exception as exc:
            print(f"[{dataset}] FAIL unpack {combo_dir.name}: {exc}")
            continue
        if codes_ix.shape != (n_sample_db, M):
            print(
                f"[{dataset}] skip {combo_dir.name}: unpacked codes shape "
                f"{codes_ix.shape} != ({n_sample_db}, {M})"
            )
            continue

        distance_table_time, adc_time, adc_sample = measure_lut_adc_cpu_repeated(
            codebooks=codebooks,
            qr_sample=qr_sample,
            codes_ix=codes_ix,
            qnorms=qnorms,
            repeats=adc_timing_repeats,
            warmup_round=adc_timing_warmup,
        )

        rel_error, mean_rel, std_rel = compute_rel_error_pq_style(adc_sample, exact_sample)
        spearman = mean_spearman_rank(adc_sample, exact_sample)
        recon_error = compute_lsq_reconstruction_error(test_db_sample, codes_ix, codebooks)

        bits_per_vector = int(M * nbits)
        per_pair_adc_time_ns = ((distance_table_time + adc_time) / pair_count) * 1e9
        now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        row = {
            "method": "LSQpp",
            "dataset": dataset,
            "experiment_folder": str(combo_dir),
            "nq": n_sample_q,
            "nb": n_sample_db,
            "nb_sample": n_sample_db,
            "nq_sample": n_sample_q,
            "pair_count": pair_count,
            "dim": dim_expected,
            "n_subquantizers": M,
            "nbits": nbits,
            "bits_per_vector": bits_per_vector,
            "train_size": train_size,
            "train_time_s": train_time_meta,
            "encoding_time_s": float(encoding_time),
            "distance_table_time_s": float(distance_table_time),
            "cdist_time_s": float(cdist_time),
            "adc_time_s": float(adc_time),
            "per_pair_adc_time_ns": float(per_pair_adc_time_ns),
            "rel_error_mean": float(mean_rel),
            "rel_error_std": float(std_rel),
            "spearman": float(spearman),
            "reconstruction_error": float(recon_error),
            "sample_mode": "first",
            "train_path": train_path_meta,
            "seed": seed,
            "evaluated_at": now_iso,
        }
        if eval_needed:
            append_row(eval_csv, row, EVAL_COLUMNS)
            existing_eval.add(key)
        if recon_needed:
            append_row(recon_csv, row, RECON_COLUMNS)
            existing_recon.add(key)

        print(
            f"[{dataset}] M={M:>2} nbits={nbits:>2} train={train_size:>6}  "
            f"rel={mean_rel:.4f}±{std_rel:.4f}  spearman={spearman:.3f}  "
            f"recon={recon_error:.4f}  adc/pair={per_pair_adc_time_ns:.1f}ns  "
            f"enc={encoding_time:.2f}s lut={distance_table_time:.2f}s adc={adc_time:.2f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid_root", type=Path, required=True,
                        help="Top-level directory of the saved-models grid")
    parser.add_argument("--data_root", type=Path,
                        default=Path("/mnthdd/cpanourg/2-hdvc/data"),
                        help="Directory containing per-dataset sub-folders (deep1b, bigann, ...)")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Where to write *_LSQpp_adc_vs_exact_eval.csv (default: --grid_root)")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--sample_db", type=int, default=10_000)
    parser.add_argument("--sample_queries", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Don't re-evaluate combos already present in the output CSVs")
    parser.add_argument(
        "--overwrite_csv",
        action="store_true",
        help="Delete this dataset's eval+recon CSVs in --out_dir before writing (fresh run, no duplicate rows)",
    )
    parser.add_argument(
        "--adc_timing_repeats",
        type=int,
        default=3,
        help="Number of timed LUT+ADC rounds; median CPU time per phase is reported (default: 3)",
    )
    parser.add_argument(
        "--no_adc_timing_warmup",
        action="store_true",
        help="Skip one untimed LUT+ADC pass before timed rounds (not recommended)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="M,nbits,train_size",
        help="Optional subset of combos to evaluate (e.g. --only 2,8,10000 16,4,100000)",
    )
    args = parser.parse_args()

    grid_root = args.grid_root.resolve()
    if not grid_root.is_dir():
        raise SystemExit(f"--grid_root does not exist: {grid_root}")
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"--data_root does not exist: {data_root}")
    out_dir = (args.out_dir or grid_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    only_combos: set[tuple[int, int, int]] | None = None
    if args.only:
        only_combos = set()
        for spec in args.only:
            parts = [int(x) for x in spec.split(",")]
            if len(parts) != 3:
                raise SystemExit(f"--only entry must be M,nbits,train_size; got {spec!r}")
            only_combos.add(tuple(parts))  # type: ignore[arg-type]

    print(f"grid_root = {grid_root}")
    print(f"data_root = {data_root}")
    print(f"out_dir   = {out_dir}")
    print(f"datasets  = {args.datasets}")
    print(f"sample_db={args.sample_db}, sample_queries={args.sample_queries}, seed={args.seed}")
    print(
        f"adc_timing: repeats={args.adc_timing_repeats}, "
        f"warmup={'off' if args.no_adc_timing_warmup else 'on'} (CPU process time, median per phase)"
    )
    if only_combos:
        print(f"only      = {sorted(only_combos)}")
    if args.overwrite_csv:
        print("overwrite_csv: on (per-dataset eval+recon CSVs are deleted before each dataset)")

    numba_warmup()

    t_start = time.time()
    for dataset in args.datasets:
        evaluate_dataset(
            dataset=dataset,
            grid_root=grid_root,
            data_root=data_root,
            out_dir=out_dir,
            sample_db=args.sample_db,
            sample_queries=args.sample_queries,
            seed=args.seed,
            skip_existing=args.skip_existing,
            only_combos=only_combos,
            adc_timing_repeats=args.adc_timing_repeats,
            adc_timing_warmup=not args.no_adc_timing_warmup,
            overwrite_csv=args.overwrite_csv,
        )
    print(f"Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
