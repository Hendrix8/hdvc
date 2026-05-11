#!/usr/bin/env python3
"""
Scan one or more QINCo2 result roots for completed
qinco_aq_faiss_compute_distances runs and write per-dataset CSVs matching
save_results_to_csv.py columns (without extra recon columns).

See Qinco2/slurm/run_pipeline_master.sh for typical result paths.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from util.file_util import FileReader

DATASETS = ("bigann", "deep", "gist", "msmarco", "openai")

EXPERIMENT_DIR_RE = re.compile(
    r"^(?P<ds>bigann|deep|gist|msmarco|openai)_M_(?P<M>\d+)_K_(?P<K>\d+)$"
)

N_TRAIN_BY_DATASET = {
    "bigann": 1_000_000,
    "deep": 1_000_000,
    "gist": 500_000,
    "msmarco": 1_000_000,
    "openai": 1_000_000,
}
NB_BY_DATASET = {
    "bigann": 1_000_000,
    "deep": 1_000_000,
    "gist": 1_000_000,
    "msmarco": 1_000_000,
    "openai": 1_000_000,
}
DIM_BY_DATASET = {
    "bigann": 128,
    "deep": 96,
    "gist": 960,
    "msmarco": 1024,
    "openai": 1536,
}

CSV_COLUMNS = [
    "method",
    "dataset",
    "experiment_folder",
    "nq",
    "nb",
    "nb_sample",
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
]


def _scalar_from_npz(z: np.lib.npyio.NpzFile, key: str) -> float:
    a = z[key]
    return float(np.asarray(a).reshape(-1)[0])


def parse_experiment_dir_from_path(path: Path) -> Optional[Tuple[str, int, int]]:
    """Walk parents; return (dataset, M, K) from first matching experiment folder name."""
    for p in [path] + list(path.parents):
        if not p.name:
            continue
        m = EXPERIMENT_DIR_RE.match(p.name)
        if m:
            return m.group("ds"), int(m.group("M")), int(m.group("K"))
    return None


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def find_main_train_conf(experiment_root: Path, npz_path: Path) -> Optional[Dict[str, Any]]:
    """Use train_save_path setting from npz path (e.g. .../bigann_M_2_K_16/setting_0/...)."""
    try:
        rel = npz_path.relative_to(experiment_root)
    except ValueError:
        rel = None
    if rel is not None and rel.parts and rel.parts[0].startswith("setting_"):
        try:
            i = int(rel.parts[0].split("_", 1)[1])
        except (IndexError, ValueError):
            i = None
        if i is not None:
            conf_path = experiment_root / f"setting_{i}" / f"conf_{i}.json"
            data = load_json(conf_path)
            if data is not None:
                return data

    for d in sorted(experiment_root.glob("setting_*")):
        if not d.is_dir():
            continue
        try:
            i = int(d.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        data = load_json(d / f"conf_{i}.json")
        if data is not None:
            return data
    return None


def compute_train_size(dataset: str, train_conf: Optional[Dict[str, Any]]) -> int:
    n_train: Optional[float] = None
    if train_conf is not None:
        n_train = train_conf.get("n_train")
        if n_train is not None:
            n_train = float(n_train)
    if n_train is None:
        n_train = 1.0
    if n_train > 1:
        return int(n_train)
    return int(np.rint(n_train * N_TRAIN_BY_DATASET[dataset]))


def adc_conf_path_for_npz(npz_path: Path) -> Path:
    d = npz_path.parent
    for cand in sorted(d.glob("conf_*.json")):
        return cand
    return d / "conf_0.json"


def iter_adc_npz_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    yield from root.rglob("qinco_aq_faiss_compute_distances_setting_*.npz")


def is_adc_complete(npz_path: Path) -> bool:
    run_log = npz_path.parent / "run.log"
    if not run_log.is_file():
        return False
    return FileReader.str_in_file("Task done", str(run_log))


def build_row_from_npz(npz_path: Path) -> Optional[Dict[str, Any]]:
    if not is_adc_complete(npz_path):
        return None

    parsed = parse_experiment_dir_from_path(npz_path)
    if parsed is None:
        return None
    dataset, M, K = parsed
    if K <= 0 or (K & (K - 1)) != 0:
        return None
    nbits = int(np.log2(K))
    experiment_root = npz_path
    while experiment_root.parent != experiment_root:
        if EXPERIMENT_DIR_RE.match(experiment_root.name):
            break
        experiment_root = experiment_root.parent
    else:
        return None

    train_conf = find_main_train_conf(experiment_root, npz_path)
    train_size = compute_train_size(dataset, train_conf)

    experiment_folder = str(npz_path.parent.resolve())

    with np.load(npz_path, allow_pickle=False) as z:
        if "faiss_adc_version" not in z:
            return None
        if int(np.asarray(z["faiss_adc_version"]).reshape(-1)[0]) < 3:
            return None

        if "n_queries" in z and "n_db" in z:
            nq = int(_scalar_from_npz(z, "n_queries"))
            nb_sample = int(_scalar_from_npz(z, "n_db"))
        elif "all_approx_dists" in z:
            nq, nb_sample = map(int, z["all_approx_dists"].shape[:2])
        else:
            adc_conf = load_json(adc_conf_path_for_npz(npz_path)) or {}
            nq = int(adc_conf.get("n_queries", 1000))
            nb_sample = int(adc_conf.get("n_db", 10000))

        if "avg_distance_table_time_s" in z:
            dist_tab = _scalar_from_npz(z, "avg_distance_table_time_s")
        else:
            dist_tab = _scalar_from_npz(z, "total_distance_table_time_s") / nq

        if "avg_adc_time_s" in z:
            adc_t = _scalar_from_npz(z, "avg_adc_time_s")
        else:
            adc_t = _scalar_from_npz(z, "total_adc_time_s") / nq
        dist_tab_tot = dist_tab * nq
        adc_t_tot = adc_t * nq
        cdist_t = _scalar_from_npz(z, "total_cdist_time_s") if "total_cdist_time_s" in z else -1
        rel_m = _scalar_from_npz(z, "avg_relative_error")
        if "std_relative_error" in z:
            rel_s = _scalar_from_npz(z, "std_relative_error")
        elif "all_relative_errors" in z:
            rel_s = float(np.std(z["all_relative_errors"]))
        else:
            rel_s = 0.0
        if not np.isfinite(rel_m):
            rel_m = -1.0
        if not np.isfinite(rel_s):
            rel_s = -1.0

    nb = NB_BY_DATASET[dataset]
    per_pair_adc_time_ns = (
        (dist_tab_tot + adc_t_tot) / (nq * nb_sample)
    ) * 1e9

    return {
        "method": "QINCo2",
        "dataset": dataset,
        "experiment_folder": experiment_folder,
        "nq": nq,
        "nb": nb,
        "nb_sample": nb_sample,
        "dim": DIM_BY_DATASET[dataset],
        "n_subquantizers": M,
        "nbits": nbits,
        "bits_per_vector": M * nbits,
        "train_size": train_size,
        "train_time_s": -1,
        "encoding_time_s": -1,
        "distance_table_time_s": round(dist_tab_tot, 6),
        "cdist_time_s": round(cdist_t, 6),
        "adc_time_s": round(adc_t_tot, 6),
        "per_pair_adc_time_ns": round(per_pair_adc_time_ns, 6),
        "rel_error_mean": round(rel_m, 6),
        "rel_error_std": round(rel_s, 6),
        "_dedupe_key": str(npz_path.resolve()),
    }


def collect_rows(roots: List[Path]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        for npz_path in iter_adc_npz_files(root):
            row = build_row_from_npz(Path(npz_path))
            if row is None:
                continue
            key = row.pop("_dedupe_key")
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def iter_experiment_roots(result_root: Path) -> Iterable[Path]:
    if not result_root.is_dir():
        return
    for child in sorted(result_root.iterdir()):
        if child.is_dir() and EXPERIMENT_DIR_RE.match(child.name):
            yield child


def experiment_has_train_qinco_aq_npz(exp: Path) -> bool:
    return any(exp.rglob("train_qinco_aq_setting_*.npz"))


def experiment_has_adc_complete_anywhere(exp: Path) -> bool:
    for npz in exp.rglob("qinco_aq_faiss_compute_distances_setting_*.npz"):
        if is_adc_complete(npz):
            return True
    return False


def build_coverage_rows(roots: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            continue
        for exp in iter_experiment_roots(root):
            m = EXPERIMENT_DIR_RE.match(exp.name)
            if not m:
                continue
            ds, M_s, K_s = m.group("ds"), m.group("M"), m.group("K")
            has_tqa = experiment_has_train_qinco_aq_npz(exp)
            has_adc = experiment_has_adc_complete_anywhere(exp)
            if has_tqa and not has_adc:
                out.append(
                    {
                        "source_root": str(root),
                        "experiment_path": str(exp.resolve()),
                        "dataset": ds,
                        "M": int(M_s),
                        "K": int(K_s),
                        "has_train_qinco_aq_npz": True,
                        "has_adc_complete": False,
                    }
                )
    return out


def main() -> None:
    default_roots = [
        "/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/compressed_qinco_res/unzip",
        "/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/qinco_new_results_compressed/"
        "lustre/fsn1/projects/rech/thj/commun/qinco2_results",
    ]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--roots",
        nargs="+",
        default=default_roots,
        help="Result roots containing dataset_M_*_K_* experiment trees",
    )
    p.add_argument(
        "--out_dir",
        default=".",
        help="Directory for per-dataset CSVs and optional coverage file",
    )
    p.add_argument(
        "--no-coverage",
        action="store_true",
        help="Do not write qinco2_adc_coverage.csv",
    )
    args = p.parse_args()
    roots = [Path(r) for r in args.roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(roots)
    by_ds: Dict[str, List[Dict[str, Any]]] = {ds: [] for ds in DATASETS}
    for row in rows:
        by_ds.setdefault(row["dataset"], []).append(row)

    for ds in DATASETS:
        df = pd.DataFrame(by_ds.get(ds, []), columns=CSV_COLUMNS)
        out_fp = out_dir / f"{ds}_QINCo2_adc_vs_exact_eval.csv"
        df.to_csv(out_fp, index=False)
        print(f"Wrote {out_fp} ({len(df)} rows)")

    if not args.no_coverage:
        cov = build_coverage_rows(roots)
        cov_path = out_dir / "qinco2_adc_coverage.csv"
        pd.DataFrame(cov).to_csv(cov_path, index=False)
        print(f"Wrote {cov_path} ({len(cov)} rows)")


if __name__ == "__main__":
    main()
