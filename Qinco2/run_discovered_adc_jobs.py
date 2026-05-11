#!/usr/bin/env python3
"""
Run qinco_aq_faiss_compute_distances for every train_qinco_aq_setting_*.npz
found under a result root, using the matching encode_test *_0-9999.npz and
dataset vectors from QINCO2_DATA_ROOT.

Skips runs whose output npz exists and run.log contains 'Task done'.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from util.file_util import FileReader
from aggregate_qinco_adc_results import DATASETS, EXPERIMENT_DIR_RE  # reuse dataset list + regex

# Paths relative to QINCO2_DATA_ROOT (same as util.var_util default_data_info)
_DATASET_VEC = {
    "bigann": ("bigann/SIFT1M/bigann_base.bvecs", "bigann/SIFT1M/bigann_query.bvecs"),
    "deep": ("deep/test_1m.fvecs", "deep/query_10k.fvecs"),
    "gist": ("gist/gist_base.fvecs", "gist/gist_query.fvecs"),
    "msmarco": ("msmarco/base.fvecs", "msmarco/query.fvecs"),
    "openai": ("openai/openai_base1m.fvecs", "openai/openai_query10k.fvecs"),
}


def parse_train_qinco_setting(dir_name: str) -> Optional[int]:
    m = re.match(r"setting_(\d+)$", dir_name)
    return int(m.group(1)) if m else None


def train_save_root_from_tqa_dir(train_qinco_dir: Path) -> Path:
    """.../setting_TR/encode_train/setting_ET/train_qinco_aq/setting_S -> .../setting_TR"""
    # train_qinco_aq -> encode_train/setting_ET -> encode_train -> setting_TR
    return train_qinco_dir.parent.parent.parent.parent


def find_encode_test_npz(train_save_root: Path) -> Optional[Path]:
    enc_root = train_save_root / "encode_test"
    if not enc_root.is_dir():
        return None
    cands = sorted(enc_root.rglob("encode_test_setting_*_0-9999.npz"))
    return cands[0] if cands else None


def experiment_and_dataset(exp_m_root: Path) -> Optional[Tuple[str, Path]]:
    """Return (dataset_name, experiment_folder bigann_M_*_K_*) from path walk."""
    cur = exp_m_root
    while cur.parent != cur:
        if EXPERIMENT_DIR_RE.match(cur.name):
            return EXPERIMENT_DIR_RE.match(cur.name).group("ds"), cur
        cur = cur.parent
    return None


def adc_out_dir(train_qinco_dir: Path) -> Path:
    return train_qinco_dir / "qinco_aq_faiss_compute_distances" / "setting_0"


def adc_complete(out_dir: Path) -> bool:
    npz = out_dir / "qinco_aq_faiss_compute_distances_setting_0.npz"
    log = out_dir / "run.log"
    if not npz.is_file() or not log.is_file():
        return False
    if not FileReader.str_in_file("Task done", str(log)):
        return False
    try:
        with np.load(npz, allow_pickle=False) as z:
            return int(np.asarray(z["faiss_adc_version"]).reshape(-1)[0]) >= 3
    except Exception:
        return False


def iter_jobs(result_root: Path) -> List[Tuple[Path, Path, Path, str, Path]]:
    """
    Each tuple: (train_qinco_dir, tqa_npz, encode_npz, dataset, out_dir).
    """
    jobs: List[Tuple[Path, Path, Path, str, Path]] = []
    for tqa_npz in sorted(result_root.rglob("train_qinco_aq_setting_*.npz")):
        if not tqa_npz.is_file():
            continue
        train_qinco_dir = tqa_npz.parent
        s = parse_train_qinco_setting(train_qinco_dir.name)
        if s is None:
            continue
        if tqa_npz.name != f"train_qinco_aq_setting_{s}.npz":
            continue
        train_save_root = train_save_root_from_tqa_dir(train_qinco_dir)
        enc = find_encode_test_npz(train_save_root)
        if enc is None:
            continue
        parsed = experiment_and_dataset(train_qinco_dir)
        if parsed is None:
            continue
        ds, _ = parsed
        if ds not in DATASETS:
            continue
        out_dir = adc_out_dir(train_qinco_dir)
        jobs.append((train_qinco_dir, tqa_npz, enc, ds, out_dir))
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result_root",
        default="/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/compressed_qinco_res/unzip",
        help="Root containing dataset_M_*_K_* experiment trees",
    )
    p.add_argument(
        "--python",
        default=os.environ.get("PYTHON", f"{os.path.expanduser('~')}/.miniconda3/envs/tsfm/bin/python"),
        help="Python with PyTorch for Qinco2/run.py",
    )
    p.add_argument("--n_db", type=int, default=10000)
    p.add_argument("--n_queries", type=int, default=1000)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print jobs only",
    )
    args = p.parse_args()

    data_root = os.environ.get("QINCO2_DATA_ROOT", "").rstrip("/")
    if not data_root:
        print("ERROR: set QINCO2_DATA_ROOT to vector dataset root", file=sys.stderr)
        sys.exit(2)

    result_root = Path(args.result_root)
    code_path = Path(_repo_root)
    jobs = iter_jobs(result_root)
    ran, skipped = 0, 0
    for _train_qinco_dir, tqa_npz, enc_npz, ds, out_dir in jobs:
        if adc_complete(out_dir):
            skipped += 1
            continue
        db_rel, q_rel = _DATASET_VEC[ds]
        db = os.path.join(data_root, db_rel)
        queries = os.path.join(data_root, q_rel)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / "qinco_aq_faiss_compute_distances_setting_0.npz"
        cmd = [
            args.python,
            str(code_path / "run.py"),
            "task=qinco_aq_faiss_compute_distances",
            "cpu=True",
            f"output={out_npz}",
            f"qinco_aq_codebooks={tqa_npz}",
            f"encoded_db={enc_npz}",
            f"db={db}",
            f"queries={queries}",
            f"ds.db={args.n_db}",
            f"ds.query={args.n_queries}",
        ]
        if args.dry_run:
            print("DRY", "cd", out_dir, "&&", " ".join(cmd))
            continue
        print(f"Running ADC: {ds} …/{tqa_npz.parent.name}/{tqa_npz.name}")
        r = subprocess.run(cmd, cwd=str(out_dir))
        if r.returncode != 0:
            print(f"FAILED ({r.returncode}): {out_dir}", file=sys.stderr)
        else:
            ran += 1
    print(f"Done. ran={ran} skipped_already_complete={skipped} pending_or_failed={len(jobs) - skipped - ran}")


if __name__ == "__main__":
    main()
