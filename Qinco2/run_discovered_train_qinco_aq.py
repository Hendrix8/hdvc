#!/usr/bin/env python3
"""
Run train_qinco_aq (AQ codebook fitting) for every encode_train_setting_*.npz that does not yet
have a *completed* train_qinco_aq_setting_*.npz under the same encode_train branch.

Prerequisite: encode_train_setting_X.npz (merged file, not .part_*) must exist. If an experiment
has no encode_train output, run encode_train first (not handled here).

By default uses ds.trainset / ds.valset = 10k / 10k (same scale as the parallel pipeline). Full
learn-set least-squares on ~1e8 vectors often stalls or runs far too long — use --full_train
only if you intend that cost.

After this finishes, run run_discovered_adc_jobs.py then aggregate_qinco_adc_results.py.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from aggregate_qinco_adc_results import EXPERIMENT_DIR_RE  # type: ignore
from util.file_util import FileReader

_DATASET_TRAIN = {
    "bigann": ("bigann/SIFT1M", "bigann_learn.bvecs"),
    "deep": ("deep", "learn_100m.fvecs"),
    "gist": ("gist", "gist_learn.fvecs"),
    "msmarco": ("msmarco", "train.fvecs"),
    "openai": ("openai", "openai_train1m.fvecs"),
}


def experiment_dataset_from_path(p: Path) -> Optional[str]:
    cur = p
    while cur.parent != cur:
        m = EXPERIMENT_DIR_RE.match(cur.name)
        if m:
            return m.group("ds")
        cur = cur.parent
    return None


def any_train_qinco_complete(enc_train_setting_dir: Path) -> bool:
    tqa_root = enc_train_setting_dir / "train_qinco_aq"
    if not tqa_root.is_dir():
        return False
    for npz in tqa_root.rglob("train_qinco_aq_setting_*.npz"):
        if not npz.is_file():
            continue
        log = npz.parent / "run.log"
        if log.is_file() and FileReader.str_in_file("Task done", str(log)):
            return True
    return False


def iter_encode_train_npz(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("encode_train_setting_*.npz")):
        if ".part_" in p.name:
            continue
        yield p


def build_job(
    enc_npz: Path, data_root: str
) -> Optional[Tuple[str, Path, Path, str]]:
    """Returns (dataset, enc_npz, out_npz, trainset_path) or None."""
    ds = experiment_dataset_from_path(enc_npz)
    if ds is None or ds not in _DATASET_TRAIN:
        return None
    enc_dir = enc_npz.parent
    subdir, fname = _DATASET_TRAIN[ds]
    trainset = os.path.join(data_root, subdir, fname)
    if not os.path.isfile(trainset):
        return None

    out_dir = enc_dir / "train_qinco_aq" / "setting_0"
    out_npz = out_dir / "train_qinco_aq_setting_0.npz"
    return ds, enc_npz, out_npz, trainset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--result_root",
        default="/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/compressed_qinco_res/unzip",
    )
    ap.add_argument(
        "--python",
        default=os.environ.get(
            "PYTHON", os.path.expanduser("~/.miniconda3/envs/tsfm/bin/python")
        ),
    )
    ap.add_argument(
        "--full_train",
        action="store_true",
        help="Use full learn mmap (no ds.trainset cap). Very slow / heavy vs default 10k/10k.",
    )
    ap.add_argument(
        "--train_subset",
        type=int,
        default=10000,
        help="ds.trainset when not --full_train (default 10000)",
    )
    ap.add_argument(
        "--val_subset",
        type=int,
        default=10000,
        help="ds.valset when not --full_train (default 10000)",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Max jobs to run (0 = no limit)")
    args = ap.parse_args()

    data_root = os.environ.get("QINCO2_DATA_ROOT", "").rstrip("/")
    if not data_root:
        print("ERROR: set QINCO2_DATA_ROOT", file=sys.stderr)
        sys.exit(2)

    root = Path(args.result_root)
    code = Path(_repo_root)

    jobs: List[Tuple[str, Path, Path, str]] = []
    skipped_complete = 0
    skipped_no_train_file = 0
    for enc_npz in iter_encode_train_npz(root):
        enc_dir = enc_npz.parent
        if any_train_qinco_complete(enc_dir):
            skipped_complete += 1
            continue
        b = build_job(enc_npz, data_root)
        if b is None:
            skipped_no_train_file += 1
            continue
        jobs.append(b)

    ran = 0
    for i, (ds, enc_npz, out_npz, trainset) in enumerate(jobs):
        if args.limit and ran >= args.limit:
            break
        out_dir = out_npz.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(code / "run.py"),
            "task=train_qinco_aq",
            f"output={out_npz}",
            f"trainset={trainset}",
            f"encoded_trainset={enc_npz}",
        ]
        if not args.full_train:
            cmd.append(f"ds.trainset={args.train_subset}")
            cmd.append(f"ds.valset={args.val_subset}")

        if args.dry_run:
            print(f"DRY [{ds}] cd {out_dir} && {' '.join(cmd)}")
            ran += 1
            continue

        print(f"[{i + 1}/{len(jobs)}] train_qinco_aq {ds} … {enc_npz.relative_to(root)}")
        r = subprocess.run(cmd, cwd=str(out_dir))
        if r.returncode != 0:
            print(f"FAILED ({r.returncode}): {out_dir}", file=sys.stderr)
        else:
            ran += 1

    print(
        f"Summary: pending_jobs={len(jobs)} ran_or_printed={ran} "
        f"skipped_already_complete={skipped_complete} "
        f"skipped_missing_trainset_file={skipped_no_train_file}"
    )


if __name__ == "__main__":
    main()
