#!/usr/bin/env python3
"""Build unified_manifest.csv from *_{METHOD}_adc_vs_exact_eval.csv files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from distance_eval.artifacts import resolve_pq_index_dir
from hdvc_paths import get_results_root

DEFAULT_QINCO_CONFIG = _REPO / "lib" / "Qinco" / "config" / "qinco_cfg.yaml"


def _find_qinco_codes_npz(results_root: Path, dataset: str, M: int, K: int) -> str:
    pat = f"**/qinco2_codes/{dataset}_M{M}_K{K}_codes.npz"
    matches = sorted(
        results_root.glob(pat),
        key=lambda p: p.stat().st_mtime if p.is_file() else 0,
        reverse=True,
    )
    return str(matches[0]) if matches else ""


def build_manifest_rows(
    results_dir: Path,
    results_root: Path | None = None,
    path_map_from: str | None = None,
    path_map_to: str | None = None,
) -> list[dict]:
    results_root = results_root or get_results_root()
    rows: list[dict] = []
    for csv_path in sorted(results_dir.rglob("*_adc_vs_exact_eval.csv")):
        try:
            df = pd.read_csv(csv_path, on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=False)
        if df.empty or "method" not in df.columns:
            continue
        stem = csv_path.stem  # e.g. deep_PQ_adc_vs_exact_eval
        for _, r in df.iterrows():
            method = str(r.get("method", "")).strip()
            if not method:
                continue
            dataset = str(r.get("dataset", stem.split("_")[0]))
            try:
                n_subq = int(r.get("n_subquantizers", 0))
            except (TypeError, ValueError):
                n_subq = 0
            try:
                nbits = int(r.get("nbits", 0))
            except (TypeError, ValueError):
                nbits = 0
            try:
                bpv = int(r.get("bits_per_vector", n_subq * nbits))
            except (TypeError, ValueError):
                bpv = n_subq * nbits
            try:
                train_size = int(r.get("train_size", 0))
            except (TypeError, ValueError):
                train_size = 0
            rel_err = r.get("rel_error_mean", float("nan"))
            nq = int(r.get("nq", 0) or 0)
            nb_sample = int(r.get("nb_sample", r.get("nb", 0)) or 0)
            dim = int(r.get("dim", 0) or 0)

            exp_folder = r.get("experiment_folder", "")
            model_path = r.get("model_path", "")
            if pd.notna(exp_folder):
                exp_folder = str(exp_folder)
                if path_map_from and path_map_to and exp_folder.startswith(path_map_from):
                    exp_folder = path_map_to + exp_folder[len(path_map_from) :]
            if pd.notna(model_path):
                model_path = str(model_path)
                if path_map_from and path_map_to and model_path.startswith(path_map_from):
                    model_path = path_map_to + model_path[len(path_map_from) :]

            row: dict = {
                "method": method,
                "dataset": dataset,
                "n_subquantizers": n_subq,
                "nbits": nbits,
                "bits_per_vector": bpv,
                "train_size": train_size,
                "nq": nq,
                "nb_sample": nb_sample,
                "dim": dim,
                "rel_error_mean": rel_err,
                "experiment_folder": str(exp_folder) if pd.notna(exp_folder) else "",
                "artifact_index": "",
                "model_path": str(model_path) if pd.notna(model_path) else "",
                "codes_npz": "",
                "config_path": "",
            }

            try:
                if method == "PQ" and exp_folder:
                    folder = resolve_pq_index_dir(
                        Path(exp_folder), n_subq, nbits, train_size
                    )
                    row["artifact_index"] = str(folder / "pq_model.index")
                elif method == "OPQ" and exp_folder:
                    row["artifact_index"] = str(Path(exp_folder) / "opq_model.index")
                elif method == "LSQpp" and exp_folder:
                    row["artifact_index"] = str(Path(exp_folder) / "lsq_model.index")
                elif method == "VAQ" and exp_folder:
                    row["artifact_index"] = str(Path(exp_folder))
                elif method == "QINCo2" and model_path:
                    row["artifact_index"] = str(model_path)
                    K = 1 << nbits if nbits > 0 else 0
                    if K > 0 and n_subq > 0:
                        row["codes_npz"] = _find_qinco_codes_npz(
                            results_root, dataset, n_subq, K
                        )
                    row["config_path"] = (
                        str(DEFAULT_QINCO_CONFIG)
                        if DEFAULT_QINCO_CONFIG.is_file()
                        else ""
                    )
                elif method == "RaBitQ":
                    row["artifact_index"] = str(exp_folder or "")
            except FileNotFoundError:
                row["artifact_index"] = ""

            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build unified_manifest.csv")
    p.add_argument(
        "--results_dir",
        type=Path,
        default=get_results_root() / "relerr_cpp",
        help="Directory containing *_adc_vs_exact_eval.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "distance_eval" / "results" / "unified_manifest.csv",
        help="Output manifest path",
    )
    p.add_argument(
        "--results_root",
        type=Path,
        default=None,
        help="Root for globbing QINCo2 codes (default: hdvc get_results_root)",
    )
    p.add_argument(
        "--path_map_from",
        type=str,
        default=None,
        help="Optional path prefix to remap in experiment/model paths",
    )
    p.add_argument(
        "--path_map_to",
        type=str,
        default=None,
        help="Replacement path prefix for --path_map_from",
    )
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_manifest_rows(
        args.results_dir,
        args.results_root,
        path_map_from=args.path_map_from,
        path_map_to=args.path_map_to,
    )
    if not rows:
        print(f"No rows from {args.results_dir}")
        return
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows → {args.output}")


if __name__ == "__main__":
    main()
