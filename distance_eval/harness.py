#!/usr/bin/env python3
"""Unified IP+norm ADC timing across methods (CPU process time)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from distance_eval.base import compose_l2
from distance_eval.data import load_db_and_queries
from distance_eval.registry import get_backend
from distance_eval.timing import cpu_timer
from hdvc_paths import get_results_root
from scripts.evals.config import DATASET_CONFIG


def _set_num_threads(n: int) -> None:
    try:
        import faiss

        faiss.omp_set_num_threads(n)
    except Exception:
        pass
    try:
        import numba

        numba.set_num_threads(n)
    except Exception:
        pass


def _make_backend(row: pd.Series, tmp_dir: Path, db: np.ndarray) -> tuple[object, Path | None]:
    """Return (backend, optional temp path to delete)."""
    method = str(row["method"]).strip()
    tmp_created: Path | None = None

    if method == "PQ":
        return get_backend("PQ")(row["artifact_index"]), None
    if method == "OPQ":
        return get_backend("OPQ")(row["artifact_index"]), None
    if method == "LSQpp":
        return get_backend("LSQpp")(row["artifact_index"]), None
    if method == "VAQ":
        dim = int(row["dim"])
        return get_backend("VAQ")(row["artifact_index"], dim=dim), None
    if method == "RaBitQ":
        return get_backend("RaBitQ")(row.get("artifact_index") or None), None
    if method == "QINCo2":
        from src.utils import write_fvecs  # noqa: WPS433 — lazy (pulls faiss via utils)

        tmp_dir.mkdir(parents=True, exist_ok=True)
        db_fvecs = tmp_dir / f"qinco_db_{row.name}_{row['dataset']}.fvecs"
        write_fvecs(str(db_fvecs), db.astype(np.float32))
        tmp_created = db_fvecs
        cfg = row.get("config_path") or ""
        if not cfg or not Path(str(cfg)).is_file():
            raise FileNotFoundError(
                "QINCo2 requires config_path in manifest (default qinco_cfg.yaml missing?)"
            )
        codes = row.get("codes_npz") or ""
        if not codes or not Path(str(codes)).is_file():
            raise FileNotFoundError(
                "QINCo2 codes_npz missing; run manifest from repo root or set codes_npz column"
            )
        b = get_backend("QINCo2")(
            model_path=row["artifact_index"],
            config_path=str(cfg),
            db_fvecs_path=str(db_fvecs),
            codes_npz=str(codes),
            use_gpu=False,
        )
        return b, tmp_created

    raise ValueError(f"Unsupported method {method}")


def run_one_row(
    row: pd.Series,
    *,
    n_runs: int,
    override_nq: int | None,
    override_nb: int | None,
    warmup_runs: int,
    path_map_from: str | None,
    path_map_to: str | None,
    data_path_map_from: str | None,
    data_path_map_to: str | None,
    tmp_dir: Path,
) -> dict | None:
    method = str(row["method"]).strip()
    if method == "RaBitQ":
        print("  skip RaBitQ: native backend not wired (see distance_eval/README.md)")
        return None
    dataset = str(row["dataset"])
    if dataset not in DATASET_CONFIG:
        print(f"  skip: unknown dataset {dataset}")
        return None

    art = str(row.get("artifact_index", "") or "").strip()
    if path_map_from and path_map_to and art.startswith(path_map_from):
        art = path_map_to + art[len(path_map_from) :]
    if not art or art == "nan":
        print(f"  skip {method}: missing artifact_index")
        return None
    if method != "QINCo2" and not Path(art).exists():
        print(f"  skip {method}: artifact not found {art}")
        return None

    ds_path, qr_path = DATASET_CONFIG[dataset]
    if data_path_map_from and data_path_map_to:
        if ds_path.startswith(data_path_map_from):
            ds_path = data_path_map_to + ds_path[len(data_path_map_from) :]
        if qr_path.startswith(data_path_map_from):
            qr_path = data_path_map_to + qr_path[len(data_path_map_from) :]
    dim = int(row["dim"])
    nq = (
        int(override_nq)
        if override_nq is not None
        else (int(row["nq"]) if row["nq"] else 1000)
    )
    nb = (
        int(override_nb)
        if override_nb is not None
        else (int(row["nb_sample"]) if row["nb_sample"] else 10000)
    )

    db, qr = load_db_and_queries(ds_path, qr_path, dim, max_db=nb, max_queries=nq)
    db = db.astype(np.float32)
    qr = qr.astype(np.float32)

    backend, tmp = _make_backend(row, tmp_dir, db)
    try:
        import time as time_mod

        t0 = time_mod.process_time()
        codes = backend.encode(db)
        encode_time = time_mod.process_time() - t0

        t0 = time_mod.process_time()
        db_norms_sq = backend.db_norms_sq(codes)
        db_prep_time = time_mod.process_time() - t0

        q_norms_sq = np.sum(qr * qr, axis=1).astype(np.float32)

        lut_times = []
        ip_times = []
        comp_times = []
        totals = []

        nq_eff = qr.shape[0]
        nb_eff = db.shape[0]
        n_pairs = nq_eff * nb_eff

        for _ in range(max(0, warmup_runs)):
            qstate = backend.prepare_query(qr)
            ip = backend.ip_estimate(qstate, codes)
            _ = compose_l2(q_norms_sq, db_norms_sq, ip)

        for _ in range(n_runs):
            with cpu_timer() as t_lut:
                qstate = backend.prepare_query(qr)
            lut_times.append(t_lut.seconds)

            with cpu_timer() as t_ip:
                ip = backend.ip_estimate(qstate, codes)
            ip_times.append(t_ip.seconds)

            with cpu_timer() as t_comp:
                _ = compose_l2(q_norms_sq, db_norms_sq, ip)
            comp_times.append(t_comp.seconds)

            totals.append(t_lut.seconds + t_ip.seconds + t_comp.seconds)

        if hasattr(backend, "close"):
            backend.close()

        def mean_std(xs: list[float]) -> tuple[float, float]:
            a = np.array(xs, dtype=np.float64)
            return float(a.mean()), float(a.std()) if len(a) > 1 else 0.0

        lut_m, lut_s = mean_std(lut_times)
        ip_m, ip_s = mean_std(ip_times)
        c_m, c_s = mean_std(comp_times)
        tot_m, tot_s = mean_std(totals)

        return {
            "method": method,
            "dataset": dataset,
            "n_subquantizers": int(row["n_subquantizers"]),
            "nbits": int(row["nbits"]),
            "bits_per_vector": int(row["bits_per_vector"]),
            "train_size": int(row["train_size"]),
            "nq": nq_eff,
            "nb_sample": nb_eff,
            "encode_time_s": float(encode_time),
            "db_prep_time_s": float(db_prep_time),
            "lut_time_s_mean": lut_m,
            "lut_time_s_std": lut_s,
            "ip_scan_time_s_mean": ip_m,
            "ip_scan_time_s_std": ip_s,
            "compose_time_s_mean": c_m,
            "compose_time_s_std": c_s,
            "adc_total_time_s_mean": tot_m,
            "adc_total_time_s_std": tot_s,
            "adc_total_time_pp_mean": tot_m / max(1, n_pairs),
            "n_runs": n_runs,
            "rel_error_mean": row.get("rel_error_mean", np.nan),
            "artifact_index": art,
        }
    finally:
        if tmp is not None and tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass


def main() -> None:
    p = argparse.ArgumentParser(description="Unified ADC timing harness")
    p.add_argument(
        "--manifest",
        type=Path,
        default=_REPO / "distance_eval" / "results" / "unified_manifest.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "distance_eval" / "results" / "unified_adc_timing.csv",
    )
    p.add_argument("--method", type=str, default=None, help="Filter to this method")
    p.add_argument("--dataset", type=str, default=None, help="Filter to this dataset")
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--nq", type=int, default=None, help="Override number of queries")
    p.add_argument(
        "--nb", type=int, default=None, help="Override number of database vectors"
    )
    p.add_argument(
        "--warmup_runs",
        type=int,
        default=2,
        help="Untimed warmup iterations before measured runs",
    )
    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--path_map_from", type=str, default=None)
    p.add_argument("--path_map_to", type=str, default=None)
    p.add_argument(
        "--data_path_map_from",
        type=str,
        default="/data/cpanourg/2-hdvc/data",
    )
    p.add_argument(
        "--data_path_map_to",
        type=str,
        default=None,
        help="Dataset root replacement (eg /mnthdd/cpanourg/2-hdvc/data)",
    )
    args = p.parse_args()

    _set_num_threads(max(1, int(args.num_threads)))

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest} — run python -m distance_eval.manifest")
        sys.exit(1)

    df = pd.read_csv(args.manifest)
    if args.method:
        df = df[df["method"] == args.method]
    if args.dataset:
        df = df[df["dataset"] == args.dataset]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output.parent / "_tmp_qinco"

    all_rows: list[dict] = []
    for idx, row in df.iterrows():
        print(f"Row {idx}: {row.get('method')} {row.get('dataset')} …")
        try:
            out = run_one_row(
                row,
                n_runs=args.n_runs,
                override_nq=args.nq,
                override_nb=args.nb,
                warmup_runs=args.warmup_runs,
                path_map_from=args.path_map_from,
                path_map_to=args.path_map_to,
                data_path_map_from=args.data_path_map_from,
                data_path_map_to=args.data_path_map_to,
                tmp_dir=tmp_dir,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        if out:
            all_rows.append(out)

    if not all_rows:
        print("No results.")
        return

    res = pd.DataFrame(all_rows)
    res.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(res)} rows)")

    by_bpv = (
        res.groupby(["dataset", "method", "bits_per_vector"])
        .agg(
            adc_total_time_pp_mean=("adc_total_time_pp_mean", "mean"),
            adc_total_time_pp_std=("adc_total_time_pp_mean", "std"),
            rel_error_mean=("rel_error_mean", "first"),
        )
        .reset_index()
    )
    by_bpv["adc_total_time_pp_std"] = by_bpv["adc_total_time_pp_std"].fillna(0)
    by_bpv.to_csv(args.output.parent / "unified_adc_timing_by_bpv.csv", index=False)

    by_m = (
        res.groupby(["dataset", "method"])
        .agg(
            adc_total_time_pp_mean=("adc_total_time_pp_mean", "mean"),
            adc_total_time_pp_std=("adc_total_time_pp_mean", "std"),
        )
        .reset_index()
    )
    by_m["adc_total_time_pp_std"] = by_m["adc_total_time_pp_std"].fillna(0)
    by_m.to_csv(args.output.parent / "unified_adc_timing_by_method.csv", index=False)


if __name__ == "__main__":
    main()
