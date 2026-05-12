#!/usr/bin/env python3
"""Run SAQ on local 10k/1k subsets with reusable IVF artifacts from RaBitQ runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

for _anc in Path(__file__).resolve().parents:
    if (_anc / "hdvc_paths.py").is_file():
        if str(_anc) not in sys.path:
            sys.path.insert(0, str(_anc))
        break
from hdvc_paths import get_results_root  # noqa: E402

from saq.saq_relerr import (  # noqa: E402
    _conda_cmake_prefix_path,
    _load_vectors,
    _read_first_csv_row,
    _read_fvecs,
    _run,
    _write_fvecs,
    _write_ivecs,
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    base_path: Path
    query_path: Path
    vec_type: str
    dim: int


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_SPECS = {
    "deep10k": DatasetSpec(
        dataset="deep10k",
        base_path=REPO_ROOT / "local" / "dataset_subsets" / "deep" / "deep_base_10k.fvecs",
        query_path=REPO_ROOT / "local" / "dataset_subsets" / "deep" / "deep_query_1k.fvecs",
        vec_type="fvecs",
        dim=96,
    ),
    "bigann10k": DatasetSpec(
        dataset="bigann10k",
        base_path=REPO_ROOT / "local" / "bigann_subsets" / "bigann_base_10k.bvecs",
        query_path=REPO_ROOT / "local" / "bigann_subsets" / "bigann_query_1k.bvecs",
        vec_type="bvecs",
        dim=128,
    ),
    "gist10k": DatasetSpec(
        dataset="gist10k",
        base_path=REPO_ROOT / "local" / "dataset_subsets" / "gist" / "gist_base_10k.fvecs",
        query_path=REPO_ROOT / "local" / "dataset_subsets" / "gist" / "gist_query_1k.fvecs",
        vec_type="fvecs",
        dim=960,
    ),
    "msmarco10k": DatasetSpec(
        dataset="msmarco10k",
        base_path=REPO_ROOT / "local" / "dataset_subsets" / "msmarco" / "msmarco_base_10k.fvecs",
        query_path=REPO_ROOT / "local" / "dataset_subsets" / "msmarco" / "msmarco_query_1k.fvecs",
        vec_type="fvecs",
        dim=1024,
    ),
    "openai10k": DatasetSpec(
        dataset="openai10k",
        base_path=REPO_ROOT / "local" / "dataset_subsets" / "openai" / "openai_base_10k.fvecs",
        query_path=REPO_ROOT / "local" / "dataset_subsets" / "openai" / "openai_query_1k.fvecs",
        vec_type="fvecs",
        dim=1536,
    ),
}


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _build_binaries(lib_root: Path, skip_build: bool) -> None:
    build_dir = lib_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    if skip_build:
        return
    cmake_configure = ["cmake", "-DBUILD_UNIT_TESTS=OFF"]
    conda_prefix_path = _conda_cmake_prefix_path()
    if conda_prefix_path:
        cmake_configure.append(f"-DCMAKE_PREFIX_PATH={conda_prefix_path}")
    cmake_configure.append("..")
    _run(cmake_configure, cwd=build_dir)
    _run(["cmake", "--build", ".", "-j"], cwd=build_dir)


def _default_shared_ivf_roots() -> list[Path]:
    results_root = get_results_root()
    return [
        results_root / "rabitq_native_ivf_run_20260511_212349" / "shared_ivf",
        results_root / "rabitq_native_ivf_addK128_20260511_222801" / "shared_ivf",
    ]


def _find_shared_artifacts(shared_ivf_roots: list[Path], dataset: str, clusters: int) -> tuple[Path, Path]:
    for root in shared_ivf_roots:
        ds_dir = root / dataset
        centroid = ds_dir / f"{dataset}_centroid_{clusters}.fvecs"
        cids = ds_dir / f"{dataset}_cluster_id_{clusters}.ivecs"
        if centroid.exists() and cids.exists():
            return centroid, cids
    roots = ", ".join(str(p) for p in shared_ivf_roots)
    raise FileNotFoundError(f"Missing shared IVF artifacts for {dataset} K={clusters} under: {roots}")


def _generate_shared_artifacts(
    spec: DatasetSpec,
    shared_dir: Path,
    clusters: int,
    force: bool,
) -> tuple[Path, Path, float]:
    shared_dir.mkdir(parents=True, exist_ok=True)
    centroid_fvec = shared_dir / f"{spec.dataset}_centroid_{clusters}.fvecs"
    cid_ivecs = shared_dir / f"{spec.dataset}_cluster_id_{clusters}.ivecs"

    start = time.perf_counter()
    if force or not centroid_fvec.exists() or not cid_ivecs.exists():
        base = _load_vectors(spec.base_path, spec.vec_type)
        if clusters == 1:
            centroid = np.mean(base, axis=0, keepdims=True, dtype=np.float32)
            cluster_id = np.zeros((base.shape[0], 1), dtype=np.int32)
        else:
            xb = np.ascontiguousarray(base.astype(np.float32))
            index = faiss.index_factory(spec.dim, f"IVF{clusters},Flat", faiss.METRIC_L2)
            index.verbose = False
            index.train(xb)
            centroid = index.quantizer.reconstruct_n(0, index.nlist).astype(np.float32)
            _, cluster_id = index.quantizer.search(xb, 1)
            cluster_id = cluster_id.astype(np.int32)
        _write_fvecs(centroid_fvec, centroid)
        _write_ivecs(cid_ivecs, cluster_id)
    prep_time_s = time.perf_counter() - start
    return centroid_fvec, cid_ivecs, prep_time_s


def _resolve_shared_artifacts(
    spec: DatasetSpec,
    shared_ivf_roots: list[Path],
    generated_root: Path,
    clusters: int,
    force_generate: bool,
) -> tuple[Path, Path, str, float]:
    try:
        centroid_src, cids_src = _find_shared_artifacts(shared_ivf_roots, spec.dataset, clusters)
        return centroid_src, cids_src, "reused", 0.0
    except FileNotFoundError:
        generated_dir = generated_root / spec.dataset
        centroid_src, cids_src, prep_time_s = _generate_shared_artifacts(
            spec=spec,
            shared_dir=generated_dir,
            clusters=clusters,
            force=force_generate,
        )
        return centroid_src, cids_src, "generated", prep_time_s


def _prepare_saq_dataset(
    saq_root: Path,
    spec: DatasetSpec,
    centroid_src: Path,
    cids_src: Path,
    clusters: int,
    force: bool,
    pca_train_size: int | None,
) -> tuple[Path, int, int, float]:
    ds_dir = saq_root / "data" / spec.dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    base = _load_vectors(spec.base_path, spec.vec_type)
    query = _load_vectors(spec.query_path, spec.vec_type)

    base_out = ds_dir / f"{spec.dataset}_base.fvecs"
    query_out = ds_dir / f"{spec.dataset}_query.fvecs"
    centroid_out = ds_dir / f"{spec.dataset}_centroid_{clusters}.fvecs"
    cids_out = ds_dir / f"{spec.dataset}_cluster_id_{clusters}.ivecs"
    base_pca_out = ds_dir / f"{spec.dataset}_base_pca.fvecs"
    query_pca_out = ds_dir / f"{spec.dataset}_query_pca.fvecs"
    centroid_pca_out = ds_dir / f"{spec.dataset}_centroid_{clusters}_pca.fvecs"
    vars_out = ds_dir / f"{spec.dataset}_base_pca.vars.fvecs"

    if force or not base_out.exists():
        _write_fvecs(base_out, base)
    if force or not query_out.exists():
        _write_fvecs(query_out, query)
    if force or not centroid_out.exists():
        shutil.copyfile(centroid_src, centroid_out)
    if force or not cids_out.exists():
        shutil.copyfile(cids_src, cids_out)

    start = time.perf_counter()
    if force or not all(p.exists() for p in [base_pca_out, query_pca_out, centroid_pca_out, vars_out]):
        centroids = _read_fvecs(centroid_src)
        pca_train = base if pca_train_size is None else base[: min(pca_train_size, len(base))]
        pca = faiss.PCAMatrix(spec.dim, spec.dim)
        pca.train(np.ascontiguousarray(pca_train.astype(np.float32)))
        base_pca = pca.apply(np.ascontiguousarray(base.astype(np.float32)))
        query_pca = pca.apply(np.ascontiguousarray(query.astype(np.float32)))
        centroid_pca = pca.apply(np.ascontiguousarray(centroids.astype(np.float32)))
        _write_fvecs(base_pca_out, base_pca)
        _write_fvecs(query_pca_out, query_pca)
        _write_fvecs(centroid_pca_out, centroid_pca)
        _write_fvecs(vars_out, np.var(base_pca, axis=0, keepdims=True).astype(np.float32))
    prep_time_s = time.perf_counter() - start
    return ds_dir, int(base.shape[0]), int(query.shape[0]), prep_time_s


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_key_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _row_merge_key(row: dict) -> tuple[str, ...]:
    return (
        _normalize_key_value(row["method"]),
        _normalize_key_value(row["dataset"]),
        _normalize_key_value(row["n_centroids"]),
        _normalize_key_value(row["nbits"]),
        _normalize_key_value(row["caq_adj_rd_lmt"]),
        _normalize_key_value(row["searcher_vars_bound_m"]),
        _normalize_key_value(row.get("use_fastscan", True)),
    )


def _merge_summary_rows(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    merged: dict[tuple[str, ...], dict] = {}
    for row in existing_rows:
        merged[_row_merge_key(row)] = row
    for row in new_rows:
        merged[_row_merge_key(row)] = row
    return list(merged.values())


def _saq_args_token(
    clusters: int,
    bits: float,
    rand_rotate: str,
    use_fastscan: str,
    caq_adj_rd_lmt: int,
    caq_adj_eps: float,
) -> str:
    token = f"ivf{clusters}_b{bits:g}"
    if rand_rotate == "false":
        token += "_norotate"
    if use_fastscan == "false":
        token += "_nofastscan"
    token += "_caq"
    if caq_adj_rd_lmt:
        token += "_adj"
        if caq_adj_rd_lmt != 6:
            token += f"_rdlmt{caq_adj_rd_lmt}"
        if caq_adj_eps != 1e-8:
            token += f"_eps{caq_adj_eps:.1e}"
    token += "_seg_pca"
    return token


def main() -> None:
    p = argparse.ArgumentParser(description="SAQ IVF grid on local 10k/1k subsets with shared IVF artifacts.")
    p.add_argument("--saq_root", type=Path, default=REPO_ROOT / "lib" / "SAQ")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["deep10k", "bigann10k", "gist10k", "msmarco10k", "openai10k"],
    )
    p.add_argument("--bits", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 3, 4, 6, 8])
    p.add_argument("--clusters", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--caq_adj_rd_lmt", type=int, nargs="+", default=[0, 2, 6, 12])
    p.add_argument("--searcher_vars_bound_m", type=float, nargs="+", default=[2, 4])
    p.add_argument("--nprobe", type=int, default=10)
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--warmup_runs", type=int, default=2)
    p.add_argument("--num_threads", type=int, default=0)
    p.add_argument("--rand_rotate", choices=["true", "false"], default="true")
    p.add_argument("--use_fastscan", choices=["true", "false"], default="true")
    p.add_argument("--caq_adj_eps", type=float, default=1e-8)
    p.add_argument("--pca_train_size", type=int, default=None)
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--force_index", action="store_true")
    p.add_argument("--skip_build", action="store_true")
    p.add_argument(
        "--output_root",
        type=Path,
        default=get_results_root() / "saq_native_ivf",
    )
    p.add_argument(
        "--shared_ivf_roots",
        type=Path,
        nargs="+",
        default=None,
        help="Directories that contain shared_ivf/<dataset>/<dataset>_centroid_K.fvecs artifacts.",
    )
    args = p.parse_args()

    saq_root = args.saq_root.resolve()
    bin_dir = saq_root / "bin"
    create_index = bin_dir / "create_index"
    test_relative_error = bin_dir / "test_relative_error"

    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_log = args.output_root / "progress.jsonl"
    shared_ivf_roots = [p.resolve() for p in (args.shared_ivf_roots or _default_shared_ivf_roots())]
    generated_shared_ivf_root = args.output_root / "shared_ivf_generated"

    _build_binaries(saq_root, args.skip_build)
    if not create_index.exists() or not test_relative_error.exists():
        raise FileNotFoundError("Expected SAQ binaries missing in lib/SAQ/bin")

    for dataset in args.datasets:
        if dataset not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset: {dataset}")
        spec = DATASET_SPECS[dataset]
        if not spec.base_path.exists() or not spec.query_path.exists():
            raise FileNotFoundError(f"Missing subset vectors for {dataset}")

        dataset_dir = args.output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []

        for clusters in sorted(set(args.clusters)):
            effective_nprobe = max(1, min(args.nprobe, clusters))
            centroid_src, cids_src, artifact_source, artifact_prep_time_s = _resolve_shared_artifacts(
                spec=spec,
                shared_ivf_roots=shared_ivf_roots,
                generated_root=generated_shared_ivf_root,
                clusters=clusters,
                force_generate=args.force_prepare,
            )
            ds_dir, nb, nq, prep_time_s = _prepare_saq_dataset(
                saq_root,
                spec,
                centroid_src,
                cids_src,
                clusters,
                args.force_prepare,
                args.pca_train_size,
            )

            _append_jsonl(
                progress_log,
                {
                    "event": "prepared_dataset",
                    "dataset": dataset,
                    "clusters": clusters,
                    "nprobe": effective_nprobe,
                    "requested_nprobe": args.nprobe,
                    "prep_time_s": prep_time_s,
                    "artifact_prep_time_s": artifact_prep_time_s,
                    "artifact_source": artifact_source,
                    "saq_data_dir": str(ds_dir),
                    "centroid_src": str(centroid_src),
                    "cids_src": str(cids_src),
                    "ts": time.time(),
                },
            )

            for bits in sorted(set(float(b) for b in args.bits)):
                for adj_rounds in sorted(set(int(v) for v in args.caq_adj_rd_lmt)):
                    for bound_m in sorted(set(float(v) for v in args.searcher_vars_bound_m)):
                        run_dir = dataset_dir / f"K{clusters}_b{bits:g}_rd{adj_rounds}_sm{bound_m:g}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        meta_json = run_dir / "metadata.json"

                        if meta_json.exists() and not args.force_index:
                            rows.append(json.loads(meta_json.read_text())["row"])
                            continue

                        common_flags = [
                            f"-dataset={dataset}",
                            f"-K={clusters}",
                            f"-B={bits:g}",
                            "-enable_PCA=true",
                            "-enable_segmentation=true",
                            f"-rand_rotate={args.rand_rotate}",
                            f"-use_fastscan={args.use_fastscan}",
                            f"-caq_adj_rd_lmt={adj_rounds}",
                            f"-caq_adj_eps={args.caq_adj_eps}",
                            f"-searcher_vars_bound_m={bound_m:g}",
                            f"-nprobe={effective_nprobe}",
                            f"-n_runs={args.n_runs}",
                            f"-warmup_runs={args.warmup_runs}",
                        ]
                        if args.num_threads:
                            common_flags.append(f"-num_threads={args.num_threads}")

                        _append_jsonl(
                            progress_log,
                            {
                                "event": "start_run",
                                "dataset": dataset,
                                "clusters": clusters,
                                "bits": bits,
                                "caq_adj_rd_lmt": adj_rounds,
                                "searcher_vars_bound_m": bound_m,
                                "nprobe": effective_nprobe,
                                "requested_nprobe": args.nprobe,
                                "ts": time.time(),
                            },
                        )

                        t0 = time.perf_counter()
                        _run([str(create_index), *common_flags], cwd=saq_root)
                        encoding_time_s = time.perf_counter() - t0
                        args_token = _saq_args_token(
                            clusters,
                            bits,
                            args.rand_rotate,
                            args.use_fastscan,
                            adj_rounds,
                            args.caq_adj_eps,
                        )
                        index_csv = saq_root / "results" / "saq" / f"{dataset}_{args_token}.index.csv"
                        if index_csv.exists():
                            try:
                                encoding_time_s = float(_read_first_csv_row(index_csv)["index_time_s"])
                            except Exception:
                                pass

                        _run([str(test_relative_error), *common_flags], cwd=saq_root)
                        relerr_csv = saq_root / "results" / "saq" / (
                            f"{dataset}_{args_token}_sm{bound_m:g}.csv"
                        )
                        if not relerr_csv.exists():
                            raise FileNotFoundError(f"Expected SAQ eval CSV missing: {relerr_csv}")
                        metrics = _read_first_csv_row(relerr_csv)

                        row = {
                            "method": "SAQ",
                            "dataset": dataset,
                            "experiment_folder": str(run_dir),
                            "nq": nq,
                            "nb": nb,
                            "nb_sample": nb,
                            "dim": spec.dim,
                            "n_subquantizers": 0,
                            "nbits": bits,
                            "bits_per_vector": spec.dim * bits,
                            "train_size": nb,
                            "train_time_s": float(prep_time_s),
                            "encoding_time_s": float(encoding_time_s),
                            "distance_table_time_s": np.nan,
                            "cdist_time_s": np.nan,
                            "adc_time_s": float(metrics["adc_time_s_mean"]),
                            "adc_time_s_std": float(metrics["adc_time_s_std"]),
                            "per_query_us_mean": float(metrics["per_query_us_mean"]),
                            "per_pair_ns_mean": float(metrics["per_pair_ns_mean"]),
                            "rel_error_mean": float(metrics["err_tot_avg"]),
                            "rel_error_std": np.nan,
                            "nq_sample": int(metrics["nq"]),
                            "sample_mode": f"saq_native_nprobe={metrics['nprobe']}",
                            "train_path": str(ds_dir / f"{dataset}_base.fvecs"),
                            "seed": np.nan,
                            "n_centroids": clusters,
                            "nprobe": int(metrics["nprobe"]),
                            "requested_nprobe": args.nprobe,
                            "caq_adj_rd_lmt": adj_rounds,
                            "use_fastscan": args.use_fastscan == "true",
                            "searcher_vars_bound_m": bound_m,
                            "centroid_fvecs": str(centroid_src),
                            "cluster_ids_ivecs": str(cids_src),
                            "eval_csv": str(relerr_csv),
                            "index_csv": str(index_csv),
                        }

                        payload = {
                            "dataset": dataset,
                            "clusters": clusters,
                            "bits": bits,
                            "caq_adj_rd_lmt": adj_rounds,
                            "use_fastscan": args.use_fastscan == "true",
                            "searcher_vars_bound_m": bound_m,
                            "row": row,
                            "metrics": metrics,
                        }
                        meta_json.write_text(json.dumps(payload, indent=2))
                        rows.append(row)

                        _append_jsonl(
                            progress_log,
                            {
                                "event": "finish_run",
                                "dataset": dataset,
                                "clusters": clusters,
                                "bits": bits,
                                "caq_adj_rd_lmt": adj_rounds,
                                "use_fastscan": args.use_fastscan == "true",
                                "searcher_vars_bound_m": bound_m,
                                "nprobe": effective_nprobe,
                                "requested_nprobe": args.nprobe,
                                "adc_time_s": row["adc_time_s"],
                                "per_pair_ns_mean": row["per_pair_ns_mean"],
                                "rel_error_mean": row["rel_error_mean"],
                                "ts": time.time(),
                            },
                        )

        if rows:
            rows.sort(key=lambda row: (row["n_centroids"], row["nbits"], row["caq_adj_rd_lmt"], row["searcher_vars_bound_m"]))
            out_csv = get_results_root() / "saq" / f"{dataset}_SAQ_adc_vs_exact_eval.csv"
            merged_rows = _merge_summary_rows(_load_rows_csv(out_csv), rows)
            merged_rows.sort(
                key=lambda row: (
                    float(row["n_centroids"]),
                    float(row["nbits"]),
                    float(row["caq_adj_rd_lmt"]),
                    float(row["searcher_vars_bound_m"]),
                )
            )
            _write_rows_csv(out_csv, merged_rows)
            _append_jsonl(
                progress_log,
                {
                    "event": "write_csv",
                    "dataset": dataset,
                    "path": str(out_csv),
                    "rows": len(merged_rows),
                    "rows_added_or_updated": len(rows),
                    "ts": time.time(),
                },
            )
            print(out_csv, flush=True)


if __name__ == "__main__":
    main()
