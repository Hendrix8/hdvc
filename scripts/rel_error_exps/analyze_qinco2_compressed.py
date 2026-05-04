#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

for _anc in Path(__file__).resolve().parents:
    if (_anc / "hdvc_paths.py").is_file():
        if str(_anc) not in sys.path:
            sys.path.insert(0, str(_anc))
        break
from hdvc_paths import get_results_root  # noqa: E402
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
    }
)


ZIP_NAME_RE = re.compile(r"^(?P<dataset>[a-zA-Z0-9_]+)_M_(?P<M>\d+)_K_(?P<K>\d+)\.zip$")
SETTING_RE = re.compile(r"/setting_(\d+)/")
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VAL_MSE_RE = re.compile(rf"validation MSE=\s*({FLOAT_RE})")
TRAIN_MSE_RE = re.compile(rf"train set MSE=\s*({FLOAT_RE})")
BEST_MIN_MSE_RE = re.compile(rf"min_MSE=\s*({FLOAT_RE})")


@dataclass
class SettingRecord:
    zip_path: str
    dataset: str
    M: int
    K: int
    setting_id: int
    bits_per_vector: float
    A: Optional[int] = None
    B: Optional[int] = None
    L: Optional[int] = None
    de: Optional[int] = None
    dh: Optional[int] = None
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    has_codes: bool = False
    has_distance_file: bool = False
    distance_relerr_mean: Optional[float] = None
    distance_relerr_std: Optional[float] = None
    val_mse_best: Optional[float] = None
    train_mse_last: Optional[float] = None
    best_min_mse: Optional[float] = None
    status: str = "unknown"


def safe_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def parse_zip_identity(zip_fp: Path) -> Tuple[str, int, int]:
    m = ZIP_NAME_RE.match(zip_fp.name)
    if not m:
        raise ValueError(f"Malformed archive name: {zip_fp.name}")
    return m.group("dataset"), int(m.group("M")), int(m.group("K"))


def parse_setting_ids(names: List[str]) -> List[int]:
    ids = set()
    for n in names:
        m = SETTING_RE.search(n)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def parse_conf(zf: zipfile.ZipFile, pref: str) -> Dict:
    for conf_name in (f"{pref}/conf_0.json", f"{pref}/conf.json"):
        try:
            with zf.open(conf_name) as f:
                return json.load(f)
        except KeyError:
            continue
    return {}


def parse_runlog_metrics(zf: zipfile.ZipFile, pref: str) -> Dict[str, Optional[float]]:
    metrics = {"val_mse_best": None, "train_mse_last": None, "best_min_mse": None}
    try:
        with zf.open(f"{pref}/run.log") as f:
            txt = f.read().decode("utf-8", errors="replace")
    except KeyError:
        return metrics

    val = [float(x) for x in VAL_MSE_RE.findall(txt)]
    tr = [float(x) for x in TRAIN_MSE_RE.findall(txt)]
    bm = [float(x) for x in BEST_MIN_MSE_RE.findall(txt)]

    if val:
        metrics["val_mse_best"] = min(val)
    if tr:
        metrics["train_mse_last"] = tr[-1]
    if bm:
        metrics["best_min_mse"] = min(bm)
    return metrics


def compute_relerr_from_npz(zf: zipfile.ZipFile, npz_name: str) -> Tuple[Optional[float], Optional[float]]:
    with zf.open(npz_name) as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=False)
        keys = list(data.keys())
        exact_key = next((k for k in keys if "exact" in k.lower() and "dist" in k.lower()), None)
        approx_key = next((k for k in keys if ("approx" in k.lower() or "adc" in k.lower()) and "dist" in k.lower()), None)
        if exact_key is None or approx_key is None:
            return None, None
        exact = np.asarray(data[exact_key], dtype=np.float64)
        approx = np.asarray(data[approx_key], dtype=np.float64)
        if exact.shape != approx.shape:
            return None, None
        denom = np.maximum(exact, 1e-6)
        rel = np.abs(approx - exact) / denom
        rel = rel[np.isfinite(rel)]
        if rel.size == 0:
            return None, None
        return float(np.mean(rel)), float(np.std(rel))


def discover_distance_file(names: List[str], setting_id: int) -> Optional[str]:
    target = f"/setting_{setting_id}/"
    for n in names:
        ln = n.lower()
        if target in n and ln.endswith(".npz") and "compute_distances" in ln:
            return n
    return None


def has_codes_for_setting(names: List[str], setting_id: int) -> bool:
    target = f"/setting_{setting_id}/"
    for n in names:
        if target in n and n.endswith(".npz") and ("encode_train" in n or "encode_test" in n):
            return True
    return False


def build_setting_record(zip_fp: Path) -> List[SettingRecord]:
    dataset, M, K = parse_zip_identity(zip_fp)
    bits_per_vector = M * math.log2(K)
    records: List[SettingRecord] = []

    with zipfile.ZipFile(zip_fp, "r") as zf:
        names = zf.namelist()
        setting_ids = parse_setting_ids(names)
        if not setting_ids:
            raise ValueError(f"No setting folders found in {zip_fp.name}")

        for sid in setting_ids:
            pref = f"{zip_fp.stem}/setting_{sid}"
            root_conf = parse_conf(zf, pref)
            encode_conf = parse_conf(zf, f"{pref}/encode_test/setting_0")
            merged = {}
            merged.update(root_conf)
            merged.update(encode_conf)

            m = parse_runlog_metrics(zf, pref)
            dist_npz = discover_distance_file(names, sid)
            d_mean, d_std = (None, None)
            if dist_npz:
                d_mean, d_std = compute_relerr_from_npz(zf, dist_npz)

            has_codes = has_codes_for_setting(names, sid)
            has_dist = dist_npz is not None and d_mean is not None
            status = "config_only"
            if has_codes and has_dist:
                status = "distance_ready"
            elif has_codes:
                status = "codes_only"
            elif any(v is not None for v in m.values()):
                status = "logs_only"

            rec = SettingRecord(
                zip_path=str(zip_fp),
                dataset=dataset,
                M=M,
                K=K,
                setting_id=sid,
                bits_per_vector=bits_per_vector,
                A=safe_int(merged.get("A") or merged.get("train_A")),
                B=safe_int(merged.get("B") or merged.get("train_B")),
                L=safe_int(merged.get("L")),
                de=safe_int(merged.get("de")),
                dh=safe_int(merged.get("dh")),
                train_size=safe_int(merged.get("n_train")),
                val_size=safe_int(merged.get("n_val")),
                has_codes=has_codes,
                has_distance_file=has_dist,
                distance_relerr_mean=d_mean,
                distance_relerr_std=d_std,
                val_mse_best=m["val_mse_best"],
                train_mse_last=m["train_mse_last"],
                best_min_mse=m["best_min_mse"],
                status=status,
            )
            records.append(rec)

    return records


def to_rows(records: List[SettingRecord]) -> pd.DataFrame:
    base_rows = []
    metric_rows = []
    for r in records:
        base_rows.append(vars(r))

        if r.distance_relerr_mean is not None:
            metric_rows.append(
                {
                    "dataset": r.dataset,
                    "M": r.M,
                    "K": r.K,
                    "setting_id": r.setting_id,
                    "bits_per_vector": r.bits_per_vector,
                    "A": r.A,
                    "B": r.B,
                    "L": r.L,
                    "de": r.de,
                    "dh": r.dh,
                    "metric_type": "distance_relerr",
                    "rel_error_mean": r.distance_relerr_mean,
                    "rel_error_std": r.distance_relerr_std,
                    "status": r.status,
                    "zip_path": r.zip_path,
                }
            )

        mse_val = r.val_mse_best if r.val_mse_best is not None else r.best_min_mse
        if mse_val is not None:
            metric_rows.append(
                {
                    "dataset": r.dataset,
                    "M": r.M,
                    "K": r.K,
                    "setting_id": r.setting_id,
                    "bits_per_vector": r.bits_per_vector,
                    "A": r.A,
                    "B": r.B,
                    "L": r.L,
                    "de": r.de,
                    "dh": r.dh,
                    "metric_type": "mse_proxy",
                    "rel_error_mean": mse_val,
                    "rel_error_std": np.nan,
                    "status": r.status,
                    "zip_path": r.zip_path,
                }
            )

    return pd.DataFrame(base_rows), pd.DataFrame(metric_rows)


def sanity_checks(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        raise ValueError("No metric rows extracted from archives.")
    if metrics[["dataset", "M", "K", "setting_id"]].isna().any().any():
        raise ValueError("Missing required id fields in extracted rows.")
    if (~np.isfinite(metrics["rel_error_mean"].astype(float))).any():
        raise ValueError("Non-finite rel_error_mean found.")


def plot_dataset(metrics_ds: pd.DataFrame, out_fp: Path, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    color_palette = [
        "tab:blue",
        "tab:green",
        "tab:purple",
        "tab:orange",
        "tab:red",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    marker_palette = ["o", "v", "s", "^", "D", "<", ">", "p", "*", "h"]

    k_values = sorted(metrics_ds["K"].dropna().unique().tolist())
    k_style = {
        int(k): {
            "color": color_palette[i % len(color_palette)],
            "marker": marker_palette[i % len(marker_palette)],
        }
        for i, k in enumerate(k_values)
    }
    labeled_k = set()

    plotted_any = False
    for metric_type in ("distance_relerr", "mse_proxy"):
        sub = metrics_ds[metrics_ds["metric_type"] == metric_type].copy()
        if sub.empty:
            continue
        linestyle = "-" if metric_type == "distance_relerr" else "--"
        alpha = 0.95 if metric_type == "distance_relerr" else 0.8
        for k, g in sub.groupby("K"):
            k = int(k)
            st = k_style[k]
            g = g.sort_values("bits_per_vector")
            label = f"K={k}" if k not in labeled_k else None
            labeled_k.add(k)
            ax.plot(
                g["bits_per_vector"],
                g["rel_error_mean"],
                color=st["color"],
                marker=st["marker"],
                linewidth=2,
                markersize=8,
                markeredgewidth=1.5,
                markeredgecolor="black",
                linestyle=linestyle,
                alpha=alpha,
                label=label,
            )
            plotted_any = True

    ax.set_title(f"{dataset}")
    ax.set_xlabel("bits per vector")
    ax.set_ylabel("relative error")
    ax.grid(False)
    ax.grid(alpha=0.8, axis="y", linestyle="--")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No distance-based data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
        )
    fig.subplots_adjust(right=0.72)
    fig.savefig(out_fp, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path, summary_lines: List[str]) -> None:
    content = [
        "# Qinco2 Compressed Analysis",
        "",
        "This folder contains analysis generated from compressed Qinco2 archives only.",
        "",
        "Metric types:",
        "- `distance_relerr`: paper-style relative distance error when exact/approx distances are available in archive npz.",
        "- `mse_proxy`: fallback proxy from run logs (`validation MSE` / `min_MSE`) when distance arrays are unavailable.",
        "- `bits_per_vector`: `M * log2(K)`.",
        "",
        "Coverage summary:",
        "",
    ] + [f"- {line}" for line in summary_lines]
    (out_dir / "README.md").write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze compressed Qinco2 result archives and generate PDFs + CSVs.")
    ap.add_argument(
        "--input_dir",
        default=str(get_results_root() / "qinco2" / "compressed_qinco_res"),
        help="Directory containing Qinco2 zip archives.",
    )
    ap.add_argument(
        "--output_dir",
        default=str(get_results_root() / "qinco2" / "compressed_analysis"),
        help="Output directory for CSV/PDF analysis artifacts.",
    )
    ap.add_argument(
        "--distance_only",
        action="store_true",
        help="Keep only true distance-based relative error rows (drop mse_proxy).",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(in_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip archives found in {in_dir}")

    records: List[SettingRecord] = []
    for z in zips:
        records.extend(build_setting_record(z))

    manifest_df, metrics_df = to_rows(records)
    if args.distance_only:
        metrics_df = metrics_df[metrics_df["metric_type"] == "distance_relerr"].copy()
    sanity_checks(metrics_df)

    manifest_fp = out_dir / "qinco2_compressed_manifest.csv"
    metrics_fp = out_dir / "qinco2_compressed_metrics_long.csv"
    manifest_df.to_csv(manifest_fp, index=False)
    metrics_df.to_csv(metrics_fp, index=False)

    for dataset, g in metrics_df.groupby("dataset"):
        ds_dir = out_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        g.to_csv(ds_dir / f"{dataset}_metrics.csv", index=False)
        plot_dataset(g, ds_dir / f"{dataset}_relative_error_plots.pdf", dataset)

    n_archives = len(zips)
    n_settings = len(manifest_df)
    n_dist = int((manifest_df["has_distance_file"] == True).sum())
    n_proxy = int(metrics_df["metric_type"].eq("mse_proxy").sum())
    by_ds = []
    for dataset, g in manifest_df.groupby("dataset"):
        dist_ready = int(g["has_distance_file"].sum())
        pct = (100.0 * dist_ready / len(g)) if len(g) else 0.0
        by_ds.append(f"{dataset}: {dist_ready}/{len(g)} settings distance-ready ({pct:.1f}%)")

    summary_lines = [
        f"Archives processed: {n_archives}",
        f"Settings processed: {n_settings}",
        f"Distance-level relerr available: {n_dist} settings",
        f"Rows using mse_proxy metric: {n_proxy}",
        *by_ds,
    ]
    write_readme(out_dir, summary_lines)

    print("Analysis complete.")
    for line in summary_lines:
        print(line)
    print(f"Manifest CSV: {manifest_fp}")
    print(f"Metrics CSV: {metrics_fp}")


if __name__ == "__main__":
    main()

