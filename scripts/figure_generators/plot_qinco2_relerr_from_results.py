#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot QINCo2 average relative error against hyperparameters."
    )
    p.add_argument(
        "--manifest_csv",
        default="/data/cpanourg/2-hdvc/results/qinco2/compressed_qinco_res/figures2/qinco2_compressed_manifest.csv",
        help="Path to qinco2 manifest CSV.",
    )
    p.add_argument(
        "--out_dir",
        default="/data/cpanourg/2-hdvc/results/qinco2/compressed_qinco_res/figures2",
        help="Output directory for figures.",
    )
    return p.parse_args()


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_true_relerr(df: pd.DataFrame, out_dir: Path) -> None:
    true_df = df[np.isfinite(df["distance_relerr_mean"])].copy()
    if true_df.empty:
        return

    true_df["bits_per_vector"] = pd.to_numeric(true_df["bits_per_vector"], errors="coerce")
    true_df["M"] = pd.to_numeric(true_df["M"], errors="coerce")
    true_df["K"] = pd.to_numeric(true_df["K"], errors="coerce")
    true_df["distance_relerr_mean"] = pd.to_numeric(
        true_df["distance_relerr_mean"], errors="coerce"
    )
    true_df = true_df.dropna(subset=["bits_per_vector", "M", "K", "distance_relerr_mean"])

    # Scatter: avg relerr vs bits/vector, colored by K, marker-size by M
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(true_df["K"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {k: cmap(i % 10) for i, k in enumerate(ks)}
    for k, g in true_df.groupby("K"):
        ax.scatter(
            g["bits_per_vector"],
            g["distance_relerr_mean"],
            s=np.clip(g["M"].astype(float) * 8.0, 30.0, 250.0),
            color=color_map[k],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.7,
            label=f"K={int(k)}",
        )
    ax.set_xlabel("bits per vector (M*log2(K))")
    ax.set_ylabel("Avg relative error (true ADC)")
    ax.set_title("QINCo2 true ADC relative error vs bitrate")
    ax.grid(alpha=0.25, linestyle="--", axis="y")
    ax.legend(frameon=False)
    _save(fig, out_dir / "qinco2_true_adc_relerr_vs_bpv_scatter.png")

    # Heatmap over (M, K) if enough points, otherwise single-point plot.
    pivot = true_df.pivot_table(
        index="M", columns="K", values="distance_relerr_mean", aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    im = ax.imshow(masked, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(v)) for v in pivot.index])
    ax.set_xlabel("K")
    ax.set_ylabel("M")
    ax.set_title("QINCo2 true ADC relative error over (M, K)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Avg relative error")
    _save(fig, out_dir / "qinco2_true_adc_relerr_heatmap_M_by_K.png")


def _plot_proxy_mse(df: pd.DataFrame, out_dir: Path) -> None:
    # This is not ADC relerr; shown only as a hyperparameter trend surrogate.
    mdf = df.copy()
    mdf["best_proxy"] = pd.to_numeric(
        mdf["distance_relerr_mean"].where(np.isfinite(mdf["distance_relerr_mean"]), np.nan),
        errors="coerce",
    )
    no_true = ~np.isfinite(mdf["best_proxy"])
    mdf.loc[no_true, "best_proxy"] = pd.to_numeric(mdf.loc[no_true, "best_min_mse"], errors="coerce")
    no_true = ~np.isfinite(mdf["best_proxy"])
    mdf.loc[no_true, "best_proxy"] = pd.to_numeric(mdf.loc[no_true, "val_mse_best"], errors="coerce")

    mdf["bits_per_vector"] = pd.to_numeric(mdf["bits_per_vector"], errors="coerce")
    mdf["M"] = pd.to_numeric(mdf["M"], errors="coerce")
    mdf["K"] = pd.to_numeric(mdf["K"], errors="coerce")
    mdf = mdf.dropna(subset=["bits_per_vector", "M", "K", "best_proxy"])
    if mdf.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, g in sorted(mdf.groupby("dataset"), key=lambda kv: kv[0]):
        grp = g.groupby("bits_per_vector", as_index=False)["best_proxy"].mean().sort_values("bits_per_vector")
        ax.plot(grp["bits_per_vector"], grp["best_proxy"], marker="o", linewidth=2, label=dataset)
    ax.set_xlabel("bits per vector (M*log2(K))")
    ax.set_ylabel("Proxy value (true relerr if available, else MSE)")
    ax.set_title("QINCo2 hyperparameter trend (proxy, not pure ADC relerr)")
    ax.grid(alpha=0.25, linestyle="--", axis="y")
    ax.legend(frameon=False, ncol=2)
    _save(fig, out_dir / "qinco2_hparam_trend_proxy_relerr_or_mse.png")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_csv)
    out_dir = Path(args.out_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    _plot_true_relerr(df, out_dir)
    _plot_proxy_mse(df, out_dir)

    n_true = int(np.isfinite(pd.to_numeric(df["distance_relerr_mean"], errors="coerce")).sum())
    print(f"Loaded {len(df)} rows from {manifest_path}")
    print(f"Rows with true ADC relerr: {n_true}")
    print(f"Wrote figures under: {out_dir}")


if __name__ == "__main__":
    main()
