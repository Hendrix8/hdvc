#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Avg relative error from README-style QINCo2 CSV exports."
    )
    p.add_argument(
        "--csv_dir",
        default="/data/cpanourg/2-hdvc/results/qinco2/compressed_qinco_res/figures2",
        help="Directory containing *_QINCo2_adc_vs_exact_eval.csv files.",
    )
    p.add_argument(
        "--out_dir",
        default="/data/cpanourg/2-hdvc/results/qinco2/compressed_qinco_res/figures2/readme_relerr_plots",
        help="Directory where plots will be written.",
    )
    return p.parse_args()


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    df = df.copy()
    for col in ("n_subquantizers", "nbits", "bits_per_vector", "rel_error_mean"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["n_subquantizers", "nbits", "bits_per_vector", "rel_error_mean"])
    if df.empty:
        return

    df["M"] = df["n_subquantizers"].astype(int)
    df["K"] = (2 ** df["nbits"].astype(int)).astype(int)

    colors = plt.get_cmap("tab10")

    # relerr vs M (line per K)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (k, g) in enumerate(sorted(df.groupby("K"), key=lambda kv: kv[0])):
        g = g.sort_values("M")
        ax.plot(
            g["M"],
            g["rel_error_mean"],
            marker="o",
            linewidth=2,
            color=colors(i % 10),
            label=f"K={k}",
        )
    ax.set_title(f"{dataset}: Avg relative error vs M")
    ax.set_xlabel("M (n_subquantizers)")
    ax.set_ylabel("Avg relative error")
    ax.grid(alpha=0.25, linestyle="--", axis="y")
    ax.legend(frameon=False, ncol=2)
    _save(fig, out_dir / f"{dataset}_avg_relerr_vs_M.png")

    # relerr vs bits_per_vector (line per K)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (k, g) in enumerate(sorted(df.groupby("K"), key=lambda kv: kv[0])):
        g = g.sort_values("bits_per_vector")
        ax.plot(
            g["bits_per_vector"],
            g["rel_error_mean"],
            marker="o",
            linewidth=2,
            color=colors(i % 10),
            label=f"K={k}",
        )
    ax.set_title(f"{dataset}: Avg relative error vs bits/vector")
    ax.set_xlabel("bits_per_vector")
    ax.set_ylabel("Avg relative error")
    ax.grid(alpha=0.25, linestyle="--", axis="y")
    ax.legend(frameon=False, ncol=2)
    _save(fig, out_dir / f"{dataset}_avg_relerr_vs_bits_per_vector.png")

    # heatmap M x K
    pivot = (
        df.pivot_table(index="M", columns="K", values="rel_error_mean", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    mat = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(f"{dataset}: Avg relative error heatmap")
    ax.set_xlabel("K")
    ax.set_ylabel("M")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(v)) for v in pivot.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Avg relative error")
    _save(fig, out_dir / f"{dataset}_avg_relerr_heatmap_M_by_K.png")


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    csvs = sorted(csv_dir.glob("*_QINCo2_adc_vs_exact_eval.csv"))
    if not csvs:
        raise FileNotFoundError(f"No QINCo2 CSV files found in {csv_dir}")

    summary = []
    for csv_fp in csvs:
        dataset = csv_fp.name.replace("_QINCo2_adc_vs_exact_eval.csv", "")
        df = pd.read_csv(csv_fp)
        _plot_dataset(df, dataset, out_dir)
        summary.append((dataset, len(df)))

    print(f"Wrote plots to {out_dir}")
    for ds, n in summary:
        print(f"{ds}: {n} rows")


if __name__ == "__main__":
    main()
