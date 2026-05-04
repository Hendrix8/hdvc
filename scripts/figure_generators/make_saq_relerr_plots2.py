#!/usr/bin/env python3
"""Generate saq_relerr_plots2.ipynb with LSQ++-style SAQ plotting logic."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


NOTEBOOK = Path(__file__).with_name("saq_relerr_plots2.ipynb")

cells = [
    markdown(
        """# SAQ Evaluation Plots 2

LSQ++-style plotting notebook for SAQ results.

This notebook follows the same flow as `lsqpp_relerr_plots.ipynb`:

- load aggregate `*_SAQ_adc_vs_exact_eval.csv` files
- aggregate by the full SAQ configuration
- produce one figure per dataset
- plot averaged trends separately from explicit configuration-legend trends
- save every figure as PDF and SVG

The output directory is `figures2` so these plots do not overwrite the earlier SAQ notebook outputs.
"""
    ),
    code(
        r'''from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 40,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 21,
})
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

PROJECT_ROOT = Path("/home/cpanourg/projects/2-hdvc")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path("/data/cpanourg/2-hdvc/results/saq")
FIGURES_DIR = DATA_DIR / "figures2"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METHODS_TO_PLOT = ["SAQ"]
DATASETS_TO_PLOT = ["deep", "bigann", "gist", "msmarco", "openai"]
FIGURE_SAVE_FORMATS = ("pdf", "svg")

ADC_TIME_COL = "adc_time_per_pair_s"
ADC_TIME_LABEL = "ADC time per pair (s)"
CONFIG_COLS = ["method", "dataset", "nbits", "bits_per_vector"]
'''
    ),
    code(
        r'''# Color and marker palettes copied from the LSQ++ notebook.
COLOR_PALETTE = [
    "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:red",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]
MARKER_PALETTE = ["o", "v", "s", "^", "D", "<", ">", "p", "*", "h"]


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in FIGURE_SAVE_FORMATS:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    print("Saved " + " and ".join(str(p) for p in saved))


def style_axes(ax, tick_fontsize=40, grid_axis="y"):
    ax.grid(True, axis=grid_axis, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_fontsize)


def adjust_ylabel_position(ax, y_label: str):
    if y_label == "Avg Relative Error":
        ax.yaxis.set_label_coords(-0.13, 0.39)


def set_sci_axes(ax):
    for axis in [ax.xaxis, ax.yaxis]:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 3))
        axis.set_major_formatter(fmt)
'''
    ),
    code(
        r'''def load_saq_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    csv_paths = sorted(data_dir.glob("*_SAQ_adc_vs_exact_eval.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No SAQ CSV files found in {data_dir}")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df["dataset"] = path.name.split("_")[0]
        if "method" not in df.columns:
            df["method"] = "SAQ"
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "bits_per_vector", "nbits", "train_size", "adc_time_s", "rel_error_mean", "rel_error_std",
        "train_time_s", "encoding_time_s", "distance_table_time_s", "cdist_time_s",
        "nb_sample", "nq_sample", "dim", "nb", "nq", "n_subquantizers", "seed",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["method"] = df["method"].replace({"SAQ": "SAQ"})
    df["pair_count"] = df["nb_sample"] * df["nq_sample"]
    df[ADC_TIME_COL] = df["adc_time_s"] / df["pair_count"]
    df["bpv_ratio"] = df["bits_per_vector"] / df["dim"]

    required = ["method", "dataset", "bits_per_vector", "nbits", "adc_time_s", "rel_error_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


raw_df = load_saq_data(DATA_DIR)
print("Rows per dataset:")
display(raw_df.groupby("dataset").size())
raw_df.sort_values(["dataset", "nbits"]).head(20)
'''
    ),
    code(
        r'''def aggregate_metric_df(df: pd.DataFrame, y_cols=("rel_error_mean", ADC_TIME_COL)) -> pd.DataFrame:
    agg_spec = {c: "mean" for c in y_cols if c in df.columns}
    for c in [
        "rel_error_std", "distance_table_time_s", "cdist_time_s",
        "train_time_s", "encoding_time_s", "dim", "nb", "nb_sample",
        "nq_sample", "pair_count", "adc_time_s", "bpv_ratio", "train_size",
    ]:
        if c in df.columns and c not in agg_spec:
            agg_spec[c] = "mean" if pd.api.types.is_numeric_dtype(df[c]) else "first"
    return (
        df.groupby(CONFIG_COLS, as_index=False)
        .agg(agg_spec)
        .sort_values(["dataset", "nbits"])
    )


plot_df = aggregate_metric_df(raw_df)
plot_df.groupby("dataset").size()
'''
    ),
    code(
        r'''def plot_avg_metric_vs_param(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    output_stem: str,
    datasets=DATASETS_TO_PLOT,
    methods=METHODS_TO_PLOT,
    output_dir: Path = FIGURES_DIR,
):
    for method in methods:
        for dataset in datasets:
            sub = df[(df["method"] == method) & (df["dataset"] == dataset)].dropna(subset=[x_col, y_col]).copy()
            if sub.empty:
                print(f"Skipping {method}/{dataset}: no rows")
                continue
            agg = sub.groupby([x_col], as_index=False)[y_col].agg(["mean", "std"]).reset_index().sort_values(x_col)
            fig, ax = plt.subplots()
            yerr = agg["std"].fillna(0)
            ax.errorbar(
                agg[x_col], agg["mean"], yerr=yerr,
                fmt="o-", color=COLOR_PALETTE[0], markersize=16,
                linewidth=2.5, markeredgewidth=2, markeredgecolor="black",
                capsize=5, capthick=2, elinewidth=1.5,
            )
            ax.set_xlabel(x_label, fontsize=40)
            ax.set_ylabel(y_label, fontsize=40)
            adjust_ylabel_position(ax, y_label)
            if x_col in ("bits_per_vector", "nbits", "bpv_ratio", "train_size"):
                vals = sorted(agg[x_col].dropna().unique())
                ax.set_xticks(vals)
                ax.set_xticklabels([
                    str(int(v)) if float(v).is_integer() else f"{v:.3g}" for v in vals
                ], rotation=0)
            style_axes(ax, tick_fontsize=34, grid_axis="y")
            set_sci_axes(ax)
            plt.tight_layout()
            stem = f"{output_stem}_{method.lower()}_{dataset}"
            save_figure(fig, output_dir, stem)
            plt.show()
            plt.close(fig)
'''
    ),
    code(
        r'''def _format_legend_value(col: str, value):
    if pd.isna(value):
        return f"{col}=NA"
    if col == "bits_per_vector":
        return f"bpv={int(value)}"
    if col == "nbits":
        return f"B={int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"{col}={int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{col}={int(value)}" if float(value).is_integer() else f"{col}={value:.3g}"
    return f"{col}={value}"


def _legend_label(row, legend_cols):
    return ", ".join(_format_legend_value(col, row[col]) for col in legend_cols)


def plot_metric_with_config_legend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    legend_cols: list,
    output_stem: str,
    datasets=DATASETS_TO_PLOT,
    methods=METHODS_TO_PLOT,
    output_dir: Path = FIGURES_DIR,
):
    for method in methods:
        for dataset in datasets:
            needed = [x_col, y_col] + legend_cols
            sub = df[(df["method"] == method) & (df["dataset"] == dataset)].dropna(subset=needed).copy()
            if sub.empty:
                print(f"Skipping {method}/{dataset}: no rows")
                continue
            agg = (
                sub.groupby([x_col] + legend_cols, as_index=False)[y_col]
                .mean()
                .sort_values(legend_cols + [x_col])
            )
            legend_settings = agg[legend_cols].drop_duplicates().sort_values(legend_cols).reset_index(drop=True)
            fig, ax = plt.subplots()
            for idx, setting in legend_settings.iterrows():
                mask = np.ones(len(agg), dtype=bool)
                for col in legend_cols:
                    mask &= agg[col].eq(setting[col])
                curve = agg.loc[mask].sort_values(x_col)
                color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
                marker = MARKER_PALETTE[idx % len(MARKER_PALETTE)]
                ax.plot(
                    curve[x_col], curve[y_col], marker=marker, linestyle="-",
                    color=color, markersize=12, linewidth=2.5,
                    markeredgewidth=1.5, markeredgecolor="black",
                    label=_legend_label(setting, legend_cols),
                )
            ax.set_xlabel(x_label, fontsize=40)
            ax.set_ylabel(y_label, fontsize=40)
            adjust_ylabel_position(ax, y_label)
            if x_col in ("bits_per_vector", "nbits", "bpv_ratio", "train_size"):
                vals = sorted(agg[x_col].dropna().unique())
                ax.set_xticks(vals)
                ax.set_xticklabels([
                    str(int(v)) if float(v).is_integer() else f"{v:.3g}" for v in vals
                ], rotation=0)
            style_axes(ax, tick_fontsize=34, grid_axis="y")
            set_sci_axes(ax)
            ax.legend(frameon=True, fontsize=13, loc="best")
            plt.tight_layout()
            stem = f"{output_stem}_{method.lower()}_{dataset}"
            save_figure(fig, output_dir, stem)
            plt.show()
            plt.close(fig)
'''
    ),
    markdown("## Avg Relative Error Plots\n"),
    code(
        r'''# Averaged trends: same LSQ++ logic, one figure per dataset.
plot_avg_metric_vs_param(
    plot_df,
    x_col="bits_per_vector",
    y_col="rel_error_mean",
    x_label="Bits per vector",
    y_label="Avg Relative Error",
    output_stem="avg_relerr_vs_bits_per_vector",
)

plot_avg_metric_vs_param(
    plot_df,
    x_col="nbits",
    y_col="rel_error_mean",
    x_label="SAQ bits per dimension",
    y_label="Avg Relative Error",
    output_stem="avg_relerr_vs_nbits",
)

plot_metric_with_config_legend(
    plot_df,
    x_col="bits_per_vector",
    y_col="rel_error_mean",
    x_label="Bits per vector",
    y_label="Avg Relative Error",
    legend_cols=["nbits"],
    output_stem="avg_relerr_vs_bits_per_vector_with_config_legend",
)

plot_metric_with_config_legend(
    plot_df,
    x_col="nbits",
    y_col="rel_error_mean",
    x_label="SAQ bits per dimension",
    y_label="Avg Relative Error",
    legend_cols=["bits_per_vector"],
    output_stem="avg_relerr_vs_nbits_with_config_legend",
)
'''
    ),
    markdown("## ADC Time Plots\n"),
    code(
        r'''plot_avg_metric_vs_param(
    plot_df,
    x_col="bits_per_vector",
    y_col=ADC_TIME_COL,
    x_label="Bits per vector",
    y_label=ADC_TIME_LABEL,
    output_stem="adc_time_vs_bits_per_vector",
)

plot_avg_metric_vs_param(
    plot_df,
    x_col="nbits",
    y_col=ADC_TIME_COL,
    x_label="SAQ bits per dimension",
    y_label=ADC_TIME_LABEL,
    output_stem="adc_time_vs_nbits",
)

plot_metric_with_config_legend(
    plot_df,
    x_col="bits_per_vector",
    y_col=ADC_TIME_COL,
    x_label="Bits per vector",
    y_label=ADC_TIME_LABEL,
    legend_cols=["nbits"],
    output_stem="adc_time_vs_bits_per_vector_with_config_legend",
)

plot_metric_with_config_legend(
    plot_df,
    x_col="nbits",
    y_col=ADC_TIME_COL,
    x_label="SAQ bits per dimension",
    y_label=ADC_TIME_LABEL,
    legend_cols=["bits_per_vector"],
    output_stem="adc_time_vs_nbits_with_config_legend",
)
'''
    ),
    markdown("## Encoding Time Plots\n"),
    code(
        r'''plot_avg_metric_vs_param(
    plot_df,
    x_col="bits_per_vector",
    y_col="encoding_time_s",
    x_label="Bits per vector",
    y_label="Encoding time (s)",
    output_stem="encoding_time_vs_bits_per_vector",
)

plot_avg_metric_vs_param(
    plot_df,
    x_col="nbits",
    y_col="encoding_time_s",
    x_label="SAQ bits per dimension",
    y_label="Encoding time (s)",
    output_stem="encoding_time_vs_nbits",
)
'''
    ),
    markdown("## Pareto Plots\n"),
    code(
        r'''def pareto_frontier_minimize(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    pts = df.sort_values([x_col, y_col]).copy()
    frontier_rows = []
    best_y = np.inf
    for _, row in pts.iterrows():
        if row[y_col] < best_y:
            frontier_rows.append(row)
            best_y = row[y_col]
    if not frontier_rows:
        return pts.iloc[0:0]
    return pd.DataFrame(frontier_rows)


def plot_pareto(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    label_cols: list,
    output_stem: str,
    datasets=DATASETS_TO_PLOT,
    methods=METHODS_TO_PLOT,
    output_dir: Path = FIGURES_DIR,
):
    for method in methods:
        for dataset in datasets:
            sub = df[(df["method"] == method) & (df["dataset"] == dataset)].dropna(subset=[x_col, y_col]).copy()
            if sub.empty:
                print(f"Skipping {method}/{dataset}: no rows")
                continue
            frontier = pareto_frontier_minimize(sub, x_col, y_col)
            fig, ax = plt.subplots()
            ax.scatter(
                sub[x_col], sub[y_col],
                color="tab:blue", marker="o", s=260,
                edgecolors="black", linewidths=2, alpha=0.75,
            )
            if len(frontier) > 0:
                ax.plot(frontier[x_col], frontier[y_col], "-", color="gray", linewidth=1.5, alpha=0.8)
                ax.scatter(
                    frontier[x_col], frontier[y_col],
                    color="#D32F2F", marker="o", s=330,
                    edgecolors="black", linewidths=2, zorder=3,
                )
            for _, row in frontier.iterrows():
                label = ", ".join(_format_legend_value(col, row[col]) for col in label_cols if col in row.index)
                ax.annotate(
                    label, (row[x_col], row[y_col]),
                    xytext=(10, 10), textcoords="offset points", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72),
                    arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8, shrinkA=0, shrinkB=5),
                )
            ax.set_xlabel(x_label, fontsize=40)
            ax.set_ylabel(y_label, fontsize=40)
            adjust_ylabel_position(ax, y_label)
            style_axes(ax, tick_fontsize=34, grid_axis="both")
            set_sci_axes(ax)
            plt.tight_layout()
            stem = f"{output_stem}_{method.lower()}_{dataset}"
            save_figure(fig, output_dir, stem)
            plt.show()
            plt.close(fig)


plot_pareto(
    plot_df,
    x_col=ADC_TIME_COL,
    y_col="rel_error_mean",
    x_label=ADC_TIME_LABEL,
    y_label="Avg Relative Error",
    label_cols=["bits_per_vector", "nbits"],
    output_stem="pareto_relerr_vs_adc_time",
)
'''
    ),
]

NOTEBOOK.write_text(
    json.dumps(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "pygments_lexer": "ipython3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=1,
    )
)
print(f"Wrote {NOTEBOOK}")
