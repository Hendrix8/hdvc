"""
Match fonts, grid, spines, and line/marker styling from relerr_cpp_plots.ipynb
for simple rel_error_mean vs nbits (or bits_per_vector) figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# --- Same as relerr_cpp_plots.ipynb cell 1 ---
# Base style - individual axes still use fontsize=40 on labels/ticks like the notebook.
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 40,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 21,
    }
)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# --- Same as relerr_cpp_plots.ipynb cell 3 ---
COLOR_PALETTE = [
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
    "tab:blue",
    "tab:green",
    "tab:purple",
    "tab:orange",
    "tab:red",
]

MARKER_PALETTE = [
    "o",
    "v",
    "s",
    "^",
    "D",
    "<",
    ">",
    "p",
    "*",
    "h",
    "H",
    "X",
    "d",
    "P",
    "8",
]


def apply_relerr_cpp_rcparams() -> None:
    """Re-apply rcParams (safe to call multiple times)."""
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 40,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 21,
        }
    )
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def style_axes_relerr_vs_metric(
    ax,
    *,
    x_label: str,
    y_label: str,
    x_col: str = "nbits",
    x_values=None,
    legend_loc_outside: bool = True,
) -> None:
    """
    Match relerr_cpp plot_relerr_vs_x() styling for labels, ticks, grid, spines.
    For x_col == 'nbits', set all integer ticks like the notebook.
    """
    ax.set_xlabel(x_label, fontsize=40)
    ax.set_ylabel(y_label, fontsize=40)
    ax.tick_params(labelsize=40)

    if x_col == "nbits" and x_values is not None:
        unique_nbits = sorted(set(x_values))
        ax.set_xticks(unique_nbits)

    # Override default grid: relerr_cpp uses y-only dashed grid
    ax.grid(False)
    ax.grid(alpha=0.8, axis="y", linestyle="--")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if legend_loc_outside and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.05, 0.5))
    elif ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="best")


def plot_curve_relerr_style(
    ax,
    df_sorted,
    *,
    x_col: str,
    y_col: str,
    label: str,
    color: str,
    marker: str,
) -> None:
    """Single curve: linewidth 2, markersize 12, black edge — same as plot_relerr_vs_x."""
    group_df_sorted = df_sorted.sort_values(x_col)
    ax.plot(
        group_df_sorted[x_col],
        group_df_sorted[y_col],
        label=label,
        color=color,
        marker=marker,
        markersize=12,
        linewidth=2,
        markeredgewidth=2,
        markeredgecolor="black",
    )


def savefig_relerr(fig, path: Path, *, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure_relerr_vs_nbits(
    df,
    *,
    method_label: str,
    y_col: str = "rel_error_mean",
    y_axis_label: str | None = None,
    x_axis_label: str = "nbits",
):
    """
    One figure: rel_error_mean (or y_col) vs nbits, matching relerr_cpp_plots line/marker/axes style.
    """
    if y_axis_label is None:
        y_axis_label = (
            "Relative error"
            if y_col == "rel_error_mean"
            else y_col.replace("_", " ").title()
        )

    apply_relerr_cpp_rcparams()
    fig, ax = plt.subplots()

    plot_curve_relerr_style(
        ax,
        df,
        x_col="nbits",
        y_col=y_col,
        label=method_label,
        color=COLOR_PALETTE[0],
        marker=MARKER_PALETTE[0],
    )
    style_axes_relerr_vs_metric(
        ax,
        x_label=x_axis_label,
        y_label=y_axis_label,
        x_col="nbits",
        x_values=df["nbits"].values,
        legend_loc_outside=True,
    )
    plt.tight_layout()
    return fig, ax


def figure_relerr_vs_nbits_multi(
    series: list[tuple[object, str]],
    *,
    y_col: str = "rel_error_mean",
    y_axis_label: str | None = None,
    x_axis_label: str = "nbits",
    title: str | None = None,
):
    """
    Multiple curves on one axes: each entry is (DataFrame, legend_label).
    DataFrames must include columns `nbits` and `y_col`.
    """
    if y_axis_label is None:
        y_axis_label = (
            "Relative error"
            if y_col == "rel_error_mean"
            else y_col.replace("_", " ").title()
        )

    apply_relerr_cpp_rcparams()
    fig, ax = plt.subplots()

    all_nbits: list = []
    for i, (df, label) in enumerate(series):
        plot_curve_relerr_style(
            ax,
            df,
            x_col="nbits",
            y_col=y_col,
            label=label,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            marker=MARKER_PALETTE[i % len(MARKER_PALETTE)],
        )
        all_nbits.extend(df["nbits"].values.tolist())

    if title:
        ax.set_title(title, fontsize=28)

    style_axes_relerr_vs_metric(
        ax,
        x_label=x_axis_label,
        y_label=y_axis_label,
        x_col="nbits",
        x_values=all_nbits,
        legend_loc_outside=True,
    )
    plt.tight_layout()
    return fig, ax
