#!/usr/bin/env python3
"""Build relerr_cpp_plots_per_figure_config.ipynb from relerr_cpp_plots.ipynb."""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "relerr_cpp_plots.ipynb"
DST = HERE / "relerr_cpp_plots_per_figure_config.ipynb"

CONFIG_CELL = r'''# =============================================================================
# Per-figure titles, font sizes, axis names, and x-ticks
# -----------------------------------------------------------------------------
# Each plot passes a `figure_key`. Entries here are merged on top of `_defaults`.
# Use None for x_label / y_label / title to keep the notebook's automatic labels.
#
# Common keys produced by this notebook:
#   f"{metric}__{x_col}"               e.g. "relerr__nbits", "spearman__n_subquantizers"
#   f"{metric}__compression_rate"      compression-rate on x-axis
#   f"additional__{y_col}__{x_col}"    rows from ADDITIONAL_PLOTS
#   "distortion__nbits" / "distortion__n_subquantizers"
#   f"pq_quality__{y_col}__{x_col}"    PQ quality metrics cell
#   f"bar__{y_col}__{x_col}"           if you enable BAR_PLOTS
#
# Other cells (train-size, Pareto, paper ADC, OPQ vs PQ, …) still use their local
# font sizes; extend those blocks to call `resolve_figure_style` the same way if needed.
# =============================================================================

FIGURE_CONFIG = {
    "_defaults": {
        "title": None,
        "title_fontsize": 40,
        "axis_label_fontsize": 40,
        "tick_label_fontsize": 39,
        "sci_offset_fontsize": 27,
        "x_label": None,
        "y_label": None,
        "xticks": None,
        "xticklabels": None,
        "y_axis_label_coords": None,
        "nbits_x_margin": None,
    },
    # --- Examples (uncomment / edit) ---
    # "relerr__nbits": {
    #     "y_axis_label_coords": (-0.22, 0.38),
    #     "nbits_x_margin": 0.08,
    #     # "xticks": [4, 6, 8, 10, 12],
    #     # "xticklabels": None,
    # },
}


def resolve_figure_style(figure_key: str | None) -> dict:
    base = dict(FIGURE_CONFIG.get("_defaults", {}))
    if figure_key and figure_key in FIGURE_CONFIG:
        base.update(FIGURE_CONFIG[figure_key])
    return base

'''


def patch_plot_cell(s: str) -> str:
    s = s.replace(
        "    num_subq_plot_subquantizers=None,\n):",
        "    num_subq_plot_subquantizers=None,\n    figure_key: str | None = None,\n):",
        1,
    )

    needle = (
        "    if datasets is None:\n"
        "        datasets = sorted(df[\"dataset\"].unique())\n    \n"
        "    if output_dir is None:\n"
    )
    inject = (
        "    if datasets is None:\n"
        "        datasets = sorted(df[\"dataset\"].unique())\n\n"
        "    _sty = resolve_figure_style(figure_key)\n"
        "    _axis_lbl_fs = _sty.get(\"axis_label_fontsize\", 40)\n"
        "    _tick_lbl_fs = _sty.get(\"tick_label_fontsize\", 39)\n"
        "    _sci_off_fs = _sty.get(\"sci_offset_fontsize\", 27)\n\n"
        "    if output_dir is None:\n"
    )
    if needle not in s:
        raise RuntimeError("plot_relerr_vs_x: datasets/output_dir anchor not found")
    s = s.replace(needle, inject, 1)

    old_xy = (
        "            ax.set_xlabel(x_label, fontsize=40)\n"
        "            # Set y-axis label - use provided y_label or generate from y_col\n"
        "            if y_label is None:\n"
    )
    new_xy = (
        "            # Set y-axis label - use provided y_label or generate from y_col\n"
        "            if y_label is None:\n"
    )
    if old_xy not in s:
        raise RuntimeError("plot_relerr_vs_x: xlabel block not found")
    s = s.replace(old_xy, new_xy, 1)

    old_y2 = (
        "                    y_label = y_col.replace(\"_\", \" \").title()\n"
        "            ax.set_ylabel(y_label, fontsize=40)\n"
    )
    new_y2 = (
        "                    y_label = y_col.replace(\"_\", \" \").title()\n"
        "            _x_eff = x_label if _sty.get(\"x_label\") is None else _sty[\"x_label\"]\n"
        "            _y_eff = y_label if _sty.get(\"y_label\") is None else _sty[\"y_label\"]\n"
        "            ax.set_xlabel(_x_eff, fontsize=_axis_lbl_fs)\n"
        "            ax.set_ylabel(_y_eff, fontsize=_axis_lbl_fs)\n"
    )
    if old_y2 not in s:
        raise RuntimeError("plot_relerr_vs_x: ylabel block not found")
    s = s.replace(old_y2, new_y2, 1)

    old_coords = (
        "            if y_col == \"rel_error_mean\" and x_col == \"nbits\":\n"
        "                # Avg relative error vs Bps (curves = N_subq): pull y-label left of y-tick numerals.\n"
        "                ax.yaxis.set_label_coords(-0.22, 0.38)\n"
        "            elif y_col == \"rel_error_mean\" and x_col == \"n_subquantizers\":\n"
        "                ax.yaxis.set_label_coords(-0.14, 0.38)\n"
    )
    new_coords = (
        "            _yc = _sty.get(\"y_axis_label_coords\")\n"
        "            if _yc is not None:\n"
        "                ax.yaxis.set_label_coords(*_yc)\n"
        "            elif y_col == \"rel_error_mean\" and x_col == \"nbits\":\n"
        "                ax.yaxis.set_label_coords(-0.22, 0.38)\n"
        "            elif y_col == \"rel_error_mean\" and x_col == \"n_subquantizers\":\n"
        "                ax.yaxis.set_label_coords(-0.14, 0.38)\n"
    )
    if old_coords not in s:
        raise RuntimeError("plot_relerr_vs_x: label coords block not found")
    s = s.replace(old_coords, new_coords, 1)

    s = s.replace(
        "            ax.tick_params(labelsize=39)",
        "            ax.tick_params(labelsize=_tick_lbl_fs)",
        1,
    )
    s = s.replace(
        "                    axis.offsetText.set_fontsize(27)",
        "                    axis.offsetText.set_fontsize(_sci_off_fs)",
        1,
    )

    tight_anchor = "            # Apply tight_layout first\n            plt.tight_layout()"
    if tight_anchor not in s:
        raise RuntimeError("tight_layout anchor not found")
    s = s.replace(
        tight_anchor,
        "            _ttl = _sty.get(\"title\")\n"
        "            if _ttl:\n"
        "                ax.set_title(_ttl, fontsize=_sty.get(\"title_fontsize\", _axis_lbl_fs))\n"
        "            # Apply tight_layout first\n            plt.tight_layout()",
        1,
    )

    old_nbits = (
        "            elif x_col == \"nbits\":\n"
        "                # Explicitly show every unique nbits value on the x-axis\n"
        "                unique_nbits = sorted(sub[x_col].unique())\n"
        "                tick_values_to_use = unique_nbits\n"
        "                ax.set_xticks(unique_nbits)\n"
        "                if y_col == \"rel_error_mean\":\n"
        "                    # Slightly widen the Bps axis span for avg relative error vs Bps figures\n"
        "                    ax.margins(x=0.08)\n"
    )
    new_nbits = (
        "            elif x_col == \"nbits\":\n"
        "                unique_nbits = sorted(sub[x_col].unique())\n"
        "                tick_values_to_use = unique_nbits\n"
        "                if _sty.get(\"xticks\") is not None:\n"
        "                    ax.set_xticks(_sty[\"xticks\"])\n"
        "                    tick_values_to_use = list(_sty[\"xticks\"])\n"
        "                    if _sty.get(\"xticklabels\") is not None:\n"
        "                        ax.set_xticklabels(_sty[\"xticklabels\"])\n"
        "                else:\n"
        "                    ax.set_xticks(unique_nbits)\n"
        "                _xm = _sty.get(\"nbits_x_margin\")\n"
        "                if _xm is None and y_col == \"rel_error_mean\":\n"
        "                    _xm = 0.08\n"
        "                if _xm is not None:\n"
        "                    ax.margins(x=float(_xm))\n"
    )
    if old_nbits not in s:
        raise RuntimeError("nbits axis block not found")
    s = s.replace(old_nbits, new_nbits, 1)

    old_call = (
        "        plot_relerr_vs_x(\n"
        "            plot_df,\n"
        "            x_col,\n"
        "            x_label,\n"
        "            y_col=y_col,\n"
        "            y_label=y_label,\n"
        "            methods=METHODS_TO_PLOT,\n"
        "            datasets=DATASETS_TO_PLOT,\n"
        "            ylim=ylim,\n"
        "            output_dir=PQ_FAISS_FIGURES_DIR,\n"
        "            nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n"
        "            num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n"
        "        )\n"
    )
    new_call = (
        "        plot_relerr_vs_x(\n"
        "            plot_df,\n"
        "            x_col,\n"
        "            x_label,\n"
        "            y_col=y_col,\n"
        "            y_label=y_label,\n"
        "            methods=METHODS_TO_PLOT,\n"
        "            datasets=DATASETS_TO_PLOT,\n"
        "            ylim=ylim,\n"
        "            output_dir=PQ_FAISS_FIGURES_DIR,\n"
        "            nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n"
        "            num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n"
        '            figure_key=f"{metric}__{x_col}",\n'
        "        )\n"
    )
    if old_call not in s:
        raise RuntimeError("main plot_relerr_vs_x call not found")
    s = s.replace(old_call, new_call, 1)

    old_add = (
        "    plot_relerr_vs_x(\n"
        "        plot_df,\n"
        "        x_col,\n"
        "        x_label,\n"
        "        y_col=y_col,\n"
        "        y_label=y_label,\n"
        "        methods=METHODS_TO_PLOT,\n"
        "        datasets=DATASETS_TO_PLOT,\n"
        "        ylim=ylim,\n"
        "        group_by=group_by,\n"
        "        output_dir=PQ_FAISS_FIGURES_DIR,\n"
        "        nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n"
        "        num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n"
        "    )\n"
    )
    new_add = (
        "    plot_relerr_vs_x(\n"
        "        plot_df,\n"
        "        x_col,\n"
        "        x_label,\n"
        "        y_col=y_col,\n"
        "        y_label=y_label,\n"
        "        methods=METHODS_TO_PLOT,\n"
        "        datasets=DATASETS_TO_PLOT,\n"
        "        ylim=ylim,\n"
        "        group_by=group_by,\n"
        "        output_dir=PQ_FAISS_FIGURES_DIR,\n"
        "        nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n"
        "        num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n"
        '        figure_key=f"additional__{y_col}__{x_col}",\n'
        "    )\n"
    )
    if old_add not in s:
        raise RuntimeError("ADDITIONAL plot_relerr_vs_x call not found")
    s = s.replace(old_add, new_add, 1)

    s = s.replace(
        "    num_subq_plot_subquantizers=None,\n):\n    \"\"\"\n    Plot chosen metric vs compression rate",
        "    num_subq_plot_subquantizers=None,\n    figure_key: str | None = None,\n):\n    \"\"\"\n    Plot chosen metric vs compression rate",
        1,
    )
    d0 = s.find("def plot_compression_rate_vs_y")
    d1 = s.find("def plot_opq_vs_pq_rot_percent")
    chunk = s[d0:d1]
    comp_needle = (
        "    if datasets is None:\n"
        "        datasets = sorted(df[\"dataset\"].unique())\n\n"
        "    if output_dir is None:\n"
    )
    comp_inject = (
        "    if datasets is None:\n"
        "        datasets = sorted(df[\"dataset\"].unique())\n\n"
        "    _sty = resolve_figure_style(figure_key)\n"
        "    _axis_lbl_fs = _sty.get(\"axis_label_fontsize\", 40)\n"
        "    _tick_lbl_fs = _sty.get(\"tick_label_fontsize\", 39)\n\n"
        "    if output_dir is None:\n"
    )
    if comp_needle not in chunk:
        raise RuntimeError("compression datasets block not found")
    chunk = chunk.replace(comp_needle, comp_inject, 1)
    s = s[:d0] + chunk + s[d1:]

    old_comp_axes = (
        '            ax.set_xlabel("Compression rate", fontsize=40)\n'
        "            ax.set_ylabel(y_label, fontsize=40)\n"
        "            ax.tick_params(labelsize=39)\n"
    )
    new_comp_axes = (
        '            _cx = "Compression rate" if _sty.get("x_label") is None else _sty["x_label"]\n'
        "            _cy = y_label if _sty.get(\"y_label\") is None else _sty[\"y_label\"]\n"
        "            ax.set_xlabel(_cx, fontsize=_axis_lbl_fs)\n"
        "            ax.set_ylabel(_cy, fontsize=_axis_lbl_fs)\n"
        "            ax.tick_params(labelsize=_tick_lbl_fs)\n"
        "            _ttl = _sty.get(\"title\")\n"
        "            if _ttl:\n"
        "                ax.set_title(_ttl, fontsize=_sty.get(\"title_fontsize\", _axis_lbl_fs))\n"
    )
    if old_comp_axes not in s:
        raise RuntimeError("compression axis block not found")
    s = s.replace(old_comp_axes, new_comp_axes, 1)

    old_comp_call = (
        "        plot_compression_rate_vs_y(\n"
        "            plot_df,\n"
        "            y_col=y_col,\n"
        "            y_label=y_label,\n"
        "            methods=METHODS_TO_PLOT,\n"
        "            datasets=DATASETS_TO_PLOT,\n"
        "            group_by=COMPRESSION_RATE_GROUPING,\n"
        "            output_dir=PQ_FAISS_FIGURES_DIR,\n"
        "            nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n"
        "            num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n"
        "        )\n"
    )
    new_comp_call = (
        "        plot_compression_rate_vs_y(\n"
        "            plot_df,\n"
        "            y_col=y_col,\n"
        "            y_label=y_label,\n"
        "            methods=METHODS_TO_PLOT,\n"
        "            datasets=DATASETS_TO_PLOT,\n"
        "            group_by=COMPRESSION_RATE_GROUPING,\n"
        "            output_dir=PQ_FAISS_FIGURES_DIR,\n"
        "            nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n"
        "            num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n"
        '            figure_key=f"{metric}__compression_rate",\n'
        "        )\n"
    )
    if old_comp_call not in s:
        raise RuntimeError("compression invocation not found")
    s = s.replace(old_comp_call, new_comp_call, 1)

    s = s.replace(
        "    output_dir: Path = None\n):\n    \"\"\"\n    Plot bar chart (bin-style)",
        "    output_dir: Path = None,\n    figure_key: str | None = None,\n):\n    \"\"\"\n    Plot bar chart (bin-style)",
        1,
    )
    bar_a = s.find("def plot_bar_chart")
    bar_b = s.find("def plot_opq_vs_pq_rot_percent")
    bchunk = s[bar_a:bar_b]
    # First datasets/output_dir block uses a whitespace-only spacer line (not a bare \\n\\n).
    bar_needle_bar = (
        "    if datasets is None:\n"
        "        datasets = sorted(df[\"dataset\"].unique())\n    \n"
        "    if output_dir is None:\n"
    )
    bar_inject = (
        "    if datasets is None:\n"
        "        datasets = sorted(df[\"dataset\"].unique())\n\n"
        "    _sty = resolve_figure_style(figure_key)\n"
        "    _axis_lbl_fs = _sty.get(\"axis_label_fontsize\", 40)\n"
        "    _tick_lbl_fs = _sty.get(\"tick_label_fontsize\", 39)\n\n"
        "    if output_dir is None:\n"
    )
    if bar_needle_bar not in bchunk:
        raise RuntimeError("bar chart datasets block not found")
    bchunk = bchunk.replace(bar_needle_bar, bar_inject, 1)
    s = s[:bar_a] + bchunk + s[bar_b:]

    old_bar_axes = (
        "# Set labels and styling\n"
        "            ax.set_xlabel(x_label, fontsize=40)\n"
        "            if y_label is None:\n"
        "                if y_col == \"train_time_s\":\n"
        "                    y_label = \"Training time (seconds)\"\n"
        "                elif y_col == \"adc_time_s\":\n"
        "                    y_label = \"ADC time (seconds)\"\n"
        "                elif y_col == \"distance_table_time_s\":\n"
        "                    y_label = \"Distance table time (seconds)\"\n"
        "                    y_label = y_col.replace(\"_\", \" \").title()\n"
        "            ax.set_ylabel(y_label, fontsize=40)\n"
        "            ax.tick_params(labelsize=39)\n"
        "            \n"
        "            "
    )
    new_bar_axes = (
        "# Set labels and styling\n"
        "            if y_label is None:\n"
        "                if y_col == \"train_time_s\":\n"
        "                    y_label = \"Training time (seconds)\"\n"
        "                elif y_col == \"adc_time_s\":\n"
        "                    y_label = \"ADC time (seconds)\"\n"
        "                elif y_col == \"distance_table_time_s\":\n"
        "                    y_label = \"Distance table time (seconds)\"\n"
        "                    y_label = y_col.replace(\"_\", \" \").title()\n"
        "            _x_eff = x_label if _sty.get(\"x_label\") is None else _sty[\"x_label\"]\n"
        "            _y_eff = y_label if _sty.get(\"y_label\") is None else _sty[\"y_label\"]\n"
        "            ax.set_xlabel(_x_eff, fontsize=_axis_lbl_fs)\n"
        "            ax.set_ylabel(_y_eff, fontsize=_axis_lbl_fs)\n"
        "            ax.tick_params(labelsize=_tick_lbl_fs)\n"
        "            _ttl = _sty.get(\"title\")\n"
        "            if _ttl:\n"
        "                ax.set_title(_ttl, fontsize=_sty.get(\"title_fontsize\", _axis_lbl_fs))\n"
        "            \n"
        "            "
    )
    if old_bar_axes not in s:
        raise RuntimeError("bar chart axis block not found")
    s = s.replace(old_bar_axes, new_bar_axes, 1)

    old_bar_call = (
        "    plot_bar_chart(\n"
        "        plot_df,\n"
        "        x_col,\n"
        "        x_label,\n"
        "        y_col=y_col,\n"
        "        y_label=y_label,\n"
        "        methods=METHODS_TO_PLOT,\n"
        "        datasets=DATASETS_TO_PLOT,\n"
        "        group_by=average_over,\n"
        "        output_dir=PQ_FAISS_FIGURES_DIR\n"
        "    )\n"
    )
    new_bar_call = (
        "    plot_bar_chart(\n"
        "        plot_df,\n"
        "        x_col,\n"
        "        x_label,\n"
        "        y_col=y_col,\n"
        "        y_label=y_label,\n"
        "        methods=METHODS_TO_PLOT,\n"
        "        datasets=DATASETS_TO_PLOT,\n"
        "        group_by=average_over,\n"
        "        output_dir=PQ_FAISS_FIGURES_DIR,\n"
        '        figure_key=f"bar__{y_col}__{x_col}",\n'
        "    )\n"
    )
    if old_bar_call in s:
        s = s.replace(old_bar_call, new_bar_call, 1)

    return s


def main() -> None:
    shutil.copy2(SRC, DST)
    nb = json.loads(DST.read_text())

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [CONFIG_CELL],
    }
    nb["cells"].insert(2, new_cell)

    plot_idx = next(
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and "def plot_relerr_vs_x(" in "".join(c.get("source", []))
    )
    src = "".join(nb["cells"][plot_idx]["source"])
    nb["cells"][plot_idx]["source"] = [patch_plot_cell(src)]

    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "# Distortion vs nbits" in src and "plot_relerr_vs_x(" in src:
            src = src.replace(
                "    nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n)",
                '    nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n    figure_key="distortion__nbits",\n)',
                1,
            )
            c["source"] = [src]
        elif "# Distortion vs M" in src and "plot_relerr_vs_x(" in src:
            src = src.replace(
                "    num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n)",
                '    num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n    figure_key="distortion__n_subquantizers",\n)',
                1,
            )
            c["source"] = [src]
        elif "pq_quality_df" in src and "plot_relerr_vs_x(" in src:
            old_pq = (
                "            plot_relerr_vs_x(\n"
                "                pq_quality_df,\n"
                "                x_col=x_col,\n"
                "                x_label=x_label,\n"
                "                y_col=y_col,\n"
                "                y_label=y_label,\n"
                "                methods=[\"PQ\"],\n"
                "                datasets=DATASETS_TO_PLOT,\n"
                "                output_dir=PQ_QUALITY_FIGURES_DIR,\n"
                "                nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n"
                "                num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n"
                "            )\n"
            )
            new_pq = (
                "            plot_relerr_vs_x(\n"
                "                pq_quality_df,\n"
                "                x_col=x_col,\n"
                "                x_label=x_label,\n"
                "                y_col=y_col,\n"
                "                y_label=y_label,\n"
                "                methods=[\"PQ\"],\n"
                "                datasets=DATASETS_TO_PLOT,\n"
                "                output_dir=PQ_QUALITY_FIGURES_DIR,\n"
                "                nbits_subquantizers=NBITS_PLOT_SUBQUANTIZERS,\n"
                "                num_subq_plot_subquantizers=NUM_SUBQ_PLOT_SUBQUANTIZERS,\n"
                '                figure_key=f"pq_quality__{y_col}__{x_col}",\n'
                "            )\n"
            )
            if old_pq in src:
                c["source"] = [src.replace(old_pq, new_pq, 1)]

    DST.write_text(json.dumps(nb))
    print("Wrote", DST)


if __name__ == "__main__":
    main()
