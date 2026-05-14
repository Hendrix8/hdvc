#!/usr/bin/env python3
"""
Generate one PQ figure notebook per main-loop (y-metric, x-column) pair from
scripts/figure_generators/relerr_cpp_plots.ipynb.

Run from anywhere:
  python3 scripts/fig_notebooks/pq/generate_split_notebooks.py

Writes only:
  scripts/fig_notebooks/pq/xy/*.ipynb

Or run the same plot from the CLI (recommended when you need to fill
``reconstruction_error`` first):

  python3 scripts/figure_generators/pq_xy_figures.py --help

Pairs are Y_METRICS × X_COLUMNS_TO_PLOT from the source cell 2, plus
EXTRA_XY_METRICS_ALL_X: same Y_METRIC_MAP keys repeated for every x column (used so
**Distortion error** gets one notebook per x-axis, mirroring avg relative error).
Filenames use ``distortion_error`` for the ``recon_error`` metric key; CONFIG still sets
``TARGET_Y_METRIC = "recon_error"``. Requires non-null ``reconstruction_error`` in
``*_reconstruction_error.csv`` under DATA_DIR (see ``scripts/evals/run_evals.py``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "scripts" / "figure_generators" / "relerr_cpp_plots.ipynb"
OUT_DIR = Path(__file__).resolve().parent

# Y_METRIC_MAP keys to duplicate for every entry in X_COLUMNS_TO_PLOT (deduped against
# Y_METRICS × X). Default adds distortion (recon_error) vs each x, same x set as relerr.
EXTRA_XY_METRICS_ALL_X: tuple[str, ...] = ("recon_error",)

APPLY_USER_OVERRIDES = """
# ----- Apply USER_* overrides from the CONFIG cell (run before this cell) -----
# None means "keep the default from this cell" — only non-None USER_* values override.
for _k in (
    "METHODS_TO_PLOT",
    "DATASETS_TO_PLOT",
    "Y_METRICS",
    "YLIM_CONFIG",
    "NBITS_PLOT_SUBQUANTIZERS",
    "NUM_SUBQ_PLOT_SUBQUANTIZERS",
    "BITS_PER_VECTOR_GROUPING",
    "COMPRESSION_RATE_GROUPING",
    "ADC_TIME_UNIT",
    "X_COLUMNS_TO_PLOT",
    "ADDITIONAL_PLOTS",
    "BAR_PLOTS",
):
    _uk = "USER_" + _k
    if _uk in globals() and globals()[_uk] is not None:
        globals()[_k] = globals()[_uk]

"""


def _cell_code(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text],
    }


def _cell_md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def _join_src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _split_c3_exec(c3: str) -> tuple[str, str, str, str]:
    split = c3.find("# Generate plots based on configuration")
    if split < 0:
        raise RuntimeError("Could not find main execution marker in cell 3")
    defs = c3[:split]
    tail = c3[split:]
    m1 = tail.find("# Generate additional plots")
    m2 = tail.find("# Generate compression rate")
    if m1 < 0 or m2 < 0:
        raise RuntimeError("Could not slice execution tail")
    return defs, tail[:m1], tail[m1:m2], tail[m2:]


PATH_OVERRIDES = """# ----- Optional path overrides (USER_* from CONFIG cell) -----
if "USER_DATA_DIR" in globals() and USER_DATA_DIR is not None:
    DATA_DIR = Path(USER_DATA_DIR)
if "USER_PQ_FAISS_ADC_SUMMARY" in globals() and USER_PQ_FAISS_ADC_SUMMARY is not None:
    PQ_FAISS_ADC_SUMMARY = Path(USER_PQ_FAISS_ADC_SUMMARY)
    PQ_FAISS_FIGURES_DIR = PQ_FAISS_ADC_SUMMARY.parent / "figures"

"""


def _patch_cell2_with_apply(c2: str) -> str:
    c2 = PATH_OVERRIDES + c2
    anchor = "plot_df = build_plot_df(relerr_df, DATA_DIR)"
    if anchor not in c2:
        raise RuntimeError("plot_df anchor not found in cell 2")
    # Insert apply block immediately after build_plot_df line (keep .head() etc.)
    idx = c2.find(anchor)
    line_end = c2.find("\n", idx)
    if line_end < 0:
        raise RuntimeError("newline after plot_df line")
    insert_at = line_end + 1
    return c2[:insert_at] + APPLY_USER_OVERRIDES + c2[insert_at:]


def _nb_shell(metadata: dict) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": metadata,
        "cells": [],
    }


def _write(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1))


# Line slices are tied to scripts/figure_generators/relerr_cpp_plots.ipynb cell 2;
# if that cell shifts, update these ranges (see ADC_TIME_UNIT / X_COLUMNS_TO_PLOT / Y_METRICS).
_C2_ADC_AND_XCOL_SLICE = slice(149, 168)  # 1-based lines 150–168
_C2_Y_METRICS_SLICE = slice(248, 259)  # Y_METRICS + Y_METRIC_MAP block


def _discover_main_xy_pairs(
    c1: str,
    c2_raw: str,
    *,
    extra_metrics_for_all_x: tuple[str, ...] = (),
) -> list[tuple[str, str, str, dict]]:
    """
    Pairs (Y_METRIC key, x column name, x axis label, exec_env) for the main
    relerr-vs-x loop: product of Y_METRICS and X_COLUMNS_TO_PLOT as in the source notebook,
    plus each ``extra_metrics_for_all_x`` key paired with every x column not already covered.
    """
    g: dict = {"__builtins__": __builtins__, "Path": Path}
    exec(compile(c1, "<cell1>", "exec"), g, g)
    lines = c2_raw.splitlines()
    exec(compile("\n".join(lines[_C2_ADC_AND_XCOL_SLICE]), "<xcols>", "exec"), g, g)
    exec(compile("\n".join(lines[_C2_Y_METRICS_SLICE]), "<ymetrics>", "exec"), g, g)
    out: list[tuple[str, str, str, dict]] = []
    seen: set[tuple[str, str]] = set()
    for m in g["Y_METRICS"]:
        for xc, xl in g["X_COLUMNS_TO_PLOT"]:
            out.append((m, xc, xl, g))
            seen.add((m, xc))
    ymap = g["Y_METRIC_MAP"]
    xcols = g["X_COLUMNS_TO_PLOT"]
    for yk in extra_metrics_for_all_x:
        if yk not in ymap:
            raise RuntimeError(
                f"extra_metrics_for_all_x: unknown metric {yk!r} (not in Y_METRIC_MAP)"
            )
        for xc, xl in xcols:
            if (yk, xc) in seen:
                continue
            out.append((yk, xc, xl, g))
            seen.add((yk, xc))
    return out


def _y_axis_title(metric: str, g: dict) -> str:
    if metric == "recall":
        return "Recall @1, @10, @100"
    mm = g.get("Y_METRIC_MAP", {})
    if metric in mm:
        return str(mm[metric][1])
    return metric


def _safe_filename_part(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.ASCII)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _xy_filename_stem_metric(metric_key: str) -> str:
    """Human-oriented file prefix; Y_METRIC_MAP key ``recon_error`` → ``distortion_error``."""
    if metric_key == "recon_error":
        return "distortion_error"
    return _safe_filename_part(metric_key)


XY_SWEEP_PIN = """
# ----- Single (y-metric, x-column) for this split notebook -----
Y_METRICS = [TARGET_Y_METRIC]
_adc_xcols = ("adc_cpu_time_pp", "adc_cpu_time_pp_ms")
_x_pin = [t for t in X_COLUMNS_TO_PLOT if t[0] == TARGET_X_COL]
if len(_x_pin) != 1 and TARGET_X_COL in _adc_xcols:
    _x_pin = [t for t in X_COLUMNS_TO_PLOT if t[0] in _adc_xcols]
if len(_x_pin) != 1:
    raise ValueError(
        "TARGET_X_COL=%r must match exactly one entry in X_COLUMNS_TO_PLOT "
        "(after USER_* overrides); got columns %r"
        % (TARGET_X_COL, [t[0] for t in X_COLUMNS_TO_PLOT])
    )
X_COLUMNS_TO_PLOT = _x_pin
del _x_pin, _adc_xcols

"""


def _cfg_intro_for_xy(cfg_intro: str, target_y: str, target_x: str) -> str:
    old = """# Metrics: names must exist in Y_METRIC_MAP in the data cell (e.g. relerr, spearman, recall, recon_error)
USER_Y_METRICS = ["relerr"]

"""
    new = f"""# Pin one main-loop figure (same keys as Y_METRICS in the data cell)
TARGET_Y_METRIC = {target_y!r}  # relerr | spearman | recall | recon_error (Distortion error → recon_error)
TARGET_X_COL = {target_x!r}  # e.g. n_subquantizers, nbits, bits_per_vector, adc_cpu_time_pp / adc_cpu_time_pp_ms

USER_Y_METRICS = None  # xy split: use TARGET_Y_METRIC (plot cell pins the sweep)

"""
    if old not in cfg_intro:
        raise RuntimeError("cfg_intro template changed; update _cfg_intro_for_xy")
    return cfg_intro.replace(old, new, 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nb_src = json.loads(SRC.read_text())
    meta = nb_src.get("metadata", {})

    c1 = _join_src(nb_src["cells"][1])
    c2_raw = _join_src(nb_src["cells"][2])
    c2 = _patch_cell2_with_apply(c2_raw)
    c3 = _join_src(nb_src["cells"][3])
    c3_defs, c3_main, _c3_add, _c3_comp = _split_c3_exec(c3)

    cfg_intro = """# =============================================================================
# CONFIG — edit USER_* assignments, then Run All.
# Any USER_* you set to a non-None value overrides the default in the data cell below.
# Leave USER_* as None to keep that default (e.g. USER_X_COLUMNS_TO_PLOT = None → use full X_COLUMNS_TO_PLOT).
# Cell order: this CONFIG → imports/paths → data & helpers → plots.
# =============================================================================

from pathlib import Path

# Paths (optional overrides)
USER_DATA_DIR = None  # e.g. Path("/home/.../results/relerr_cpp")
USER_PQ_FAISS_ADC_SUMMARY = None  # e.g. Path(".../pq_faiss_adc_summary.csv")

# Who to plot (all datasets in one go)
USER_METHODS_TO_PLOT = ["PQ"]
USER_DATASETS_TO_PLOT = ["deep", "bigann", "gist", "msmarco", "openai"]

# Metrics: names must exist in Y_METRIC_MAP in the data cell (e.g. relerr, spearman, recall, recon_error)
USER_Y_METRICS = ["relerr"]

# Per-x-column y limits or None for auto
USER_YLIM_CONFIG = {}

# Subquantizer filters (same dict structure as the data cell defaults)
USER_NBITS_PLOT_SUBQUANTIZERS = None
USER_NUM_SUBQ_PLOT_SUBQUANTIZERS = None

# Grouping / ADC time unit
USER_BITS_PER_VECTOR_GROUPING = None
USER_COMPRESSION_RATE_GROUPING = None
USER_ADC_TIME_UNIT = None

# X columns for the main sweep (list of (column, label) tuples) — None keeps data-cell default
USER_X_COLUMNS_TO_PLOT = None

USER_ADDITIONAL_PLOTS = None
USER_BAR_PLOTS = None

"""

    def pack_xy(title: str, cfg_text: str, exec_block: str) -> dict:
        cells = [
            _cell_md(title),
            _cell_code(cfg_text),
            _cell_code(c1),
            _cell_code(c2),
            _cell_code(exec_block),
        ]
        out = _nb_shell(meta)
        out["cells"] = cells
        return out

    xy_dir = OUT_DIR / "xy"
    xy_dir.mkdir(parents=True, exist_ok=True)
    pairs = _discover_main_xy_pairs(
        c1, c2_raw, extra_metrics_for_all_x=EXTRA_XY_METRICS_ALL_X
    )
    c3_xy_body = c3_defs + XY_SWEEP_PIN + c3_main
    for ykey, xc, xl, ge in pairs:
        stem = f"{_xy_filename_stem_metric(ykey)}__{_safe_filename_part(xc)}"
        title = (
            f"# PQ — {_y_axis_title(ykey, ge)} vs {xl}\n\n"
            "Single **y** vs **x** pair from the main metrics loop (`plot_relerr_vs_x`). "
            "Legend grouping follows the source notebook for that x-axis "
            "(for example, curves grouped by $N_{subq}$ when x is bits per subquantizer).\n\n"
            "Same plot from the shell (optional `--run-recon-eval` for distortion): "
            f"`python3 scripts/figure_generators/pq_xy_figures.py --metric {ykey} --x {xc} "
            f"--data-dir <path-to-relerr_cpp> [--run-recon-eval] --no-show`\n"
        )
        _write(
            xy_dir / f"xy_{stem}.ipynb",
            pack_xy(title, _cfg_intro_for_xy(cfg_intro, ykey, xc), c3_xy_body),
        )

    print("Wrote", len(pairs), "notebooks under", xy_dir)


if __name__ == "__main__":
    main()
