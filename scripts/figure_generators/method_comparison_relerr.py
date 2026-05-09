"""
Load relerr-style CSVs from multiple result roots and compare methods at a target
bits_per_vector (closest row per dataset + method).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

for _anc in Path(__file__).resolve().parents:
    if (_anc / "hdvc_paths.py").is_file():
        if str(_anc) not in sys.path:
            sys.path.insert(0, str(_anc))
        break
from hdvc_paths import get_results_root  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from relerr_quick_plot_style import COLOR_PALETTE, apply_relerr_cpp_rcparams


@dataclass(frozen=True)
class MethodSource:
    """One method family loaded from CSV rows under ``results_dir`` / ``glob_pattern``."""

    label: str
    results_dir: Path
    glob_pattern: str
    #: If set, keep only rows whose CSV ``method`` column equals this, then rename to ``label``.
    method_in_csv: str | None = None
    #: If set and the CSV has ``train_size``, keep only matching rows (e.g. align LSQ++ with PQ).
    train_size_filter: int | None = None


def load_concat_sources(sources: Iterable[MethodSource]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src in sources:
        paths = sorted(src.results_dir.glob(src.glob_pattern))
        for p in paths:
            try:
                df = pd.read_csv(p, on_bad_lines="skip")
            except TypeError:
                df = pd.read_csv(p, error_bad_lines=False, warn_bad_lines=False)
            if "method" not in df.columns or "dataset" not in df.columns:
                continue
            if "bits_per_vector" not in df.columns or "rel_error_mean" not in df.columns:
                continue
            if src.method_in_csv is not None:
                df = df[df["method"] == src.method_in_csv].copy()
                if df.empty:
                    continue
                df["method"] = src.label
            if src.train_size_filter is not None and "train_size" in df.columns:
                ts = pd.to_numeric(df["train_size"], errors="coerce")
                want = float(src.train_size_filter)
                df = df[ts.notna() & (ts == want)].copy()
                if df.empty:
                    continue
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pick_closest_bpv(
    df: pd.DataFrame,
    dataset: str,
    method: str,
    target_bpv: float,
    *,
    max_abs_bpv_diff: float | None = None,
) -> pd.Series | None:
    sub = df[(df["dataset"] == dataset) & (df["method"] == method)].copy()
    if sub.empty:
        return None
    sub["_bpv_dist"] = (sub["bits_per_vector"].astype(float) - float(target_bpv)).abs()
    sub = sub.sort_values(["_bpv_dist", "rel_error_mean"], ascending=[True, True])
    best = sub.iloc[0]
    if max_abs_bpv_diff is not None:
        if float(best["_bpv_dist"]) > float(max_abs_bpv_diff):
            return None
    return best


def build_comparison_table(
    df: pd.DataFrame,
    datasets: list[str],
    methods: list[str],
    target_bits_per_vector: float,
    *,
    max_abs_bpv_diff: float | None = None,
) -> pd.DataFrame:
    """
    Rows: datasets. Columns: method -> rel_error_mean, plus actual bits_per_vector used.
    If max_abs_bpv_diff is set, rows farther than that (in bits_per_vector) are treated as missing.
    """
    rows = []
    for ds in datasets:
        rec: dict = {"dataset": ds}
        for m in methods:
            row = pick_closest_bpv(
                df, ds, m, target_bits_per_vector, max_abs_bpv_diff=max_abs_bpv_diff
            )
            if row is None:
                rec[f"{m}_rel_error"] = np.nan
                rec[f"{m}_bits_per_vector"] = np.nan
            else:
                rec[f"{m}_rel_error"] = float(row["rel_error_mean"])
                rec[f"{m}_bits_per_vector"] = float(row["bits_per_vector"])
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_grouped_bar_relerr(
    comp: pd.DataFrame,
    methods: list[str],
    *,
    target_bits_per_vector: float,
    title_suffix: str = "",
    figsize: tuple[float, float] = (12, 6),
    output_path: Path | None = None,
    dpi: int = 300,
    close_fig: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grouped bars: x = dataset, one bar per method (rel_error_mean).
    """
    apply_relerr_cpp_rcparams()
    datasets = comp["dataset"].tolist()
    n_ds = len(datasets)
    n_m = len(methods)
    x = np.arange(n_ds, dtype=float)
    width = min(0.8 / max(n_m, 1), 0.15)

    fig, ax = plt.subplots(figsize=figsize)

    for i, m in enumerate(methods):
        col = f"{m}_rel_error"
        if col not in comp.columns:
            continue
        vals = comp[col].values
        offset = width * (i - (n_m - 1) / 2)
        ax.bar(
            x + offset,
            vals,
            width,
            label=m,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            edgecolor="black",
            linewidth=1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=28)
    ax.set_ylabel("Relative error", fontsize=40)
    ax.tick_params(axis="y", labelsize=40)
    ax.grid(alpha=0.8, axis="y", linestyle="--")
    for spine in ax.spines.values():
        spine.set_visible(False)

    t = f"Relative error (closest bits/vector to {int(target_bits_per_vector)})"
    if title_suffix:
        t = f"{t}\n{title_suffix}"
    ax.set_title(t, fontsize=28)
    ax.legend(frameon=False, fontsize=18, loc="upper right", ncol=min(3, n_m))

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    if close_fig:
        plt.close(fig)
    return fig, ax


def default_method_sources(
    data_root: Path | str | None = None,
    *,
    lsq_train_size_filter: int | None = 1_000_000,
) -> tuple[MethodSource, ...]:
    """
    Default result locations under ``data_root`` (same layout as eval export scripts).

    LSQ++ rows use ``method`` = ``LSQpp`` in CSV; they are normalized to label ``LSQ++``.
    ``lsq_train_size_filter`` picks one training budget when several share the same
    ``bits_per_vector`` (defaults to 1e6 to match typical PQ/OPQ relerr_cpp runs).
    """
    root = Path(data_root) if data_root is not None else get_results_root()
    return (
        MethodSource("TQMSE", root / "turboquant", "*_TQMSE_adc_vs_exact_eval.csv"),
        MethodSource("TQProd", root / "turboquant", "*_TQProd_adc_vs_exact_eval.csv"),
        MethodSource("RaBitQ", root / "rabitq", "*_RaBitQ_adc_vs_exact_eval.csv"),
        MethodSource("SAQ", root / "saq", "*_SAQ_adc_vs_exact_eval.csv"),
        MethodSource("VAQ", root / "vaq", "*_VAQ_adc_vs_exact_eval.csv"),
        MethodSource("PQ", root / "relerr_cpp", "*_PQ_adc_vs_exact_eval.csv"),
        MethodSource("OPQ", root / "relerr_cpp", "*_OPQ_adc_vs_exact_eval.csv"),
        MethodSource(
            "LSQ++",
            root / "lsqpp",
            "*_LSQpp_adc_vs_exact_eval.csv",
            method_in_csv="LSQpp",
            train_size_filter=lsq_train_size_filter,
        ),
    )


DEFAULT_SOURCES: tuple[MethodSource, ...] = default_method_sources()


def plot_unified_adc_pareto_if_available(
    *,
    timing_csv: Path | None = None,
    output_png: Path | None = None,
    dataset: str | None = None,
) -> Path | None:
    """
    If ``distance_eval/results/unified_adc_timing.csv`` exists (from
    ``python -m distance_eval.harness``), write a Pareto-style scatter
    (rel_error_mean vs adc_total_time_pp_mean) next to it.
    """
    root = Path(__file__).resolve().parents[2]
    timing = timing_csv or (root / "distance_eval" / "results" / "unified_adc_timing.csv")
    if not timing.is_file():
        return None
    out = output_png or (root / "distance_eval" / "results" / "unified_adc_pareto.png")
    # Import here so scripts work when distance_eval is optional
    from distance_eval.figures import plot_unified_timing

    plot_unified_timing(timing, out, dataset=dataset)
    return out
