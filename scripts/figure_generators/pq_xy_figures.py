#!/usr/bin/env python3
"""
Run a single PQ “main loop” figure (same logic as scripts/fig_notebooks/pq/xy/*.ipynb)
from the command line by executing relerr_cpp_plots.ipynb cells 1–3 with pinned
TARGET_Y_METRIC / TARGET_X_COL.

This avoids notebook JSON / Run-All friction and can optionally fill
``reconstruction_error`` via ``scripts/evals/run_evals.py`` before plotting.

Examples
--------
  # Distortion vs Bps — compute reconstruction first, then plot (non-interactive)
  python3 scripts/figure_generators/pq_xy_figures.py \\
    --metric recon_error --x nbits \\
    --data-dir results/relerr_cpp \\
    --data-root /data/cpanourg/2-hdvc/data \\
    --run-recon-eval --no-show

  # Avg relative error vs M (uses existing CSVs only)
  python3 scripts/figure_generators/pq_xy_figures.py \\
    --metric relerr --x n_subquantizers \\
    --data-dir results/relerr_cpp --no-show
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_generate_split_module():
    """Load fig_notebooks/pq/generate_split_notebooks.py (XY_SWEEP_PIN, split helpers)."""
    here = Path(__file__).resolve()
    gen_path = here.parent.parent / "fig_notebooks" / "pq" / "generate_split_notebooks.py"
    spec = importlib.util.spec_from_file_location("_pq_xy_gen", gen_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {gen_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _join_cell(nb: dict, index: int) -> str:
    return "".join(nb["cells"][index].get("source", []))


def _run_reconstruction_evals(
    data_dir: Path,
    data_root: Path,
    *,
    max_rec_samples: int,
    pattern: str = "*_PQ_adc_vs_exact_eval.csv",
) -> None:
    repo = Path(__file__).resolve().parents[2]
    scripts_dir = repo / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from evals.run_evals import run_evals

    inputs = sorted(data_dir.glob(pattern))
    if not inputs:
        raise FileNotFoundError(f"No inputs matching {pattern!r} under {data_dir}")
    for inp in inputs:
        print(f"[pq_xy_figures] run_evals reconstruction_error only: {inp}")
        run_evals(
            inp,
            data_dir,
            data_root=data_root,
            eval_measures=["reconstruction_error"],
            max_rec_samples=max_rec_samples,
        )


def _build_exec_script(
    *,
    data_dir: Path,
    pq_summary: Path | None,
    target_metric: str,
    target_x: str,
    gen,
) -> str:
    nb_path = gen.REPO / "scripts" / "figure_generators" / "relerr_cpp_plots.ipynb"
    nb = json.loads(nb_path.read_text())
    c1 = _join_cell(nb, 1)
    c2_raw = _join_cell(nb, 2)
    c3 = _join_cell(nb, 3)
    c3_defs, c3_main, _, _ = gen._split_c3_exec(c3)
    c2 = gen._patch_cell2_with_apply(c2_raw)

    override = f"""

# ----- pq_xy_figures.py overrides (after notebook cell 1 defaults) -----
DATA_DIR = Path(r"{data_dir.resolve()}")
"""
    if pq_summary is not None:
        pq = pq_summary.resolve()
        override += f"""PQ_FAISS_ADC_SUMMARY = Path(r"{pq}")
PQ_FAISS_FIGURES_DIR = PQ_FAISS_ADC_SUMMARY.parent / "figures"
"""
    pin = f"""
TARGET_Y_METRIC = {target_metric!r}
TARGET_X_COL = {target_x!r}
"""
    return (
        c1
        + override
        + "\n"
        + c2
        + "\n"
        + c3_defs
        + pin
        + gen.XY_SWEEP_PIN
        + c3_main
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PQ metric vs x — execute relerr_cpp_plots notebook logic from CLI")
    p.add_argument(
        "--metric",
        default="relerr",
        help="Y_METRIC_MAP key (relerr, recon_error, spearman, recall, …)",
    )
    p.add_argument(
        "--x",
        dest="x_col",
        required=True,
        help="X column name, e.g. nbits, n_subquantizers, bits_per_vector, adc_cpu_time_pp",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with *_adc_vs_exact_eval.csv and merged metric CSVs",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("HDVC_DATA_ROOT", "/data/cpanourg/2-hdvc/data")),
        help="Dataset root passed to run_evals (for --run-recon-eval)",
    )
    p.add_argument(
        "--pq-summary",
        type=Path,
        default=None,
        help="Optional pq_faiss_adc_summary.csv; if set, overrides PQ_FAISS_ADC_SUMMARY after cell 1",
    )
    p.add_argument(
        "--run-recon-eval",
        action="store_true",
        help="Run run_evals(..., eval_measures=['reconstruction_error']) on each *_PQ_adc_vs_exact_eval.csv first",
    )
    p.add_argument(
        "--max-rec-samples",
        type=int,
        default=10_000,
        help="Max DB vectors for reconstruction_error when using --run-recon-eval",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Use matplotlib Agg backend (save PDFs, no GUI windows)",
    )
    args = p.parse_args(argv)

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"ERROR: --data-dir is not a directory: {data_dir}", file=sys.stderr)
        return 1

    if args.run_recon_eval:
        _run_reconstruction_evals(
            data_dir,
            args.data_root,
            max_rec_samples=args.max_rec_samples,
        )

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    gen = _load_generate_split_module()
    script = _build_exec_script(
        data_dir=data_dir,
        pq_summary=args.pq_summary,
        target_metric=args.metric,
        target_x=args.x_col,
        gen=gen,
    )
    g: dict = {"__name__": "__main__", "__builtins__": __builtins__, "Path": Path}
    exec(compile(script, str(gen.REPO / "relerr_cpp_plots.ipynb_cells_1_3"), "exec"), g, g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
