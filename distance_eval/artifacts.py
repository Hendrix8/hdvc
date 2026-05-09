"""Resolve experiment folders to on-disk model artifacts."""

from __future__ import annotations

from pathlib import Path


def resolve_pq_index_dir(
    experiment_folder: Path,
    n_subquantizers: int | None = None,
    nbits: int | None = None,
    train_size: int | None = None,
) -> Path:
    """Folder containing ``pq_model.index`` (same logic as scripts/evals/run_evals)."""
    experiment_folder = Path(experiment_folder)
    model_path = experiment_folder / "pq_model.index"
    if model_path.exists():
        return experiment_folder

    has_fallback = (
        n_subquantizers is not None
        and n_subquantizers > 0
        and nbits is not None
        and nbits > 0
        and train_size is not None
        and train_size > 0
    )
    if not has_fallback:
        raise FileNotFoundError(f"Model not found: {model_path} (no fallback params)")

    parent = experiment_folder.parent
    pattern = f"subq{n_subquantizers}_nbits{nbits}_train{train_size}_*"
    for folder in sorted(parent.glob(pattern)):
        if (folder / "pq_model.index").exists():
            return folder

    raise FileNotFoundError(f"Model not found: {model_path} (no match for {pattern} in {parent})")


def resolve_opq_index_path(experiment_folder: Path) -> Path:
    p = Path(experiment_folder) / "opq_model.index"
    if not p.is_file():
        raise FileNotFoundError(f"OPQ model not found: {p}")
    return p


def resolve_lsq_index_path(experiment_folder: Path) -> Path:
    p = Path(experiment_folder) / "lsq_model.index"
    if not p.is_file():
        raise FileNotFoundError(f"LSQ++ model not found: {p}")
    return p
