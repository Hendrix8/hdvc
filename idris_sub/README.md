# `idris_sub` — project description pack (eDARI / administrative PDF)

This folder supports building a **standalone English PDF** describing compute needs for research on vector-compression benchmarking (HDVC line), RAMSAD (time-series anomaly-detector selection via retrieval), and WETW/WESee (DTW-consistent embeddings).

## eDARI constraints (typical)

- **Format:** PDF only, **&lt; 20 MB**
- **Description length:** often **≤ 15 pages** — this template targets ~6–10 pages once figures are included; shorten if needed.

## Figures

Paper repositories reference assets under `res/` next to their `.tex` sources; those PDFs are **not** vendored in git here.

1. Copy PDFs into [`figures/`](figures/) following [`figures/MANIFEST.md`](figures/MANIFEST.md), **or**
2. Run [`scripts/sync_figures_from_papers.sh`](scripts/sync_figures_from_papers.sh) after setting `HDVC_RES_ROOT`, `RAMSAD_RES_ROOT`, and `WETW_RES_ROOT` to directories that contain the respective `res/` folder.

The LaTeX source builds even when figures are missing (grey placeholders appear).

## Build

Requires `pdflatex` (TeX Live).

```bash
cd /path/to/2-hdvc/idris_sub
chmod +x build.sh scripts/sync_figures_from_papers.sh   # once
./build.sh
```

Output: **`project_description.pdf`** in this directory.

## Edit before submission

The narrative is complete (three workstreams aligned with the HDVC, RAMSAD, and WETW Overleaf manuscripts). You only need to:

1. Set `\AllocHours` and `\AllocPartition` in the preamble of [`project_description/main.tex`](project_description/main.tex) to the exact values from your DARI/Genci **Ressources** tab (or replace the grey placeholder text there).

2. Adjust affiliation/email if your institution requires official wording.

## Halting errors

If LaTeX fails, run manually for logs:

```bash
cd project_description
pdflatex -output-directory=build main.tex
```
