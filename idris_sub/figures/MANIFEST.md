# Figure manifest for `idris_sub`

Curated PDFs for the project description. Copy from your paper build trees (each paper expects a `res/` directory next to its `.tex` sources). Relative paths below are **as referenced** in the respective `paper.tex` / `neurips_2026.tex`.

| Local file (this folder) | Source paper | Path under that paper’s `res/` |
|--------------------------|--------------|--------------------------------|
| `hdvc_pareto_deep.pdf` | [HDVC_bench_paper](../../HDVC_bench_paper/paper.tex) | `figures/pareto_bpv/paper_relerr_vs_adc_by_bpv_deep.pdf` |
| `hdvc_adc_vs_bpv_deep.pdf` | HDVC_bench_paper | `figures/adc_vs_bits_per_vec/paper_adc_vs_bits_per_vector_deep.pdf` |
| `ramsad_overview.pdf` | [RAMSAD_paper](../../../3-tsfm/RAMSAD_paper/neurips_2026.tex) | `figs/RAMSAD_overview23.pdf` |
| `ramsad_retrieval.pdf` | RAMSAD_paper | `figs/RAMSAD_retrieval1.pdf` |
| `wetw_wesee_architecture.pdf` | [wetw_paper](../../../WETW/wetw_paper/paper.tex) | `figures/wesee_architecture1.pdf` |
| `wetw_workflow.pdf` | wetw_paper | `figures/wetw_workflow.pdf` |

**Optional extras** (not wired into `main.tex` by default): HDVC `figures/method_comparison_*` from the benchmark paper; WETW `figures/classif_speedup.pdf`, `figures/clust_speedup.pdf`; RAMSAD `figs/ID_vs_OOD.pdf`, `figs/Boxplots_and_CD.pdf`.

After copying, run [`../build.sh`](../build.sh) from `idris_sub`, or use [`scripts/sync_figures_from_papers.sh`](../scripts/sync_figures_from_papers.sh).
