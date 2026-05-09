#!/usr/bin/env bash
# Copy curated figure PDFs into idris_sub/figures/ from three paper "res" trees.
# Set absolute paths to the directory that *contains* res/ for each project, e.g.:
#   export HDVC_RES_ROOT="/home/you/projects/2-hdvc/HDVC_bench_paper"
#   export RAMSAD_RES_ROOT="/home/you/projects/3-tsfm/RAMSAD_paper"
#   export WETW_RES_ROOT="/home/you/projects/WETW/wetw_paper"
# If your layout has res/ elsewhere, point these at the parent of that res/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/figures"
ERR=0

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Error: environment variable ${name} is not set (absolute path to paper root containing res/)." >&2
    ERR=1
  fi
}

require_var HDVC_RES_ROOT
require_var RAMSAD_RES_ROOT
require_var WETW_RES_ROOT
if [[ "${ERR}" -ne 0 ]]; then
  echo "Set the three variables, then re-run." >&2
  exit 1
fi

for v in HDVC_RES_ROOT RAMSAD_RES_ROOT WETW_RES_ROOT; do
  r="${!v}/res"
  if [[ ! -d "$r" ]]; then
    echo "Error: expected directory not found: $r" >&2
    exit 1
  fi
done

mkdir -p "${DEST}"

cp -v "${HDVC_RES_ROOT}/res/figures/pareto_bpv/paper_relerr_vs_adc_by_bpv_deep.pdf" \
  "${DEST}/hdvc_pareto_deep.pdf"
cp -v "${HDVC_RES_ROOT}/res/figures/adc_vs_bits_per_vec/paper_adc_vs_bits_per_vector_deep.pdf" \
  "${DEST}/hdvc_adc_vs_bpv_deep.pdf"

cp -v "${RAMSAD_RES_ROOT}/res/figs/RAMSAD_overview23.pdf" \
  "${DEST}/ramsad_overview.pdf"
cp -v "${RAMSAD_RES_ROOT}/res/figs/RAMSAD_retrieval1.pdf" \
  "${DEST}/ramsad_retrieval.pdf"

cp -v "${WETW_RES_ROOT}/res/figures/wesee_architecture1.pdf" \
  "${DEST}/wetw_wesee_architecture.pdf"
cp -v "${WETW_RES_ROOT}/res/figures/wetw_workflow.pdf" \
  "${DEST}/wetw_workflow.pdf"

echo "Done. Figures in ${DEST}. Run ${ROOT}/build.sh"
