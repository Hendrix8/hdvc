#!/bin/bash
# Start the full README workflow on $mnthdd results (AQ train → regen ADC bash → ADC → CSV ×2 grids).
# Tunables: PARALLEL_WORKERS (default 16), PYTHON, QINCO2_PIPELINE_RESULTS_DIR

set -euo pipefail

RESULT_DIR="${QINCO2_PIPELINE_RESULTS_DIR:-/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/results}"
MASTER="${RESULT_DIR}/run_pipeline_master.sh"
PYTHON="${PYTHON:-${HOME}/.miniconda3/envs/tsfm/bin/python}"
export PARALLEL_WORKERS="${PARALLEL_WORKERS:-16}"
export PATH="$(dirname "$PYTHON"):$PATH"

if [[ ! -f "$MASTER" ]]; then
  echo "Missing $MASTER" >&2
  exit 2
fi

{
  echo ""
  echo "======== FULL PIPELINE START $(date -Is) PARALLEL_WORKERS=${PARALLEL_WORKERS} ========"
} >>"${RESULT_DIR}/pipeline_master.log"

(
  echo $$
  exec env PARALLEL_WORKERS="${PARALLEL_WORKERS}" PYTHON="${PYTHON}" PATH="${PATH}" \
    bash "${MASTER}" >>"${RESULT_DIR}/pipeline_master.log" 2>&1
) &
echo $! >"${RESULT_DIR}/pipeline_master.pid"

echo "Started PID $(cat "${RESULT_DIR}/pipeline_master.pid"); log: ${RESULT_DIR}/pipeline_master.log"
echo "Monitor: tail -f ${RESULT_DIR}/pipeline_master.log"
