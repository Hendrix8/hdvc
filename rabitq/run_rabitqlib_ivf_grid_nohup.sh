#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/cpanourg/.conda/envs/dtwrl_env2/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing python interpreter: ${PYTHON}" >&2
  exit 1
fi

RESULTS_ROOT="${HDVC_RESULTS_ROOT:-${REPO_ROOT}/results}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/rabitq_native_ivf_nohup_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONPATH="${REPO_ROOT}"

CPUSET="${CPUSET:-0}"
AFFINITY=()
if command -v taskset >/dev/null 2>&1; then
  AFFINITY=(taskset -c "${CPUSET}")
fi

LOG="${RUN_ROOT}/nohup.out"
PID_FILE="${RUN_ROOT}/nohup.pid"
ENV_LOG="${RUN_ROOT}/env.txt"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "python=${PYTHON}"
  echo "repo_root=${REPO_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "cpuset=${CPUSET}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
  echo "NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS}"
  echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
  echo
  "${PYTHON}" scripts/evals/check_benchmark_env.py || true
} > "${ENV_LOG}"

nohup "${AFFINITY[@]}" "${PYTHON}" -m rabitq.rabitqlib_ivf_grid --output_root "${RUN_ROOT}" "$@" > "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Launched RaBitQ IVF grid"
echo "PID: $(cat "${PID_FILE}")"
echo "Run root: ${RUN_ROOT}"
echo "Log: ${LOG}"
