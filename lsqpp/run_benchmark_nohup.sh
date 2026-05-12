#!/usr/bin/env bash
# Launch LSQ++ grid with conservative benchmark settings under nohup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -d /mnthdd/cpanourg/2-hdvc/data ]]; then
    DATA_ROOT="/mnthdd/cpanourg/2-hdvc"
  elif [[ -d /data/cpanourg/2-hdvc/data ]]; then
    DATA_ROOT="/data/cpanourg/2-hdvc"
  else
    echo "Could not infer DATA_ROOT; set DATA_ROOT explicitly." >&2
    exit 1
  fi
fi

PYTHON="${PYTHON:-/home/cpanourg/.conda/envs/dtwrl_env2/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter not found: ${PYTHON}" >&2
  exit 1
fi

DATASETS="${DATASETS:-deep bigann gist}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-benchmark_$(date +%Y%m%d_%H%M%S)}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/lsqpp/${EXPERIMENT_TAG}}"
LOG_DIR="${DATA_ROOT}/${RESULTS_SUBDIR}/launcher_logs"
mkdir -p "${LOG_DIR}"

export DATA_ROOT
export DATASETS
export RESULTS_SUBDIR
export PYTHON
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MAX_DATASET_JOBS="${MAX_DATASET_JOBS:-1}"
export PYTHONPATH="${REPO_ROOT}"

AFFINITY_CMD=()
if command -v taskset >/dev/null 2>&1; then
  CPUSET="${CPUSET:-0}"
  AFFINITY_CMD=(taskset -c "${CPUSET}")
fi

ENV_REPORT="${LOG_DIR}/benchmark_env.txt"
NOHUP_LOG="${LOG_DIR}/nohup.out"
PID_FILE="${LOG_DIR}/nohup.pid"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "repo_root=${REPO_ROOT}"
  echo "data_root=${DATA_ROOT}"
  echo "results_subdir=${RESULTS_SUBDIR}"
  echo "datasets=${DATASETS}"
  echo "python=${PYTHON}"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "mkl_num_threads=${MKL_NUM_THREADS}"
  echo "numba_num_threads=${NUMBA_NUM_THREADS}"
  echo "openblas_num_threads=${OPENBLAS_NUM_THREADS}"
  echo "max_dataset_jobs=${MAX_DATASET_JOBS}"
  echo "cpuset=${CPUSET:-<none>}"
  echo
  "${PYTHON}" scripts/evals/check_benchmark_env.py || true
} > "${ENV_REPORT}"

nohup "${AFFINITY_CMD[@]}" bash lsqpp/run_all_datasets_grid_nohup.sh > "${NOHUP_LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Launched LSQ++ benchmark run"
echo "PID: $(cat "${PID_FILE}")"
echo "Results: ${DATA_ROOT}/${RESULTS_SUBDIR}"
echo "Log: ${NOHUP_LOG}"
echo "Env report: ${ENV_REPORT}"
