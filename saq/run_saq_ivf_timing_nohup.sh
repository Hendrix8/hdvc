#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/cpanourg/.conda/envs/dtwrl_env2/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing python interpreter: ${PYTHON}" >&2
  exit 1
fi

# Keep timing reruns separate from the main SAQ summaries by default.
RESULTS_ROOT="${HDVC_TIMING_RESULTS_ROOT:-${REPO_ROOT}/results_timing}"
RUN_ROOT="${RUN_ROOT:-${RESULTS_ROOT}/saq_native_ivf_timing_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONPATH="${REPO_ROOT}"
export HDVC_RESULTS_ROOT="${RESULTS_ROOT}"

DATASETS=("${DATASETS[@]:-deep10k bigann10k gist10k msmarco10k openai10k}")
CLUSTERS=("${CLUSTERS[@]:-64 128 256}")
BITS=("${BITS[@]:-0.25 1 2 3 4 6 8}")
CAQ_ADJ_RD_LMT=("${CAQ_ADJ_RD_LMT[@]:-6}")
SEARCHER_VARS_BOUND_M=("${SEARCHER_VARS_BOUND_M[@]:-4}")

N_RUNS="${N_RUNS:-10}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
NUM_THREADS="${NUM_THREADS:-24}"
SKIP_BUILD="${SKIP_BUILD:-1}"
RAND_ROTATE="${RAND_ROTATE:-true}"
USE_FASTSCAN="${USE_FASTSCAN:-true}"
PCA_TRAIN_SIZE="${PCA_TRAIN_SIZE:-}"
CPUSET="${CPUSET:-}"
APPEND_LOG="${APPEND_LOG:-0}"

AFFINITY=()
if [[ -n "${CPUSET}" ]] && command -v taskset >/dev/null 2>&1; then
  AFFINITY=(taskset -c "${CPUSET}")
fi

LOG="${RUN_ROOT}/nohup.out"
PID_FILE="${RUN_ROOT}/nohup.pid"
ENV_LOG="${RUN_ROOT}/env.txt"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "python=${PYTHON}"
  echo "repo_root=${REPO_ROOT}"
  echo "results_root=${RESULTS_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "cpuset=${CPUSET:-<none>}"
  echo "datasets=${DATASETS[*]}"
  echo "clusters=${CLUSTERS[*]}"
  echo "bits=${BITS[*]}"
  echo "caq_adj_rd_lmt=${CAQ_ADJ_RD_LMT[*]}"
  echo "searcher_vars_bound_m=${SEARCHER_VARS_BOUND_M[*]}"
  echo "n_runs=${N_RUNS}"
  echo "warmup_runs=${WARMUP_RUNS}"
  echo "num_threads=${NUM_THREADS}"
  echo "rand_rotate=${RAND_ROTATE}"
  echo "use_fastscan=${USE_FASTSCAN}"
  echo "skip_build=${SKIP_BUILD}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
  echo "NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS}"
  echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
  echo
  "${PYTHON}" scripts/evals/check_benchmark_env.py || true
} > "${ENV_LOG}"

CMD=(
  "${PYTHON}" -u -m saq.saq_ivf_grid
  --output_root "${RUN_ROOT}"
  --datasets "${DATASETS[@]}"
  --clusters "${CLUSTERS[@]}"
  --bits "${BITS[@]}"
  --caq_adj_rd_lmt "${CAQ_ADJ_RD_LMT[@]}"
  --searcher_vars_bound_m "${SEARCHER_VARS_BOUND_M[@]}"
  --n_runs "${N_RUNS}"
  --warmup_runs "${WARMUP_RUNS}"
  --num_threads "${NUM_THREADS}"
  --rand_rotate "${RAND_ROTATE}"
  --use_fastscan "${USE_FASTSCAN}"
)

if [[ "${SKIP_BUILD}" == "1" ]]; then
  CMD+=(--skip_build)
fi
if [[ -n "${PCA_TRAIN_SIZE}" ]]; then
  CMD+=(--pca_train_size "${PCA_TRAIN_SIZE}")
fi

if [[ "${APPEND_LOG}" == "1" ]]; then
  nohup "${AFFINITY[@]}" "${CMD[@]}" >> "${LOG}" 2>&1 &
else
  nohup "${AFFINITY[@]}" "${CMD[@]}" > "${LOG}" 2>&1 &
fi
echo $! > "${PID_FILE}"

echo "Launched SAQ timing rerun"
echo "PID: $(cat "${PID_FILE}")"
echo "Run root: ${RUN_ROOT}"
echo "Results root: ${RESULTS_ROOT}"
echo "Log: ${LOG}"
echo
echo "Track progress with:"
echo "  tail -f ${RUN_ROOT}/progress.jsonl"
echo "  tail -f ${LOG}"
