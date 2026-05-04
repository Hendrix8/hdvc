#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cpanourg/projects/2-hdvc"
DATA_ROOT="/data/cpanourg/2-hdvc/data"
LOG_DIR="${ROOT}/logs/saq_failed_rerun_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${PYTHON_BIN:-/home/cpanourg/.miniconda3/envs/dtwrl_env2/bin/python}"
HDVC_RESULTS_ROOT="${HDVC_RESULTS_ROOT:-/data/cpanourg/2-hdvc/results}"

BITS="${BITS:-1 2 4 8}"
CLUSTERS="${CLUSTERS:-4096}"
SAQ_THREADS="${SAQ_THREADS:-16}"
MAX_BASE="${MAX_BASE:-1000000}"
MAX_QUERY="${MAX_QUERY:-200}"
PCA_TRAIN_SIZE="${PCA_TRAIN_SIZE:-100000}"

mkdir -p "${LOG_DIR}" "${HDVC_RESULTS_ROOT}/saq"
cd "${ROOT}"

run_one() {
  local dataset="$1"
  local vec_type="$2"
  local base="$3"
  local query="$4"
  shift 4
  local extra_args=("$@")
  local log="${LOG_DIR}/${dataset}.log"

  {
    echo "[$(date --iso-8601=seconds)] START ${dataset}"
    echo "bits=${BITS}; clusters=${CLUSTERS}; max_base=${MAX_BASE}; max_query=${MAX_QUERY}; pca_train_size=${PCA_TRAIN_SIZE}"
    set +e
    HDVC_RESULTS_ROOT="${HDVC_RESULTS_ROOT}" "${PYTHON_BIN}" "${ROOT}/saq/saq_relerr.py" \
      --dataset "${dataset}" \
      --base "${base}" \
      --query "${query}" \
      --vec_type "${vec_type}" \
      --clusters "${CLUSTERS}" \
      --bits ${BITS} \
      --num_threads "${SAQ_THREADS}" \
      --max_base "${MAX_BASE}" \
      --max_query "${MAX_QUERY}" \
      --pca_train_size "${PCA_TRAIN_SIZE}" \
      "${extra_args[@]}"
    local rc=$?
    set -e
    echo "[$(date --iso-8601=seconds)] EXIT ${dataset} rc=${rc}"
    if (( rc != 0 )); then
      return "${rc}"
    fi
    echo "[$(date --iso-8601=seconds)] DONE ${dataset}"
  } >"${log}" 2>&1
}

echo "Logs: ${LOG_DIR}"
echo "Results: ${HDVC_RESULTS_ROOT}/saq"

# OpenAI already has prepared PCA/IVF artifacts; do not force data prep.
run_one "openai" "fvecs" \
  "${DATA_ROOT}/openai/openai_base1m.fvecs" \
  "${DATA_ROOT}/openai/openai_query10k.fvecs"

# BigANN failed during preparation when competing with other jobs; rerun alone.
run_one "bigann" "bvecs" \
  "${DATA_ROOT}/bigann/SIFT1M/bigann_base.bvecs" \
  "${DATA_ROOT}/bigann/SIFT1M/bigann_query.bvecs" \
  --force_prepare

echo "[$(date --iso-8601=seconds)] failed SAQ rerun finished"
