#!/usr/bin/env bash
set -euo pipefail

# SAQ grid sized for an overnight-ish run on the 1M/base subsets available under
# /data/cpanourg/2-hdvc/data. Override knobs from the environment if needed:
#
#   MAX_DATASET_JOBS=1 SAQ_THREADS=24 BITS="1 2 4" bash saq/run_saq_grid_12h_nohup.sh
#
# The script launches one process per dataset, with each process sweeping bits
# sequentially. This avoids running multiple PCA/IVF builds for the same dataset.

ROOT="/home/cpanourg/projects/2-hdvc"
DATA_ROOT="/data/cpanourg/2-hdvc/data"
LOG_DIR="${ROOT}/logs/saq_grid_$(date +%Y%m%d_%H%M%S)"

MAX_DATASET_JOBS="${MAX_DATASET_JOBS:-2}"
SAQ_THREADS="${SAQ_THREADS:-16}"
CLUSTERS="${CLUSTERS:-4096}"
BITS="${BITS:-1 2 4 8}"
MAX_BASE="${MAX_BASE:-1000000}"
MAX_QUERY="${MAX_QUERY:-200}"
PCA_TRAIN_SIZE="${PCA_TRAIN_SIZE:-100000}"
HDVC_RESULTS_ROOT="${HDVC_RESULTS_ROOT:-/data/cpanourg/2-hdvc/results}"
PYTHON_BIN="${PYTHON_BIN:-/home/cpanourg/.miniconda3/envs/dtwrl_env2/bin/python}"

mkdir -p "${LOG_DIR}" "${HDVC_RESULTS_ROOT}/saq"

run_one() {
  local dataset="$1"
  local vec_type="$2"
  local base="$3"
  local query="$4"

  local log="${LOG_DIR}/${dataset}.log"
  {
    echo "[$(date --iso-8601=seconds)] START ${dataset}"
    echo "base=${base}"
    echo "query=${query}"
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
      --force_prepare
    local rc=$?
    set -e
    echo "[$(date --iso-8601=seconds)] EXIT ${dataset} rc=${rc}"
    if (( rc != 0 )); then
      return "${rc}"
    fi

    echo "[$(date --iso-8601=seconds)] DONE ${dataset}"
  } >"${log}" 2>&1
}

wait_for_slot() {
  local running_jobs=()
  while true; do
    mapfile -t running_jobs < <(jobs -pr || true)
    if (( "${#running_jobs[@]}" < MAX_DATASET_JOBS )); then
      break
    fi
    sleep 30
  done
  return 0
}

cd "${ROOT}"

echo "Logs: ${LOG_DIR}"
echo "Results: ${HDVC_RESULTS_ROOT}/saq"
echo "Dataset parallelism: ${MAX_DATASET_JOBS}"

if (( MAX_DATASET_JOBS <= 1 )); then
  run_one "deep" "fvecs" \
    "${DATA_ROOT}/deep1b/dataset/fvecs/test_1m.fvecs" \
    "${DATA_ROOT}/deep1b/dataset/fvecs/query_10k.fvecs"

  run_one "bigann" "bvecs" \
    "${DATA_ROOT}/bigann/SIFT1M/bigann_base.bvecs" \
    "${DATA_ROOT}/bigann/SIFT1M/bigann_query.bvecs"

  run_one "msmarco" "fvecs" \
    "${DATA_ROOT}/msmarco/base1m.fvecs" \
    "${DATA_ROOT}/msmarco/query10k.fvecs"

  run_one "openai" "fvecs" \
    "${DATA_ROOT}/openai/openai_base1m.fvecs" \
    "${DATA_ROOT}/openai/openai_query10k.fvecs"

  run_one "gist" "fvecs" \
    "${DATA_ROOT}/gist/gist_base.fvecs" \
    "${DATA_ROOT}/gist/gist_query.fvecs"

  echo "[$(date --iso-8601=seconds)] all SAQ dataset jobs finished"
  exit 0
fi

wait_for_slot
run_one "deep" "fvecs" \
  "${DATA_ROOT}/deep1b/dataset/fvecs/test_1m.fvecs" \
  "${DATA_ROOT}/deep1b/dataset/fvecs/query_10k.fvecs" &

wait_for_slot
run_one "bigann" "bvecs" \
  "${DATA_ROOT}/bigann/SIFT1M/bigann_base.bvecs" \
  "${DATA_ROOT}/bigann/SIFT1M/bigann_query.bvecs" &

wait_for_slot
run_one "msmarco" "fvecs" \
  "${DATA_ROOT}/msmarco/base1m.fvecs" \
  "${DATA_ROOT}/msmarco/query10k.fvecs" &

wait_for_slot
run_one "openai" "fvecs" \
  "${DATA_ROOT}/openai/openai_base1m.fvecs" \
  "${DATA_ROOT}/openai/openai_query10k.fvecs" &

wait_for_slot
run_one "gist" "fvecs" \
  "${DATA_ROOT}/gist/gist_base.fvecs" \
  "${DATA_ROOT}/gist/gist_query.fvecs" &

wait

echo "[$(date --iso-8601=seconds)] all SAQ dataset jobs finished"
