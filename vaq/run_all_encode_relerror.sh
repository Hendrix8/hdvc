#!/usr/bin/env bash
# VAQ on Deep + BigANN (SIFT1M) + GIST + MSMARCO + OpenAI:
# train + codes + Python ADC vs exact rel. error + CSVs; no C++ ANN (--skip_search).
# All paths and knobs are literals below — no env vars required.
# Usage: bash vaq/run_all_encode_relerror.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="/data/cpanourg/2-hdvc"
DATA_PREFIX="${DATA_ROOT}/data"
RESULTS_SUBDIR="results/vaq"
TRAIN_SIZE=100000
SAMPLE_DB=10000
SAMPLE_QUERIES=1000
REFINE="100,200"
K=100
LEARN_RATIO=0.05
METHOD_STR="VAQ256m32min7max8var1,HEAP"
VAQ_BINARY="${REPO_ROOT}/lib/VAQ/build/examples/run_vaq"
PYTHON="python3"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONPATH="${REPO_ROOT}"

if [[ ! -f "${VAQ_BINARY}" ]]; then
  echo "ERROR: VAQ binary not found: ${VAQ_BINARY}" >&2
  exit 1
fi

run_one() {
  local DS="$1"
  local DATASET_PATH TRAIN_PATH QUERY_PATH
  local -a DIM_OPT=()

  case "${DS}" in
    deep)
      DATASET_PATH="${DATA_PREFIX}/deep1b/dataset/test_1m.bin"
      TRAIN_PATH="${DATA_PREFIX}/deep1b/dataset/learn_100m.bin"
      QUERY_PATH="${DATA_PREFIX}/deep1b/dataset/query_10k.bin"
      DIM_OPT=(--dim 96)
      ;;
    bigann)
      # SIFT1M
      DATASET_PATH="${DATA_PREFIX}/bigann/SIFT1M/bigann_base.bvecs"
      QUERY_PATH="${DATA_PREFIX}/bigann/SIFT1M/bigann_query.bvecs"
      TRAIN_PATH="${DATA_PREFIX}/bigann/SIFT1M/bigann_learn.bvecs"
      ;;
    gist)
      DATASET_PATH="${DATA_PREFIX}/gist/gist_base.fvecs"
      QUERY_PATH="${DATA_PREFIX}/gist/gist_query.fvecs"
      TRAIN_PATH="${DATA_PREFIX}/gist/gist_learn.fvecs"
      ;;
    msmarco)
      DATASET_PATH="${DATA_PREFIX}/msmarco/base1m.fvecs"
      QUERY_PATH="${DATA_PREFIX}/msmarco/query10k.fvecs"
      TRAIN_PATH="${DATA_PREFIX}/msmarco/train1m.fvecs"
      ;;
    openai)
      DATASET_PATH="${DATA_PREFIX}/openai/openai_base1m.fvecs"
      QUERY_PATH="${DATA_PREFIX}/openai/openai_query10k.fvecs"
      TRAIN_PATH="${DATA_PREFIX}/openai/openai_train1m.fvecs"
      ;;
    *)
      echo "Unknown dataset key: ${DS}" >&2
      return 1
      ;;
  esac

  for f in "${DATASET_PATH}" "${TRAIN_PATH}" "${QUERY_PATH}"; do
    if [[ ! -f "${f}" ]]; then
      echo "SKIP ${DS}: missing file: ${f}" >&2
      return 0
    fi
  done

  echo ""
  echo "############################################"
  echo "# VAQ | ${DS} | encode + rel. error (no C++ ANN)"
  echo "############################################"

  "${PYTHON}" -m vaq.eval \
    --dataset_path "${DATASET_PATH}" \
    --train_path "${TRAIN_PATH}" \
    --query_path "${QUERY_PATH}" \
    "${DIM_OPT[@]}" \
    --dataset_name "${DS}" \
    --data_root "${DATA_ROOT}" \
    --method "${METHOD_STR}" \
    --train_size "${TRAIN_SIZE}" \
    --sample_db "${SAMPLE_DB}" \
    --sample_queries "${SAMPLE_QUERIES}" \
    --results_dir "${RESULTS_SUBDIR}" \
    --vaq_binary "${VAQ_BINARY}" \
    --refine "${REFINE}" \
    --k "${K}" \
    --learn_ratio "${LEARN_RATIO}" \
    --sample_mode first \
    --skip_search

  echo "Aggregate CSV: ${DATA_ROOT}/${RESULTS_SUBDIR}/${DS}_VAQ_adc_vs_exact_eval.csv"
}

echo "repo      = ${REPO_ROOT}"
echo "data_root = ${DATA_ROOT}"
echo "method    = ${METHOD_STR}"
echo "order     = deep bigann gist msmarco openai"
echo ""

for DS in deep bigann gist msmarco openai; do
  run_one "${DS}"
done

echo ""
echo "Done."
