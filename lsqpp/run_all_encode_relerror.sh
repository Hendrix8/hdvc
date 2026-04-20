#!/usr/bin/env bash
# LSQ++ on Deep + BigANN (SIFT1M) + GIST + MSMARCO + OpenAI (encode + ADC rel. error; no index search).
# Usage: bash lsqpp/run_all_encode_relerror.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="/data/cpanourg/2-hdvc"
DATA_PREFIX="${DATA_ROOT}/data"
RESULTS_SUBDIR="results/lsqpp"
TRAIN_SIZE=100000
SAMPLE_DB=10000
SAMPLE_QUERIES=1000
N_SUBQUANTIZERS=32
NBITS=8
PYTHON="python3"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONPATH="${REPO_ROOT}"

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
      echo "Unknown dataset: ${DS}" >&2
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
  echo "# LSQ++ | ${DS}"
  echo "############################################"

  "${PYTHON}" -m lsqpp.eval \
    --dataset_path "${DATASET_PATH}" \
    --train_path "${TRAIN_PATH}" \
    --query_path "${QUERY_PATH}" \
    "${DIM_OPT[@]}" \
    --dataset_name "${DS}" \
    --data_root "${DATA_ROOT}" \
    --n_subquantizers "${N_SUBQUANTIZERS}" \
    --nbits "${NBITS}" \
    --train_size "${TRAIN_SIZE}" \
    --sample_db "${SAMPLE_DB}" \
    --sample_queries "${SAMPLE_QUERIES}" \
    --results_dir "${RESULTS_SUBDIR}" \
    --sample_mode first

  echo "Aggregate CSV: ${DATA_ROOT}/${RESULTS_SUBDIR}/${DS}_LSQpp_adc_vs_exact_eval.csv"
}

echo "repo=${REPO_ROOT}  data_root=${DATA_ROOT}"
echo ""

for DS in deep bigann gist msmarco openai; do
  run_one "${DS}"
done

echo "Done."
