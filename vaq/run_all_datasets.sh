#!/usr/bin/env bash
# VAQ eval (default method) on bigann, gist, msmarco, openai, optional deep.
# Requires: lib/VAQ/build/examples/run_vaq
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${DATA_ROOT:-/data/cpanourg/2-hdvc}"
DATA_ROOT="${DATA_ROOT%/}"
DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT}/data}"

DATASETS="${DATASETS:-bigann gist msmarco openai}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/vaq}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
METHOD="${METHOD:-VAQ256m32min7max8var1,HEAP}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"
VAQ_BINARY="${VAQ_BINARY:-${REPO_ROOT}/lib/VAQ/build/examples/run_vaq}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "${VAQ_BINARY}" ]]; then
  echo "ERROR: VAQ binary not found: ${VAQ_BINARY}" >&2
  echo "Build the VAQ project (CMake) so run_vaq exists." >&2
  exit 1
fi

dataset_paths() {
  local name="$1"
  DIM_OPT=()
  case "${name}" in
    bigann)
      DATASET_PATH="${BIGANN_DATASET_PATH:-${DATA_PREFIX}/bigann/SIFT1M/bigann_base.bvecs}"
      QUERY_PATH="${BIGANN_QUERY_PATH:-${DATA_PREFIX}/bigann/SIFT1M/bigann_query.bvecs}"
      TRAIN_PATH="${BIGANN_TRAIN_PATH:-${DATA_PREFIX}/bigann/SIFT1M/bigann_learn.bvecs}"
      ;;
    gist)
      DATASET_PATH="${GIST_DATASET_PATH:-${DATA_PREFIX}/gist/gist_base.fvecs}"
      QUERY_PATH="${GIST_QUERY_PATH:-${DATA_PREFIX}/gist/gist_query.fvecs}"
      TRAIN_PATH="${GIST_TRAIN_PATH:-${DATA_PREFIX}/gist/gist_learn.fvecs}"
      ;;
    msmarco)
      DATASET_PATH="${MSMARCO_DATASET_PATH:-${DATA_PREFIX}/msmarco/base1m.fvecs}"
      QUERY_PATH="${MSMARCO_QUERY_PATH:-${DATA_PREFIX}/msmarco/query10k.fvecs}"
      TRAIN_PATH="${MSMARCO_TRAIN_PATH:-${DATA_PREFIX}/msmarco/train1m.fvecs}"
      ;;
    openai)
      DATASET_PATH="${OPENAI_DATASET_PATH:-${DATA_PREFIX}/openai/openai_base1m.fvecs}"
      QUERY_PATH="${OPENAI_QUERY_PATH:-${DATA_PREFIX}/openai/openai_query10k.fvecs}"
      TRAIN_PATH="${OPENAI_TRAIN_PATH:-${DATA_PREFIX}/openai/openai_train1m.fvecs}"
      ;;
    deep)
      DATASET_PATH="${DEEP_DATASET_PATH:-${DATA_PREFIX}/deep1b/dataset/deep1b-96-100m.bin}"
      QUERY_PATH="${DEEP_QUERY_PATH:-${DATA_PREFIX}/deep1b/queries/queries-hard10p-deep1b-len96-1000.bin}"
      TRAIN_PATH="${DEEP_TRAIN_PATH:-${DATA_PREFIX}/deep1b/dataset/deep1b-96-100m.bin}"
      DIM_OPT=(--dim 96)
      ;;
    *)
      echo "Unknown dataset: ${name}" >&2
      return 1
      ;;
  esac
}

for DS in ${DATASETS}; do
  echo ""
  echo "############################################"
  echo "# VAQ | ${DS} | ${METHOD}"
  echo "############################################"

  if ! dataset_paths "${DS}"; then
    continue
  fi

  SKIP_ARGS=()
  if [[ "${SKIP_SEARCH}" == "1" ]]; then
    SKIP_ARGS+=(--skip_search)
  fi
  "${PYTHON}" -m vaq.eval \
    --dataset_path "${DATASET_PATH}" \
    --train_path "${TRAIN_PATH}" \
    --query_path "${QUERY_PATH}" \
    "${DIM_OPT[@]}" \
    --dataset_name "${DS}" \
    --data_root "${DATA_ROOT}" \
    --method "${METHOD}" \
    --train_size "${TRAIN_SIZE}" \
    --sample_db 10000 \
    --sample_queries 1000 \
    --results_dir "${RESULTS_SUBDIR}" \
    --vaq_binary "${VAQ_BINARY}" \
    --sample_mode first \
    "${SKIP_ARGS[@]}"

  echo "Aggregate: ${DATA_ROOT}/${RESULTS_SUBDIR}/${DS}_VAQ_adc_vs_exact_eval.csv"
done

echo "Done."
