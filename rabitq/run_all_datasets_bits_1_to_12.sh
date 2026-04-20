#!/usr/bin/env bash
# RaBitQ sweep (bits 1–12): BigANN/SIFT, GIST, MSMARCO, OpenAI (+ optional Deep).
# Paths follow gpu_cpp_scripts/opq/run_opq_learn_then_train.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${DATA_ROOT:-/data/cpanourg/2-hdvc}"
DATA_ROOT="${DATA_ROOT%/}"
# Vector files usually live under ${DATA_ROOT}/data/{bigann,gist,...} (see scripts/evals/config.py)
DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT}/data}"

# Space-separated: bigann gist msmarco openai  (add "deep" for Deep1B .bin)
DATASETS="${DATASETS:-bigann gist msmarco openai}"

RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/rabitq}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
SEED="${SEED:-123}"
SAMPLE_DB="${SAMPLE_DB:-10000}"
SAMPLE_QUERIES="${SAMPLE_QUERIES:-1000}"
PYTHON="${PYTHON:-python3}"

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
      echo "Use: bigann, gist, msmarco, openai, deep" >&2
      return 1
      ;;
  esac
}

for DS in ${DATASETS}; do
  echo ""
  echo "############################################"
  echo "# Dataset: ${DS}"
  echo "############################################"

  if ! dataset_paths "${DS}"; then
    continue
  fi

  for B in $(seq 1 12); do
    echo ""
    echo "--------------------------------------------"
    echo "RaBitQ | ${DS} | bits_per_query_dim=${B} | train=${TRAIN_SIZE}"
    echo "--------------------------------------------"

    "${PYTHON}" -m rabitq.eval \
      --dataset_path "${DATASET_PATH}" \
      --train_path "${TRAIN_PATH}" \
      --query_path "${QUERY_PATH}" \
      "${DIM_OPT[@]}" \
      --dataset_name "${DS}" \
      --data_root "${DATA_ROOT}" \
      --bits_per_query_dim "${B}" \
      --train_size "${TRAIN_SIZE}" \
      --seed "${SEED}" \
      --sample_db "${SAMPLE_DB}" \
      --sample_queries "${SAMPLE_QUERIES}" \
      --results_dir "${RESULTS_SUBDIR}" \
      --sample_mode first
  done

  echo "Aggregate CSV: ${DATA_ROOT}/${RESULTS_SUBDIR}/${DS}_RaBitQ_adc_vs_exact_eval.csv"
done

echo ""
echo "Done. Per-dataset CSVs under ${DATA_ROOT}/${RESULTS_SUBDIR}/"
