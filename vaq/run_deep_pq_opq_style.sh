#!/usr/bin/env bash
# VAQ on Deep — same *data layout* as PQ and same *train_size grid* as OPQ.
#
# Paths (PQ-style, three files): test_1m.bin, learn_100m.bin, query_10k.bin
# Train sizes (OPQ run_opq.sh): 10k, 100k, 1M (override TRAIN_SIZES)
#
# VAQ method strings are not the same knobs as PQ/OPQ (M × nbits). The default is the
# proven config from scripts/rel_error_exps/VAQ.py. Override with METHODS (space-separated)
# or a single METHOD for multiple train sizes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${DATA_ROOT:-/data/cpanourg/2-hdvc}"
DATA_ROOT="${DATA_ROOT%/}"
DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT}/data}"

DATASET_PATH="${DEEP_TEST_PATH:-${DATA_PREFIX}/deep1b/dataset/test_1m.bin}"
TRAIN_PATH="${DEEP_LEARN_PATH:-${DATA_PREFIX}/deep1b/dataset/learn_100m.bin}"
QUERY_PATH="${DEEP_QUERY_PATH:-${DATA_PREFIX}/deep1b/dataset/query_10k.bin}"
DIM=96

DATASET_NAME="deep"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/vaq}"
SAMPLE_DB="${SAMPLE_DB:-10000}"
SAMPLE_QUERIES="${SAMPLE_QUERIES:-1000}"

# OPQ run_opq.sh: TRAIN_SIZES=(10000 100000 1000000)
TRAIN_SIZES="${TRAIN_SIZES:-10000 100000 1000000}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"

# Default VAQ recipe (same as legacy VAQ.py). Add more space-separated entries to sweep.
METHODS="${METHODS:-VAQ256m32min7max8var1,HEAP}"

VAQ_BINARY="${VAQ_BINARY:-${REPO_ROOT}/lib/VAQ/build/examples/run_vaq}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "${VAQ_BINARY}" ]]; then
  echo "ERROR: VAQ binary not found: ${VAQ_BINARY}" >&2
  exit 1
fi

for f in "${DATASET_PATH}" "${TRAIN_PATH}" "${QUERY_PATH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing data file: ${f}" >&2
    exit 1
  fi
done

echo "DATA_ROOT=${DATA_ROOT}"
echo "test (base): ${DATASET_PATH}"
echo "learn:       ${TRAIN_PATH}"
echo "queries:     ${QUERY_PATH}"
echo "methods:     ${METHODS}"
echo "train sizes: ${TRAIN_SIZES}"
echo "SKIP_SEARCH=${SKIP_SEARCH}"
echo ""

for METHOD in ${METHODS}; do
  for TRAIN_SIZE in ${TRAIN_SIZES}; do
    echo ""
    echo "============================================"
    echo "VAQ | deep | method=${METHOD} | train_size=${TRAIN_SIZE}"
    echo "============================================"

    SKIP_ARGS=()
    if [[ "${SKIP_SEARCH}" == "1" ]]; then
      SKIP_ARGS+=(--skip_search)
    fi
    "${PYTHON}" -m vaq.eval \
      --dataset_path "${DATASET_PATH}" \
      --train_path "${TRAIN_PATH}" \
      --query_path "${QUERY_PATH}" \
      --dim "${DIM}" \
      --dataset_name "${DATASET_NAME}" \
      --data_root "${DATA_ROOT}" \
      --method "${METHOD}" \
      --train_size "${TRAIN_SIZE}" \
      --sample_db "${SAMPLE_DB}" \
      --sample_queries "${SAMPLE_QUERIES}" \
      --results_dir "${RESULTS_SUBDIR}" \
      --vaq_binary "${VAQ_BINARY}" \
      --sample_mode first \
      "${SKIP_ARGS[@]}"
  done
done

echo ""
echo "Done. Aggregate CSV: ${DATA_ROOT}/${RESULTS_SUBDIR}/${DATASET_NAME}_VAQ_adc_vs_exact_eval.csv"
