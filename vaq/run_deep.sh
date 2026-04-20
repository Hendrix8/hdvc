#!/usr/bin/env bash
# VAQ on Deep — PQ-style paths; you choose hyperparameters via env vars.
#
# Option A — full method string:
#   METHOD="VAQ256m32min7max8var1,HEAP" ./vaq/run_deep.sh
#
# Option B — build method from parts (used when METHOD is unset):
#   TOTAL_BITS=256 N_SUBSPACES=32 MIN_BITS=7 MAX_BITS=8 VARIANCE=1 SEARCH=HEAP ./vaq/run_deep.sh
#
# Other knobs:
#   TRAIN_SIZE, SAMPLE_DB, SAMPLE_QUERIES, REFINE, K, LEARN_RATIO, DATA_ROOT, RESULTS_SUBDIR
#   SKIP_SEARCH=1 — train/encode only in C++ (no ANN); Python still computes ADC vs exact rel. error + CSVs.
# Paths:
#   DEEP_TEST_PATH, DEEP_LEARN_PATH, DEEP_QUERY_PATH (defaults under DATA_PREFIX/deep1b/dataset/)
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
DIM="${DIM:-96}"
DATASET_NAME="${DATASET_NAME:-deep}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/vaq}"

TRAIN_SIZE="${TRAIN_SIZE:-100000}"
SAMPLE_DB="${SAMPLE_DB:-10000}"
SAMPLE_QUERIES="${SAMPLE_QUERIES:-1000}"
REFINE="${REFINE:-100,200}"
K="${K:-100}"
LEARN_RATIO="${LEARN_RATIO:-0.05}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"

# Method: set METHOD, or set these components (defaults match legacy VAQ.py)
TOTAL_BITS="${TOTAL_BITS:-256}"
N_SUBSPACES="${N_SUBSPACES:-32}"
MIN_BITS="${MIN_BITS:-7}"
MAX_BITS="${MAX_BITS:-8}"
VARIANCE="${VARIANCE:-1}"
SEARCH="${SEARCH:-HEAP}"

if [[ -n "${METHOD:-}" ]]; then
  METHOD_STR="${METHOD}"
else
  METHOD_STR="VAQ${TOTAL_BITS}m${N_SUBSPACES}min${MIN_BITS}max${MAX_BITS}var${VARIANCE},${SEARCH}"
fi

VAQ_BINARY="${VAQ_BINARY:-${REPO_ROOT}/lib/VAQ/build/examples/run_vaq}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "${VAQ_BINARY}" ]]; then
  echo "ERROR: VAQ binary not found: ${VAQ_BINARY}" >&2
  exit 1
fi
for f in "${DATASET_PATH}" "${TRAIN_PATH}" "${QUERY_PATH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}" >&2
    exit 1
  fi
done

echo "method       = ${METHOD_STR}"
echo "train_size   = ${TRAIN_SIZE}"
echo "sample_db    = ${SAMPLE_DB}  sample_queries = ${SAMPLE_QUERIES}"
echo "refine       = ${REFINE}  k=${K}  learn_ratio=${LEARN_RATIO}  skip_search=${SKIP_SEARCH}"
echo "test/base    = ${DATASET_PATH}"
echo "learn        = ${TRAIN_PATH}"
echo "queries      = ${QUERY_PATH}"
echo ""

SKIP_ARGS=()
if [[ "${SKIP_SEARCH}" == "1" ]]; then
  SKIP_ARGS+=(--skip_search)
fi

exec "${PYTHON}" -m vaq.eval \
  --dataset_path "${DATASET_PATH}" \
  --train_path "${TRAIN_PATH}" \
  --query_path "${QUERY_PATH}" \
  --dim "${DIM}" \
  --dataset_name "${DATASET_NAME}" \
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
  "${SKIP_ARGS[@]}"
