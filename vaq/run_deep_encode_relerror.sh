#!/usr/bin/env bash
# Deep VAQ: train + codes + Python ADC vs exact rel. error + CSVs. No C++ ANN (--skip_search).
# Self-contained: edit the *_ROOT / paths block below if your data lives elsewhere.
# Usage: bash vaq/run_deep_encode_relerror.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- constants only (no env vars required) ---
DATA_ROOT="/data/cpanourg/2-hdvc"
DATA_PREFIX="${DATA_ROOT}/data"
DATASET_PATH="${DATA_PREFIX}/deep1b/dataset/test_1m.bin"
TRAIN_PATH="${DATA_PREFIX}/deep1b/dataset/learn_100m.bin"
QUERY_PATH="${DATA_PREFIX}/deep1b/dataset/query_10k.bin"
DIM=96
DATASET_NAME="deep"
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
for f in "${DATASET_PATH}" "${TRAIN_PATH}" "${QUERY_PATH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}" >&2
    exit 1
  fi
done

echo "repo         = ${REPO_ROOT}"
echo "data_root    = ${DATA_ROOT}"
echo "method       = ${METHOD_STR}"
echo "train_size   = ${TRAIN_SIZE}"
echo "skip_search  = 1 (encode + rel. error only; no C++ ANN)"
echo ""

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
  --skip_search
