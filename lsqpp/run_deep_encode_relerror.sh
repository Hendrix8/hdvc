#!/usr/bin/env bash
# Deep LSQ++ (Faiss): train + encode + ADC vs exact rel. error + CSVs. No IVF/HNSW search.
# Edit paths below if needed. Usage: bash lsqpp/run_deep_encode_relerror.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="/data/cpanourg/2-hdvc"
DATA_PREFIX="${DATA_ROOT}/data"
DATASET_PATH="${DATA_PREFIX}/deep1b/dataset/test_1m.bin"
TRAIN_PATH="${DATA_PREFIX}/deep1b/dataset/learn_100m.bin"
QUERY_PATH="${DATA_PREFIX}/deep1b/dataset/query_10k.bin"
DIM=96
DATASET_NAME="deep"
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

for f in "${DATASET_PATH}" "${TRAIN_PATH}" "${QUERY_PATH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}" >&2
    exit 1
  fi
done

exec "${PYTHON}" -m lsqpp.eval \
  --dataset_path "${DATASET_PATH}" \
  --train_path "${TRAIN_PATH}" \
  --query_path "${QUERY_PATH}" \
  --dim "${DIM}" \
  --dataset_name "${DATASET_NAME}" \
  --data_root "${DATA_ROOT}" \
  --n_subquantizers "${N_SUBQUANTIZERS}" \
  --nbits "${NBITS}" \
  --train_size "${TRAIN_SIZE}" \
  --sample_db "${SAMPLE_DB}" \
  --sample_queries "${SAMPLE_QUERIES}" \
  --results_dir "${RESULTS_SUBDIR}" \
  --sample_mode first
