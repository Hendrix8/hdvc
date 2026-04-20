#!/usr/bin/env bash
# RaBitQ on DEEP: bits per query dimension 1–12, PQ-aligned eval (first 10k base × first 1k queries).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

DATA_FP="${DATA_FP:-/data/cpanourg/2-hdvc}"
# Base / test vectors (first 1M rows used): raw .bin float32 row-major, dim=96
DATASET_PATH="${DATASET_PATH:-${DATA_FP}/data/deep1b/dataset/deep1b-96-100m.bin}"
QUERY_PATH="${QUERY_PATH:-${DATA_FP}/data/deep1b/queries/queries-hard10p-deep1b-len96-1000.bin}"
# Training vectors: separate learn file, OR set equal to DATASET_PATH for non-overlapping slices from one file
TRAIN_PATH="${TRAIN_PATH:-${DATA_FP}/data/deep1b/dataset/deep1b-96-100m.bin}"

DATASET_NAME="deep"
DIM=96
DATA_ROOT="${DATA_ROOT:-${DATA_FP}}"
# All CSVs and per-run folders: /data/cpanourg/2-hdvc/results/rabitq/...
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/rabitq}"

TRAIN_SIZE="${TRAIN_SIZE:-100000}"
SEED="${SEED:-123}"
SAMPLE_DB="${SAMPLE_DB:-10000}"
SAMPLE_QUERIES="${SAMPLE_QUERIES:-1000}"

PYTHON="${PYTHON:-python3}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for B in $(seq 1 12); do
  echo ""
  echo "--------------------------------------------"
  echo "RaBitQ | deep | bits_per_query_dim=${B} | train=${TRAIN_SIZE}"
  echo "--------------------------------------------"

  "${PYTHON}" -m rabitq.eval \
    --dataset_path "${DATASET_PATH}" \
    --train_path "${TRAIN_PATH}" \
    --query_path "${QUERY_PATH}" \
    --dim "${DIM}" \
    --dataset_name "${DATASET_NAME}" \
    --data_root "${DATA_ROOT}" \
    --bits_per_query_dim "${B}" \
    --train_size "${TRAIN_SIZE}" \
    --seed "${SEED}" \
    --sample_db "${SAMPLE_DB}" \
    --sample_queries "${SAMPLE_QUERIES}" \
    --results_dir "${RESULTS_SUBDIR}" \
    --sample_mode first
done

echo "============================================"
echo "Aggregate CSV: ${DATA_ROOT}/${RESULTS_SUBDIR}/${DATASET_NAME}_RaBitQ_adc_vs_exact_eval.csv"
echo "Per-run folders: ${DATA_ROOT}/${RESULTS_SUBDIR}/${DATASET_NAME}/bits*_train*_*/summary.csv"
echo "============================================"
