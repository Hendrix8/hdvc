#!/usr/bin/env bash
export LD_LIBRARY_PATH=/home/cpanourg/projects/2-hdvc/local/openblas/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="/data/cpanourg/2-hdvc/data"
RESULTS_DIR="/data/cpanourg/2-hdvc/results/train_size_exps"

#   --dataset_path "${DATA_ROOT}/gist/gist_base.fvecs" \
#   --query_path "${DATA_ROOT}/gist/gist_query.fvecs" \
#   --train_path "${DATA_ROOT}/gist/gist_learn.fvecs" \
#   --dataset_name gist \

# --dataset_path "${DATA_ROOT}/deep1b/dataset/fvecs/test_1m.fvecs" \
# --query_path "${DATA_ROOT}/deep1b/dataset/fvecs/query_10k.fvecs" \
# --train_path "${DATA_ROOT}/deep1b/dataset/fvecs/learn_100m.fvecs" \
# --dataset_name deep \

# --dataset_path "${DATA_ROOT}/msmarco/base1m.fvecs" \
# --query_path "${DATA_ROOT}/msmarco/query10k.fvecs" \
# --train_path "${DATA_ROOT}/msmarco/train1m.fvecs" \
# --dataset_name msmarco \

# --dataset_path "${DATA_ROOT}/bigann/SIFT1M/bigann_base.bvecs" \
# --query_path "${DATA_ROOT}/bigann/SIFT1M/bigann_query.bvecs" \
# --train_path "${DATA_ROOT}/bigann/SIFT1M/bigann_learn.bvecs" \
# --dataset_name bigann \

# --dataset_path "${DATA_ROOT}/openai/openai_base1m.fvecs" \
# --query_path "${DATA_ROOT}/openai/openai_query10k.fvecs" \
# --train_path "${DATA_ROOT}/openai/openai_train1m.fvecs" \
# --dataset_name openai \

run_one () {
  local M="$1"
  local NBITS="$2"
  local TRAIN_SIZE="$3"
  local SAMPLE_DB="$4"
  local SAMPLE_Q="$5"

  ./pq_eval \
    --dataset_path "${DATA_ROOT}/deep1b/dataset/fvecs/test_1m.fvecs" \
    --query_path "${DATA_ROOT}/deep1b/dataset/fvecs/query_10k.fvecs" \
    --train_path "${DATA_ROOT}/deep1b/dataset/fvecs/learn_100m.fvecs" \
    --dataset_name deep \
    --n_subquantizers "${M}" \
    --nbits "${NBITS}" \
    --train_size "${TRAIN_SIZE}" \
    --sample_db "${SAMPLE_DB}" \
    --sample_queries "${SAMPLE_Q}" \
    --data_root "${DATA_ROOT}" \
    --results_dir "${RESULTS_DIR}" \
    --gpu_device "${GPU_DEVICE:-1}"
}

# Run experiments sequentially to avoid RAM spikes
# Format: M NBITS TRAIN_SIZE SAMPLE_DB SAMPLE_QUERIES
experiments=(
# DEEP
  # "1 12 1000000 10000 1000"
#   "2 12 1000000 10000 1000"
#   "3 12 1000000 10000 1000"
#   "4 12 1000000 10000 1000"
#   "6 12 1000000 10000 1000"
  "96 8 50000 10000 1000"
  "96 8 100000  10000 1000"
  "96 8 250000 10000 1000"
  "96 8 500000 10000 1000"
  "96 8 750000 10000 1000"
  "96 8 1000000 10000 1000"
#   "12 12 1000000 10000 1000"
#   "16 12 1000000 10000 1000"
#   "24 12 1000000 10000 1000"
#   "32 12 1000000 10000 1000"
#   "48 12 1000000 10000 1000"
#   "96 12 1000000 10000 1000"
)
# experiments=(
# # BigANN (SIFT1M)
#   "1 12 1000000 10000 1000"
#   "2 12 1000000 10000 1000"
#   "4 12 1000000 10000 1000"
#   "8 12 1000000 10000 1000"
#   "16 12 1000000 10000 1000"
#   "32 12 1000000 10000 1000"
#   "64 12 1000000 10000 1000"
#   "128 12 1000000 10000 1000"
# )

# experiments=(
#   # GIST
#   "1 6 1000000 10000 1000"
#   "4 6 1000000 10000 1000"
#   "8 6 1000000 10000 1000"
#   "12 6 1000000 10000 1000"
#   "16 6 1000000 10000 1000"
#   "24 6 1000000 10000 1000"
#   "48 6 1000000 10000 1000"
#   "96 6 1000000 10000 1000"
#   "192 6 1000000 10000 1000"
#   "320 6 1000000 10000 1000"
#   "480 6 1000000 10000 1000"
#   "960 6 1000000 10000 1000"
# )

# experiments=(
  # MSMARCO
  # "1 4 1000000 10000 1000"
  # # "2 4 1000000 10000 1000"
  # "4 4 1000000 10000 1000"
  # # "8 4 1000000 10000 1000"
  # "16 4 1000000 10000 1000"
  # # "32 4 1000000 10000 1000"
  # "64 4 1000000 10000 1000"
  # # "128 4 1000000 10000 1000"
  # "256 4 1000000 10000 1000"
  # # "512 4 1000000 10000 1000"
  # "1024 4 1000000 10000 1000"

# )


for exp in "${experiments[@]}"; do
  read -r M NBITS TRAIN_SIZE SAMPLE_DB SAMPLE_Q <<< "$exp"
  echo ">>> Running M=${M} nbits=${NBITS}"
  run_one "$M" "$NBITS" "$TRAIN_SIZE" "$SAMPLE_DB" "$SAMPLE_Q"
done