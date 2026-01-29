#!/usr/bin/env bash
export LD_LIBRARY_PATH=/home/cpanourg/projects/2-hdvc/local/openblas/lib:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="/mnthdd/cpanourg/2-hdvc/data"
RESULTS_DIR="/mnthdd/cpanourg/2-hdvc/results/relerr_cpp"

#   --dataset_path "${DATA_ROOT}/gist/gist_base.fvecs" \
#   --query_path "${DATA_ROOT}/gist/gist_query.fvecs" \
#   --train_path "${DATA_ROOT}/gist/gist_learn.fvecs" \
#   --dataset_name gist \

run_one () {
  local M="$1"
  local NBITS="$2"
  local TRAIN_SIZE="$3"
  local SAMPLE_DB="$4"
  local SAMPLE_Q="$5"

  ./opq_eval \
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
    --opq_max_train_points 1000000
}

# Run experiments sequentially to avoid RAM spikes
# Format: M NBITS TRAIN_SIZE SAMPLE_DB SAMPLE_QUERIES
experiments=(
  # "1 8 1000000 10000 1000"
  "2 4 1000000 10000 1000"
  # "3 4 1000000 10000 1000"
  "4 4 1000000 10000 1000"
  # "6 4 1000000 10000 1000"
  "8 4 1000000 10000 1000"
  # "12 4 1000000 10000 1000"
  "16 4 1000000 10000 1000"
  # "24 4 1000000 10000 1000"
  "32 4 1000000 10000 1000"
  # "48 4 1000000 10000 1000"
  "96 4 1000000 10000 1000"
)

for exp in "${experiments[@]}"; do
  run_one $exp
done