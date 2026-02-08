#!/usr/bin/env bash
# Run PQ experiments in parallel. Edit the experiments array below, then run this script.
#
# Usage:
#   ./run_pq_eval_parallel.sh
#
# Options (env vars):
#   GPU_DEVICES="0 1 2 3" - GPUs to use, round-robin across jobs (default: 0 only)
#   DRY_RUN=1           - print commands without running

export LD_LIBRARY_PATH=/home/cpanourg/projects/2-hdvc/local/openblas/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="/data/cpanourg/2-hdvc/data"
RESULTS_DIR="/data/cpanourg/2-hdvc/results/relerr_cpp"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# # Dataset config (change as needed)
# DATASET_PATH="${DATA_ROOT}/deep1b/dataset/fvecs/test_1m.fvecs"
# QUERY_PATH="${DATA_ROOT}/deep1b/dataset/fvecs/query_10k.fvecs"
# TRAIN_PATH="${DATA_ROOT}/deep1b/dataset/fvecs/learn_100m.fvecs"
# DATASET_NAME="deep"

# Dataset config (change as needed)
DATASET_PATH="${DATA_ROOT}/bigann/SIFT1M/bigann_base.bvecs"
QUERY_PATH="${DATA_ROOT}/bigann/SIFT1M/bigann_query.bvecs"
TRAIN_PATH="${DATA_ROOT}/bigann/SIFT1M/bigann_learn.bvecs"
DATASET_NAME="bigann"


# Experiments: M NBITS TRAIN_SIZE SAMPLE_DB SAMPLE_QUERIES
# experiments=(
# # DEEP
#   "1 12 1000000 10000 1000"
#   "2 12 1000000 10000 1000"
#   "3 12 1000000 10000 1000"
#   "4 12 1000000 10000 1000"
#   "6 12 1000000 10000 1000"
#   "8 12 1000000 10000 1000"
#   "12 12 1000000 10000 1000"
#   "16 12 1000000 10000 1000"
#   "24 12 1000000 10000 1000"
#   "32 12 1000000 10000 1000"
#   "48 12 1000000 10000 1000"
#   "96 12 1000000 10000 1000"
# )

experiments=(
# BigANN (SIFT1M)
  "1 4 1000000 10000 1000"
  "2 4 1000000 10000 1000"
  "4 4 1000000 10000 1000"
  "8 4 1000000 10000 1000"
  "16 4 1000000 10000 1000"
  "32 4 1000000 10000 1000"
  "64 4 1000000 10000 1000"
  "128 4 1000000 10000 1000"
)

# GPU assignment: space-separated list. Jobs use round-robin.
# Default: both GPUs (0 1). Override: GPU_DEVICES="0 1 2" ./run_pq_eval_parallel.sh
GPU_DEVICES=(${GPU_DEVICES:-0 1})

run_one() {
  local M="$1"
  local NBITS="$2"
  local TRAIN_SIZE="$3"
  local SAMPLE_DB="$4"
  local SAMPLE_Q="$5"
  local GPU_IDX="$6"
  local LOG_FILE="$7"

  local GpuDev="${GPU_DEVICES[$((GPU_IDX % ${#GPU_DEVICES[@]}))]}"

  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "[DRY_RUN] M=${M} nbits=${NBITS} gpu=${GpuDev} log=${LOG_FILE}"
    return 0
  fi

  echo ">>> [$(date +%H:%M:%S)] Starting M=${M} nbits=${NBITS} (GPU ${GpuDev}) -> ${LOG_FILE}"
  ./pq_eval \
    --dataset_path "${DATASET_PATH}" \
    --query_path "${QUERY_PATH}" \
    --train_path "${TRAIN_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --n_subquantizers "${M}" \
    --nbits "${NBITS}" \
    --train_size "${TRAIN_SIZE}" \
    --sample_db "${SAMPLE_DB}" \
    --sample_queries "${SAMPLE_Q}" \
    --data_root "${DATA_ROOT}" \
    --results_dir "${RESULTS_DIR}" \
    --gpu_device "${GpuDev}" >> "$LOG_FILE" 2>&1
  local rc=$?
  echo ">>> [$(date +%H:%M:%S)] Done M=${M} nbits=${NBITS} (exit $rc)"
  return $rc
}

# Run experiments in parallel
pids=()

for i in "${!experiments[@]}"; do
  exp="${experiments[$i]}"
  read -r M NBITS TRAIN_SIZE SAMPLE_DB SAMPLE_Q <<< "$exp"
  log_file="${LOG_DIR}/pq_M${M}_nbits${NBITS}.log"

  if [[ -n "${DRY_RUN:-}" ]]; then
    run_one "$M" "$NBITS" "$TRAIN_SIZE" "$SAMPLE_DB" "$SAMPLE_Q" "$i" "$log_file"
  else
    run_one "$M" "$NBITS" "$TRAIN_SIZE" "$SAMPLE_DB" "$SAMPLE_Q" "$i" "$log_file" &
    pids+=($!)
  fi
done

if [[ -z "${DRY_RUN:-}" ]]; then
  echo "Waiting for ${#pids[@]} jobs..."
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      ((failed++)) || true
    fi
  done
  echo "All done. Failed: $failed"
  exit $failed
fi
