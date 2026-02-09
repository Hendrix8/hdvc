#!/usr/bin/env bash
# Run PQ experiments in parallel. Set nbits, train_size, sample_db, sample_queries once,
# then provide a list of M (n_subquantizers) values to run.
#
# Usage:
#   ./run_pq_eval_parallel_mult.sh
#
# Options (env vars):
#   GPU_DEVICES="0 1 2 3" - GPUs to use, round-robin (default: 0 1)
#   DRY_RUN=1           - print commands without running

export LD_LIBRARY_PATH=/home/cpanourg/projects/2-hdvc/local/openblas/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="/data/cpanourg/2-hdvc/data"
RESULTS_DIR="/data/cpanourg/2-hdvc/results/relerr_cpp"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Dataset config (change as needed)
DATASET_PATH="${DATA_ROOT}/bigann/SIFT1M/bigann_base.bvecs"
QUERY_PATH="${DATA_ROOT}/bigann/SIFT1M/bigann_query.bvecs"
TRAIN_PATH="${DATA_ROOT}/bigann/SIFT1M/bigann_learn.bvecs"
DATASET_NAME="bigann"

# Fixed params (set once)
NBITS=12
TRAIN_SIZE=1000000
SAMPLE_DB=10000
SAMPLE_QUERIES=1000

# List of M (n_subquantizers) values to run
M_VALUES=(
  # SIFT1M
  1
  2
  4
  8
  16
  32
  64
  128
)

# GPU assignment: space-separated list. Jobs use round-robin.
GPU_DEVICES=(${GPU_DEVICES:-0 1})
echo "Using GPUs: ${GPU_DEVICES[*]} (${#GPU_DEVICES[@]} device(s))"
echo "Fixed: nbits=${NBITS} train=${TRAIN_SIZE} sample_db=${SAMPLE_DB} sample_q=${SAMPLE_QUERIES}"
echo "M values: ${M_VALUES[*]}"

run_one() {
  local M="$1"
  local GPU_IDX="$2"
  local LOG_FILE="$3"

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
    --sample_queries "${SAMPLE_QUERIES}" \
    --data_root "${DATA_ROOT}" \
    --results_dir "${RESULTS_DIR}" \
    --gpu_device "${GpuDev}" >> "$LOG_FILE" 2>&1
  local rc=$?
  echo ">>> [$(date +%H:%M:%S)] Done M=${M} nbits=${NBITS} (exit $rc)"
  return $rc
}

# Run experiments in parallel
pids=()

for i in "${!M_VALUES[@]}"; do
  M="${M_VALUES[$i]}"
  log_file="${LOG_DIR}/pq_M${M}_nbits${NBITS}.log"

  if [[ -n "${DRY_RUN:-}" ]]; then
    run_one "$M" "$i" "$log_file"
  else
    run_one "$M" "$i" "$log_file" &
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
  [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 0 || exit 0
fi
