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

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Set DATASET_NAME here to use a specific dataset, or leave unset to use
# environment variable or default (openai)
# Options: "deep", "msmarco", "gist", "openai", "bigann"
# DATASET_NAME="gist"  # Uncomment and set to override

# Dataset configuration - can be set in file above or via DATASET_NAME environment variable
# If DATASET_NAME is set (either in file or env), it will use that dataset's config
if [[ -n "${DATASET_NAME:-}" ]]; then
  case "${DATASET_NAME}" in
    bigann)
      DATASET_PATH="${DATA_ROOT}/bigann/SIFT1M/bigann_base.bvecs"
      QUERY_PATH="${DATA_ROOT}/bigann/SIFT1M/bigann_query.bvecs"
      TRAIN_PATH="${DATA_ROOT}/bigann/SIFT1M/bigann_learn.bvecs"
      ;;
    gist)
      DATASET_PATH="${DATA_ROOT}/gist/gist_base.fvecs"
      QUERY_PATH="${DATA_ROOT}/gist/gist_query.fvecs"
      TRAIN_PATH="${DATA_ROOT}/gist/gist_learn.fvecs"
      ;;
    msmarco)
      DATASET_PATH="${DATA_ROOT}/msmarco/base1m.fvecs"
      QUERY_PATH="${DATA_ROOT}/msmarco/query10k.fvecs"
      TRAIN_PATH="${DATA_ROOT}/msmarco/train1m.fvecs"
      ;;
    openai)
      DATASET_PATH="${DATA_ROOT}/openai/openai_base1m.fvecs"
      QUERY_PATH="${DATA_ROOT}/openai/openai_query10k.fvecs"
      TRAIN_PATH="${DATA_ROOT}/openai/openai_train1m.fvecs"
      ;;
    deep)
      DATASET_PATH="${DATA_ROOT}/deep1b/fvecs/test_1m.fvecs"
      QUERY_PATH="${DATA_ROOT}/deep1b/fvecs/query_10k.fvecs"
      TRAIN_PATH="${DATA_ROOT}/deep1b/fvecs/learn_100m.fvecs"
      ;;
    *)
      echo "Error: Unknown dataset name: ${DATASET_NAME}"
      echo "Supported datasets: bigann, gist, msmarco, openai, deep"
      exit 1
      ;;
  esac
else
  # Default dataset (if DATASET_NAME not set, use openai)
  DATASET_PATH="${DATA_ROOT}/openai/openai_base1m.fvecs"
  QUERY_PATH="${DATA_ROOT}/openai/openai_query10k.fvecs"
  TRAIN_PATH="${DATA_ROOT}/openai/openai_train1m.fvecs"
  DATASET_NAME="openai"
fi




# Fixed params (set once)
# NBITS can be overridden via environment variable
NBITS=${NBITS:-12}
TRAIN_SIZE=1000000
SAMPLE_DB=10000
SAMPLE_QUERIES=1000

# List of M (n_subquantizers) values to run
# Can be overridden via M_VALUES_ENV environment variable (space-separated string)
# Otherwise, use dataset-specific defaults
if [[ -n "${M_VALUES_ENV:-}" ]]; then
  # M_VALUES_ENV is a space-separated string, convert to array
  read -ra M_VALUES <<< "${M_VALUES_ENV}"
else
  # Dataset-specific M_VALUES defaults
  case "${DATASET_NAME}" in
    bigann)
      M_VALUES=(1 2 4 8 16 32 64 128)
      ;;
    gist)
      M_VALUES=(1 3 5 8 12 20 40 60 80 120 320 480 960)
      # M_VALUES=(8 60)
      ;;
    msmarco)
      M_VALUES=(1 2 4 8 16 32 64 128 256 512 1024)
      ;;
    openai)
      M_VALUES=(1 4 8 16 32 64 128 256 512 768 1536)
      ;;
    deep)
      M_VALUES=(1 2 3 4 6 8 12 16 24 32 48 96)
      ;;
    *)
      # Default fallback
      M_VALUES=(1 4 8 16 32 64 128)
      ;;
  esac
fi


# GPU assignment: space-separated list. Jobs use round-robin.
GPU_DEVICES=(${GPU_DEVICES:-0 1})
GPU_DEVICES=(0)
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
