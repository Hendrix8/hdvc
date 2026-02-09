#!/usr/bin/env bash
# Run ONLY missing (M, nbits) PQ experiments for gist, in parallel on GPU.
# It checks the existing CSV and skips combinations that already exist.
#
# Usage:
#   ./run_pq_eval_parallel_mult_missing.sh
#   DRY_RUN=1 ./run_pq_eval_parallel_mult_missing.sh   # just print what would run
#
# Config below assumes:
#   - dataset = gist
#   - CSV = /data/cpanourg/2-hdvc/results/relerr_cpp/gist_PQ_adc_vs_exact_eval.csv

export LD_LIBRARY_PATH=/home/cpanourg/projects/2-hdvc/local/openblas/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="/data/cpanourg/2-hdvc/data"
RESULTS_DIR="/data/cpanourg/2-hdvc/results/relerr_cpp"
CSV_PATH="${RESULTS_DIR}/gist_PQ_adc_vs_exact_eval.csv"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# GIST dataset config
DATASET_PATH="${DATA_ROOT}/gist/gist_base.fvecs"
QUERY_PATH="${DATA_ROOT}/gist/gist_query.fvecs"
TRAIN_PATH="${DATA_ROOT}/gist/gist_learn.fvecs"
DATASET_NAME="gist"

# Fixed params (same as your gist runs)
TRAIN_SIZE=1000000
SAMPLE_DB=10000
SAMPLE_QUERIES=1000

# Target nbits and M values you care about
NBITS_LIST=(4 6 8 10 12)
M_VALUES=(1 3 5 8 12 20 40 60 80 120 320 480 960)

# GPU assignment: space-separated list. Jobs use round-robin.
GPU_DEVICES=(${GPU_DEVICES:-0 1})
echo "Using GPUs: ${GPU_DEVICES[*]} (${#GPU_DEVICES[@]} device(s))"
echo "Target nbits: ${NBITS_LIST[*]}"
echo "Target M values: ${M_VALUES[*]}"
echo "CSV: ${CSV_PATH}"

run_one() {
  local M="$1"
  local NBITS="$2"
  local GPU_IDX="$3"
  local LOG_FILE="$4"

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

# Build list of missing experiments
declare -a TODO_M
declare -a TODO_BITS

echo "Scanning CSV for existing (M, nbits) combinations..."
for bits in "${NBITS_LIST[@]}"; do
  for M in "${M_VALUES[@]}"; do
    # Match on ,M,nbits, in the appropriate columns (n_subquantizers, nbits)
    if [[ -f "${CSV_PATH}" ]] && grep -q ",${M},${bits}," "${CSV_PATH}"; then
      # Already exists, skip
      continue
    fi
    TODO_M+=("${M}")
    TODO_BITS+=("${bits}")
  done
done

if [[ ${#TODO_M[@]} -eq 0 ]]; then
  echo "No missing experiments found. Nothing to run."
  [[ "${BASH_SOURCE[0]}" != "${0}" ]] && return 0 || exit 0
fi

echo "Missing experiments to run (${#TODO_M[@]} total):"
for i in "${!TODO_M[@]}"; do
  echo "  M=${TODO_M[$i]}, nbits=${TODO_BITS[$i]}"
done

# Run missing experiments in parallel
pids=()

for i in "${!TODO_M[@]}"; do
  M="${TODO_M[$i]}"
  NBITS="${TODO_BITS[$i]}"
  log_file="${LOG_DIR}/pq_M${M}_nbits${NBITS}.log"

  if [[ -n "${DRY_RUN:-}" ]]; then
    run_one "$M" "$NBITS" "$i" "$log_file"
  else
    run_one "$M" "$NBITS" "$i" "$log_file" &
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

