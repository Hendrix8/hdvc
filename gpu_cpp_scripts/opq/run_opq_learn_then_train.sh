#!/usr/bin/env bash
# Run OPQ experiments in two phases:
#   1) SEQUENTIAL: learn OPQ rotations per (dataset, M) on CPU with learn_opq_rot
#   2) PARALLEL:   train PQ on GPU + evaluate using train_opq, reusing cached OPQ
#
# Usage:
#   ./run_opq_learn_then_train.sh
#   ./run_opq_learn_then_train.sh deep gist --nbits 4 6 8
#
# You can edit DATASETS, NBITS_VALUES, and GPU_DEVICES below, or override
# datasets/nbits from the command line.

# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# -------------------------------
# User-configurable defaults
# -------------------------------

# Datasets to run
# DATASETS=("deep" "msmarco" "gist" "openai" "bigann")
DATASETS=("deep") # "gist" "msmarco")

# nbits values
NBITS_VALUES=(4 6 8 10 12)


# Number of vectors to use for OPQ rotation training (per dataset),
# independent of the PQ train_size used later in Phase 2.
ROT_TRAIN_SIZE=1000000
OPQ_MAX_TRAIN_POINTS=500000 # $((256 * 256))

# Number of training vectors to use for PQ (train_opq) in Phase 2.
TRAIN_SIZE=1000000

# GPUs to use (round-robin in parallel phase)
GPU_DEVICES=(0)


# -------------------------------
# Paths/config shared with C++
# -------------------------------

DATA_ROOT="/data/cpanourg/2-hdvc/data"
RESULTS_DIR="/data/cpanourg/2-hdvc/results/relerr_cpp"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

export LD_LIBRARY_PATH=/home/cpanourg/projects/2-hdvc/local/openblas/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:$LD_LIBRARY_PATH

# -------------------------------
# CLI overrides
# -------------------------------

TEMP_DATASETS=()
TEMP_NBITS=()
PARSE_NBITS=false

for arg in "$@"; do
  if [[ "${arg}" == "--nbits" ]]; then
    PARSE_NBITS=true
  elif [[ "${PARSE_NBITS}" == true ]]; then
    TEMP_NBITS+=("${arg}")
  else
    TEMP_DATASETS+=("${arg}")
  fi
done

if [[ ${#TEMP_DATASETS[@]} -gt 0 ]]; then
  DATASETS=("${TEMP_DATASETS[@]}")
fi

if [[ ${#TEMP_NBITS[@]} -gt 0 ]]; then
  NBITS_VALUES=("${TEMP_NBITS[@]}")
fi

echo "============================================================"
echo "OPQ pipeline: sequential OPQ learning, then parallel PQ+eval"
echo "============================================================"
echo "Datasets   : ${DATASETS[*]}"
echo "nbits      : ${NBITS_VALUES[*]}"
echo "GPUs       : ${GPU_DEVICES[*]} (${#GPU_DEVICES[@]} device(s))"
echo "DATA_ROOT  : ${DATA_ROOT}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "Script dir : ${SCRIPT_DIR}"
echo "============================================================"
echo

START_TIME=$(date +%s)

# -------------------------------
# Helpers
# -------------------------------

dataset_config() {
  local name="$1"
  case "${name}" in
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
      DATASET_PATH="${DATA_ROOT}/deep1b/dataset/fvecs/test_1m.fvecs"
      QUERY_PATH="${DATA_ROOT}/deep1b/dataset/fvecs/query_10k.fvecs"
      TRAIN_PATH="${DATA_ROOT}/deep1b/dataset/fvecs/learn_100m.fvecs"
      ;;
    *)
      echo "Error: Unknown dataset name: ${name}" >&2
      echo "Supported datasets: bigann, gist, msmarco, openai, deep" >&2
      return 1
      ;;
  esac
}

dataset_m_values() {
  local name="$1"
  case "${name}" in
    bigann)
      echo "1 2 4 8 16 32 64 128"
      ;;
    gist)
      echo "1 3 5 8 12 20 40 60 80 120 320 480 960"
      ;;
    msmarco)
      echo "1 2 4 8 16 32 64 128 256 512 1024"
      ;;
    openai)
      echo "1 4 8 16 32 64 128 256 512 768 1536"
      ;;
    deep)
      # echo "1 2 3 4 6 8 12 16 24 32 48 96"
      echo "1 4 12 24 32 96"
      ;;
    *)
      echo "1 4 8 16 32 64 128"
      ;;
  esac
}

# -------------------------------
# Phase 1: Sequential OPQ learning
# -------------------------------

echo "============================"
echo "Phase 1: OPQ rotation learn"
echo "============================"


for DATASET_NAME in "${DATASETS[@]}"; do
  echo
  echo "----------------------------------------"
  echo "Dataset: ${DATASET_NAME}  [OPQ learn]"
  echo "Time   : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "----------------------------------------"

  if ! dataset_config "${DATASET_NAME}"; then
    echo "Skipping dataset ${DATASET_NAME} due to config error."
    continue
  fi

  M_VALUES_STR="$(dataset_m_values "${DATASET_NAME}")"
  read -ra M_VALUES <<< "${M_VALUES_STR}"
  echo "M values: ${M_VALUES[*]}"

  OPQ_MODEL_ROOT="${RESULTS_DIR}/opq_transforms"
  OPQ_MODEL_DIR="${OPQ_MODEL_ROOT}/${DATASET_NAME}"
  mkdir -p "${OPQ_MODEL_DIR}"

  for M in "${M_VALUES[@]}"; do
    OPQ_MODEL_PATH="${OPQ_MODEL_DIR}/opq_M${M}_${OPQ_MAX_TRAIN_POINTS}.vt"

    echo
    echo "  >>> [$(date +%H:%M:%S)] OPQ learn for M=${M}"
    echo "      train_path    = ${TRAIN_PATH}"
    echo "      rot_train_sz  = ${ROT_TRAIN_SIZE}"
    echo "      max_train_pts = ${OPQ_MAX_TRAIN_POINTS}"
    echo "      model_path    = ${OPQ_MODEL_PATH}"

    if [[ -f "${OPQ_MODEL_PATH}" ]]; then
      echo "      OPQ model already exists, skipping."
      continue
    fi

    ./learn_opq_rot \
      --train_path "${TRAIN_PATH}" \
      --dataset_name "${DATASET_NAME}" \
      --n_subquantizers "${M}" \
      --rot_train_size "${ROT_TRAIN_SIZE}" \
      --opq_max_train_points "${OPQ_MAX_TRAIN_POINTS}" \
      --results_dir "${RESULTS_DIR}" \
      --opq_model_path "${OPQ_MODEL_PATH}"

    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "      ✗ learn_opq_rot failed for M=${M} (rc=${rc})"
    else
      echo "      ✓ OPQ learn finished for M=${M}"
    fi
  done

  echo
  echo "Finished OPQ learning for dataset ${DATASET_NAME}"
done

echo
echo "============================"
echo "Phase 2: PQ train + eval (GPU, parallel)"
echo "============================"

echo "Using GPUs: ${GPU_DEVICES[*]} (${#GPU_DEVICES[@]} device(s))"
echo "nbits grid: ${NBITS_VALUES[*]}"
echo

run_one_train() {
  local dataset_name="$1"
  local dataset_path="$2"
  local query_path="$3"
  local train_path="$4"
  local M="$5"
  local nbits="$6"
  local gpu_idx="$7"
  local log_file="$8"

  local gpu_dev="${GPU_DEVICES[$((gpu_idx % ${#GPU_DEVICES[@]}))]}"

  local opq_model_root="${RESULTS_DIR}/opq_transforms"
  local opq_model_dir="${opq_model_root}/${dataset_name}"
  local opq_model_path="${opq_model_dir}/opq_M${M}_${OPQ_MAX_TRAIN_POINTS}.vt"

  echo ">>> [$(date +%H:%M:%S)] START dataset=${dataset_name} M=${M} nbits=${nbits} gpu=${gpu_dev}"
  echo "    log   = ${log_file}"
  echo "    model = ${opq_model_path}"

  if [[ ! -f "${opq_model_path}" ]]; then
    echo "!! OPQ model missing for dataset=${dataset_name}, M=${M}: ${opq_model_path}" | tee -a "${log_file}"
    return 1
  fi

  ./train_opq \
    --dataset_path "${dataset_path}" \
    --query_path "${query_path}" \
    --train_path "${train_path}" \
    --dataset_name "${dataset_name}" \
    --n_subquantizers "${M}" \
    --nbits "${nbits}" \
    --train_size "${TRAIN_SIZE}" \
    --sample_db 10000 \
    --sample_queries 1000 \
    --data_root "${DATA_ROOT}" \
    --results_dir "${RESULTS_DIR}" \
    --opq_model_path "${opq_model_path}" \
    --gpu_device "${gpu_dev}" >> "${log_file}" 2>&1

  local rc=$?
  echo ">>> [$(date +%H:%M:%S)] DONE dataset=${dataset_name} M=${M} nbits=${nbits} (rc=${rc})"
  return "${rc}"
}

for DATASET_NAME in "${DATASETS[@]}"; do
  echo
  echo "----------------------------------------"
  echo "Dataset: ${DATASET_NAME}  [PQ train + eval]"
  echo "Time   : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "----------------------------------------"

  if ! dataset_config "${DATASET_NAME}"; then
    echo "Skipping dataset ${DATASET_NAME} due to config error."
    continue
  fi

  M_VALUES_STR="$(dataset_m_values "${DATASET_NAME}")"
  read -ra M_VALUES <<< "${M_VALUES_STR}"

  echo "M values: ${M_VALUES[*]}"

  for NBITS in "${NBITS_VALUES[@]}"; do
    echo
    echo "  >>> nbits=${NBITS} (dataset=${DATASET_NAME})"
    echo "      launch time: $(date '+%Y-%m-%d %H:%M:%S')"

    pids=()
    idx=0

    for M in "${M_VALUES[@]}"; do
      log_file="${LOG_DIR}/opq_${DATASET_NAME}_M${M}_nbits${NBITS}.log"

      run_one_train "${DATASET_NAME}" "${DATASET_PATH}" "${QUERY_PATH}" "${TRAIN_PATH}" \
        "${M}" "${NBITS}" "${idx}" "${log_file}" &
      pids+=($!)
      ((idx++))
    done

    echo "  Waiting for ${#pids[@]} jobs for nbits=${NBITS}..."
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        ((failed++)) || true
      fi
    done
    echo "  All jobs for nbits=${NBITS} finished. Failed: ${failed}"
  done

  echo
  echo "Completed PQ train + eval for dataset ${DATASET_NAME}"
  echo "----------------------------------------"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo
echo "============================================================"
echo "OPQ pipeline finished"
echo "============================================================"
echo "Total time : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Datasets   : ${DATASETS[*]}"
echo "nbits grid : ${NBITS_VALUES[*]}"
echo "Logs       : ${LOG_DIR}"
echo "============================================================"

