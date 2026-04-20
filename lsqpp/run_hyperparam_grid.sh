#!/usr/bin/env bash
# LSQ++ (Faiss LocalSearchQuantizer ST_norm_qint8): sweep M × nbits × train_size on Deep.
# Writes one row per run to:
#   /data/cpanourg/2-hdvc/results/${DATASET_NAME}_LSQpp_adc_vs_exact_eval.csv
# (relative errors, train/encode/LUT/ADC/cdist times, etc. — see lsqpp/eval.py summary).
#
# Edit the constants below (grid lists, paths). No environment variables required.
# Set DRY_RUN=1 below to print commands only (no Faiss runs).
#
# Usage:
#   bash lsqpp/run_hyperparam_grid.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Output root: aggregate CSV lands under ${DATA_ROOT}/results/ ---
DATA_ROOT="/data/cpanourg/2-hdvc"
DATA_PREFIX="${DATA_ROOT}/data"
RESULTS_SUBDIR="results"

# --- Deep (PQ-style paths) ---
DATASET_PATH="${DATA_PREFIX}/deep1b/dataset/test_1m.bin"
TRAIN_PATH="${DATA_PREFIX}/deep1b/dataset/learn_100m.bin"
QUERY_PATH="${DATA_PREFIX}/deep1b/dataset/query_10k.bin"
DIM=96
DATASET_NAME="deep"

# --- Fixed eval subset (match PQ/OPQ rel-error protocols) ---
SAMPLE_DB=10000
SAMPLE_QUERIES=1000
SEED=123

# --- Grid (aligned with scripts/rel_error_exps/run_opq.sh style) ---
N_SUBQUANTIZERS_LIST="1 8 32 64 96"
NBITS_LIST="4 6 8 10 12"
TRAIN_SIZES="10000 100000 1000000"

PYTHON="python3"
# Set to 1 to print commands without running (or: DRY_RUN=1 bash ... for one-off)
DRY_RUN=0

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONPATH="${REPO_ROOT}"

for f in "${DATASET_PATH}" "${TRAIN_PATH}" "${QUERY_PATH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}" >&2
    exit 1
  fi
done

echo "=== LSQ++ hyperparameter grid (Deep) ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "RESULTS -> ${DATA_ROOT}/${RESULTS_SUBDIR}/${DATASET_NAME}_LSQpp_adc_vs_exact_eval.csv"
echo "test: ${DATASET_PATH}"
echo "learn: ${TRAIN_PATH}"
echo "queries: ${QUERY_PATH}"
echo "M list: ${N_SUBQUANTIZERS_LIST}"
echo "nbits list: ${NBITS_LIST}"
echo "train sizes: ${TRAIN_SIZES}"
echo "DRY_RUN=${DRY_RUN}"
echo ""

RUNS=0
FAILED=0

for M in ${N_SUBQUANTIZERS_LIST}; do
  for NBITS in ${NBITS_LIST}; do
    for TRAIN_SIZE in ${TRAIN_SIZES}; do
      RUNS=$((RUNS + 1))
      echo ""
      echo "--------------------------------------------"
      echo "Run ${RUNS}:  M=${M}  nbits=${NBITS}  bits/vector=$((M * NBITS))  train_size=${TRAIN_SIZE}"
      echo "--------------------------------------------"

      CMD=(
        "${PYTHON}" -m lsqpp.eval
        --dataset_path "${DATASET_PATH}"
        --train_path "${TRAIN_PATH}"
        --query_path "${QUERY_PATH}"
        --dim "${DIM}"
        --dataset_name "${DATASET_NAME}"
        --data_root "${DATA_ROOT}"
        --n_subquantizers "${M}"
        --nbits "${NBITS}"
        --train_size "${TRAIN_SIZE}"
        --sample_db "${SAMPLE_DB}"
        --sample_queries "${SAMPLE_QUERIES}"
        --results_dir "${RESULTS_SUBDIR}"
        --seed "${SEED}"
        --sample_mode first
      )

      if [[ "${DRY_RUN}" == "1" ]]; then
        printf ' %q' "${CMD[@]}"
        echo ""
        continue
      fi

      if ! "${CMD[@]}"; then
        echo "ERROR: run failed (M=${M} nbits=${NBITS} train=${TRAIN_SIZE})" >&2
        FAILED=$((FAILED + 1))
      fi
    done
  done
done

echo ""
echo "Finished ${RUNS} planned runs. Failed: ${FAILED}."
echo "Aggregate CSV: ${DATA_ROOT}/${RESULTS_SUBDIR}/${DATASET_NAME}_LSQpp_adc_vs_exact_eval.csv"
