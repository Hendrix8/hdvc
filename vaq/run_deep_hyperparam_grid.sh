#!/usr/bin/env bash
# Deep VAQ — hyperparameter *grids* aligned with PQ / OPQ experiment layouts.
#
# OPQ reference: scripts/rel_error_exps/run_opq.sh
#   N_SUBQUANTIZERS_LIST=(4 8 16 32)  NBITS_LIST=(8 9 10)  TRAIN_SIZES=(10000 100000 1000000)
#
# PQ reference: scripts/rel_error_exps/run_pq.sh
#   N_SUBQUANTIZERS_LIST × NBITS_LIST × TRAIN_SIZES (same M / nbits meaning: subspaces × bits per subquantizer)
#
# VAQ method shape: VAQ{TOTAL}m{M}min{MIN_BITS}max{MAX_BITS}var{VAR},{SEARCH}
#   - We set M = number of subspaces (same role as PQ/OPQ "M" / n_subquantizers).
#   - We set TOTAL = max(M * NBITS, MIN_TOTAL_VAQ_BITS). The leading TOTAL is the VAQ bit budget field;
#     small values often make the internal GLP step fail — MIN_TOTAL_VAQ_BITS defaults to 128.
#   - MIN_BITS / MAX_BITS default to 7 / 8 (stable with the stock build); they are *not* forced to NBITS
#     (unlike PQ/OPQ per-codebook depth). Override MIN_BITS, MAX_BITS if your build accepts them.
#
# Usage:
#   ./vaq/run_deep_hyperparam_grid.sh
#   PQ_GRID=1 N_SUBQUANTIZERS_LIST="8 16" NBITS_LIST="8 10" ./vaq/run_deep_hyperparam_grid.sh
#   DRY_RUN=1 ./vaq/run_deep_hyperparam_grid.sh   # print only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${DATA_ROOT:-/data/cpanourg/2-hdvc}"
DATA_ROOT="${DATA_ROOT%/}"
DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT}/data}"

DATASET_PATH="${DEEP_TEST_PATH:-${DATA_PREFIX}/deep1b/dataset/test_1m.bin}"
TRAIN_PATH="${DEEP_LEARN_PATH:-${DATA_PREFIX}/deep1b/dataset/learn_100m.bin}"
QUERY_PATH="${DEEP_QUERY_PATH:-${DATA_PREFIX}/deep1b/dataset/query_10k.bin}"
DIM="${DIM:-96}"
DATASET_NAME="${DATASET_NAME:-deep}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/vaq}"
SAMPLE_DB="${SAMPLE_DB:-10000}"
SAMPLE_QUERIES="${SAMPLE_QUERIES:-1000}"

REFINE="${REFINE:-100,200}"
K="${K:-100}"
LEARN_RATIO="${LEARN_RATIO:-0.05}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"

# VAQ-specific (defaults match legacy VAQ.py-style stability)
MIN_BITS="${MIN_BITS:-7}"
MAX_BITS="${MAX_BITS:-8}"
VARIANCE="${VARIANCE:-1}"
SEARCH="${SEARCH:-HEAP}"
MIN_TOTAL_VAQ_BITS="${MIN_TOTAL_VAQ_BITS:-128}"

VAQ_BINARY="${VAQ_BINARY:-${REPO_ROOT}/lib/VAQ/build/examples/run_vaq}"
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"

# --- Grids (same variable names as PQ/OPQ launchers) ---
# PQ_GRID=0 → same M / nbits / train lists as scripts/rel_error_exps/run_opq.sh (Deep)
# PQ_GRID=1 → wider M sweep (adds 64); still override lists with env if you want
PQ_GRID="${PQ_GRID:-0}"

if [[ "${PQ_GRID}" == "1" ]]; then
  N_SUBQUANTIZERS_LIST="${N_SUBQUANTIZERS_LIST:-1 8 32 64 96}"
  NBITS_LIST="${NBITS_LIST:-4 6 8 10 12}"
else
  N_SUBQUANTIZERS_LIST="${N_SUBQUANTIZERS_LIST:-1 8 32 64 96}"
  NBITS_LIST="${NBITS_LIST:-4 6 8 10 12}"
fi

TRAIN_SIZES="${TRAIN_SIZES:-10000 100000 1000000}"

if [[ ! -f "${VAQ_BINARY}" ]]; then
  echo "ERROR: VAQ binary not found: ${VAQ_BINARY}" >&2
  exit 1
fi
for f in "${DATASET_PATH}" "${TRAIN_PATH}" "${QUERY_PATH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}" >&2
    exit 1
  fi
done

build_method() {
  local M="$1"
  local NBITS="$2"
  local TOTAL=$((M * NBITS))
  if (( TOTAL < MIN_TOTAL_VAQ_BITS )); then
    TOTAL=${MIN_TOTAL_VAQ_BITS}
  fi
  echo "VAQ${TOTAL}m${M}min${MIN_BITS}max${MAX_BITS}var${VARIANCE},${SEARCH}"
}

# Require M | dim (96) for equal-length subspaces in standard PQ/OPQ setups
divides_dim() {
  local M="$1"
  (( DIM % M == 0 ))
}

echo "=== VAQ Deep hyperparameter grid ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "test: ${DATASET_PATH}"
echo "learn: ${TRAIN_PATH}"
echo "queries: ${QUERY_PATH}"
echo "dim=${DIM}  MIN_TOTAL_VAQ_BITS=${MIN_TOTAL_VAQ_BITS}  MIN_BITS=${MIN_BITS} MAX_BITS=${MAX_BITS}"
echo "M list: ${N_SUBQUANTIZERS_LIST}"
echo "nbits list: ${NBITS_LIST}"
echo "train sizes: ${TRAIN_SIZES}"
echo "PQ_GRID=${PQ_GRID}"
echo "SKIP_SEARCH=${SKIP_SEARCH}"
echo ""

RUNS=0
for M in ${N_SUBQUANTIZERS_LIST}; do
  if ! divides_dim "${M}"; then
    echo "SKIP M=${M} (does not divide dim=${DIM})"
    continue
  fi
  for NBITS in ${NBITS_LIST}; do
    METHOD="$(build_method "${M}" "${NBITS}")"
    for TRAIN_SIZE in ${TRAIN_SIZES}; do
      RUNS=$((RUNS + 1))
      echo ""
      echo "--------------------------------------------"
      echo "Run ${RUNS}:  M=${M}  nbits=${NBITS}  (PQ bits/vector=$((M*NBITS)))  train=${TRAIN_SIZE}"
      echo "  method=${METHOD}"
      echo "--------------------------------------------"

      CMD=(
        "${PYTHON}" -m vaq.eval
        --dataset_path "${DATASET_PATH}"
        --train_path "${TRAIN_PATH}"
        --query_path "${QUERY_PATH}"
        --dim "${DIM}"
        --dataset_name "${DATASET_NAME}"
        --data_root "${DATA_ROOT}"
        --method "${METHOD}"
        --train_size "${TRAIN_SIZE}"
        --sample_db "${SAMPLE_DB}"
        --sample_queries "${SAMPLE_QUERIES}"
        --results_dir "${RESULTS_SUBDIR}"
        --vaq_binary "${VAQ_BINARY}"
        --refine "${REFINE}"
        --k "${K}"
        --learn_ratio "${LEARN_RATIO}"
        --sample_mode first
      )
      if [[ "${SKIP_SEARCH}" == "1" ]]; then
        CMD+=(--skip_search)
      fi

      if [[ "${DRY_RUN}" == "1" ]]; then
        printf ' %q' "${CMD[@]}"
        echo ""
        continue
      fi

      "${CMD[@]}"
    done
  done
done

echo ""
echo "Finished ${RUNS} planned runs (some M may have been skipped)."
echo "Aggregate CSV: ${DATA_ROOT}/${RESULTS_SUBDIR}/${DATASET_NAME}_VAQ_adc_vs_exact_eval.csv"
