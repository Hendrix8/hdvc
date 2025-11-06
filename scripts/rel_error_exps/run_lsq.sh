#!/bin/bash
# ==========================================================
#  Multi-experiment launcher for LSQ++ — dynamically calls LSQpp.py
# ==========================================================

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
conda activate dtwrl_env2

# ---- Paths ----
DATA_FP="/data/cpanourg/2-hdvc"
DATASET_PATH="${DATA_FP}/data/deep1b/dataset/deep1b-96-100m.bin"
QUERY_PATH="${DATA_FP}/data/deep1b/queries/queries-hard10p-deep1b-len96-1000.bin"
DATASET_NAME="deep"
DIM=96
DATA_ROOT="${DATA_FP}"
RESULTS_DIR="${DATA_FP}/results/relerr"

# ---- Hyperparameter grids ----
METHODS=("LSQ++")
N_SUBQUANTIZERS_LIST=(4 8 16 32)
NBITS_LIST=(8 9 10)
TRAIN_SIZES=(10000 100000 1000000)
SAMPLE_DB=10000
SAMPLE_QUERIES=1000

# ==========================================================
# Run experiments
# ==========================================================
PY_FILE="LSQpp.py"

if [ ! -f "$PY_FILE" ]; then
  echo "❌ Error: ${PY_FILE} not found!"
  exit 1
fi

for N_SUBQ in "${N_SUBQUANTIZERS_LIST[@]}"; do
  for NBITS in "${NBITS_LIST[@]}"; do
    for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do

      echo ""
      echo "--------------------------------------------"
      echo "Running LSQ++ | subq=${N_SUBQ}, nbits=${NBITS}, train=${TRAIN_SIZE}"
      echo "--------------------------------------------"

      python3 "${PY_FILE}" \
        --dataset_path "${DATASET_PATH}" \
        --query_path "${QUERY_PATH}" \
        --dim ${DIM} \
        --dataset_name "${DATASET_NAME}" \
        --data_root "${DATA_ROOT}" \
        --n_subquantizers ${N_SUBQ} \
        --nbits ${NBITS} \
        --train_size ${TRAIN_SIZE} \
        --sample_db ${SAMPLE_DB} \
        --sample_queries ${SAMPLE_QUERIES} \
        --results_dir "${RESULTS_DIR}"

      echo "✅ Completed LSQ++ (${N_SUBQ}x${NBITS}, train=${TRAIN_SIZE})"
    done
  done
done

echo "============================================"
echo "All LSQ++ experiments completed!"
echo "Results at: ${DATA_ROOT}/${RESULTS_DIR}/${DATASET_NAME}_adc_vs_exact_eval.csv"
echo "============================================"

