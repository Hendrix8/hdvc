#!/bin/bash
# ==========================================================
#  Multi-experiment launcher for RaBitQ — dynamically calls RaBitQ.py
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
METHODS=("RaBitQ")
BITS_PER_QUERY_DIM_LIST=(1 2 4 6 8 10)  # Bits per query dimension
TRAIN_SIZES=(10000 100000 1000000)
SEED=123  # Random seed for rotation matrix
SAMPLE_DB=10000
SAMPLE_QUERIES=1000

# ==========================================================
# Run experiments
# ==========================================================
PY_FILE="RaBitQ.py"

if [ ! -f "$PY_FILE" ]; then
  echo "❌ Error: ${PY_FILE} not found!"
  exit 1
fi

for BITS_PER_DIM in "${BITS_PER_QUERY_DIM_LIST[@]}"; do
  for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do

    echo ""
    echo "--------------------------------------------"
    echo "Running RaBitQ | bits_per_query_dim=${BITS_PER_DIM}, train=${TRAIN_SIZE}"
    echo "--------------------------------------------"

    python3 "${PY_FILE}" \
      --dataset_path "${DATASET_PATH}" \
      --query_path "${QUERY_PATH}" \
      --dim ${DIM} \
      --dataset_name "${DATASET_NAME}" \
      --data_root "${DATA_ROOT}" \
      --bits_per_query_dim ${BITS_PER_DIM} \
      --train_size ${TRAIN_SIZE} \
      --seed ${SEED} \
      --sample_db ${SAMPLE_DB} \
      --sample_queries ${SAMPLE_QUERIES} \
      --results_dir "${RESULTS_DIR}"

    echo "✅ Completed RaBitQ (bits_per_query_dim=${BITS_PER_DIM}, train=${TRAIN_SIZE})"
  done
done

echo "============================================"
echo "All RaBitQ experiments completed!"
echo "Results at: ${DATA_ROOT}/${RESULTS_DIR}/${DATASET_NAME}_adc_vs_exact_eval.csv"
echo "============================================"

