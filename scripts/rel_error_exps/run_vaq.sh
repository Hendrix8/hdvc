#!/bin/bash
# ==========================================================
#  Multi-experiment launcher for VAQ — dynamically calls VAQ.py
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
VAQ_BINARY="/home/cpanourg/projects/2-hdvc/lib/VAQ/build/examples/run_vaq"

# ---- Hyperparameter grids ----
# VAQ method string format: VAQ{totalBits}m{subspaces}min{minBits}max{maxBits}var{variance},{searchMethod}
# Examples:
#   - VAQ256m32min7max8var1,HEAP (256 bits, 32 subspaces, 7-8 bits per subspace, variance 1, HEAP search)
#   - VAQ128m16min6max7var1,EA (128 bits, 16 subspaces, 6-7 bits per subspace, variance 1, EA search)
#
# You can either specify full METHOD strings, OR use individual parameters (total_bits, n_subspaces, etc.)
# If you specify both, individual parameters will override the method string

METHODS=(
    "VAQ256m32min7max8var1,HEAP"
    "VAQ128m16min6max7var1,HEAP"
    # Add more method strings as needed
)

# Alternative: Use individual parameters instead of full method strings
# Leave empty to use METHODS above, or specify to override/construct methods
TOTAL_BITS_LIST=(32 64 128 256)  # e.g., (128 256 512)
N_SUBSPACES_LIST=(8 16 32)  # e.g., (16 32)
MIN_BITS_LIST=(4 7)  # e.g., (6 7)
MAX_BITS_LIST=(8 12)  # e.g., (7 8)
VARIANCE_LIST=(1.0)  # e.g., (0.95 1.0)
SEARCH_METHOD="HEAP"  # Default search method when using individual parameters


TRAIN_SIZES=(10000 100000 1000000)
REFINE_LIST=("100,200")
K_LIST=(100)
LEARN_RATIOS=(0.05)
SAMPLE_DB=10000
SAMPLE_QUERIES=1000

# ==========================================================
# Run experiments
# ==========================================================
PY_FILE="VAQ.py"

if [ ! -f "$PY_FILE" ]; then
  echo "❌ Error: ${PY_FILE} not found!"
  exit 1
fi

# Check if using individual parameters or method strings
if [ ${#TOTAL_BITS_LIST[@]} -gt 0 ] && [ ${#N_SUBSPACES_LIST[@]} -gt 0 ] && \
   [ ${#MIN_BITS_LIST[@]} -gt 0 ] && [ ${#MAX_BITS_LIST[@]} -gt 0 ]; then
  # Use individual parameters
  echo "Using individual parameter grids..."
  for TOTAL_BITS in "${TOTAL_BITS_LIST[@]}"; do
    for N_SUBSPACES in "${N_SUBSPACES_LIST[@]}"; do
      for MIN_BITS in "${MIN_BITS_LIST[@]}"; do
        for MAX_BITS in "${MAX_BITS_LIST[@]}"; do
          VAR_TO_USE=${VARIANCE_LIST[0]:-1.0}
          if [ ${#VARIANCE_LIST[@]} -gt 0 ]; then
            for VAR in "${VARIANCE_LIST[@]}"; do
              VAR_TO_USE=$VAR
              for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do
                for REFINE in "${REFINE_LIST[@]}"; do
                  for K in "${K_LIST[@]}"; do
                    for LEARN_RATIO in "${LEARN_RATIOS[@]}"; do
                      echo ""
                      echo "--------------------------------------------"
                      echo "Running VAQ | bits=${TOTAL_BITS}, subspaces=${N_SUBSPACES}, min=${MIN_BITS}, max=${MAX_BITS}, var=${VAR_TO_USE}"
                      echo "  train=${TRAIN_SIZE}, refine=${REFINE}, k=${K}, learn_ratio=${LEARN_RATIO}"
                      echo "--------------------------------------------"

                      python3 "${PY_FILE}" \
                        --dataset_path "${DATASET_PATH}" \
                        --query_path "${QUERY_PATH}" \
                        --dim ${DIM} \
                        --dataset_name "${DATASET_NAME}" \
                        --data_root "${DATA_ROOT}" \
                        --total_bits ${TOTAL_BITS} \
                        --n_subspaces ${N_SUBSPACES} \
                        --min_bits ${MIN_BITS} \
                        --max_bits ${MAX_BITS} \
                        --variance ${VAR_TO_USE} \
                        --train_size ${TRAIN_SIZE} \
                        --refine "${REFINE}" \
                        --k ${K} \
                        --learn_ratio ${LEARN_RATIO} \
                        --sample_db ${SAMPLE_DB} \
                        --sample_queries ${SAMPLE_QUERIES} \
                        --results_dir "${RESULTS_DIR}" \
                        --vaq_binary "${VAQ_BINARY}"

                      echo "✅ Completed VAQ (bits=${TOTAL_BITS}, subspaces=${N_SUBSPACES}, min=${MIN_BITS}, max=${MAX_BITS}, var=${VAR_TO_USE})"
                    done
                  done
                done
              done
            done
          else
            for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do
              for REFINE in "${REFINE_LIST[@]}"; do
                for K in "${K_LIST[@]}"; do
                  for LEARN_RATIO in "${LEARN_RATIOS[@]}"; do
                    echo ""
                    echo "--------------------------------------------"
                    echo "Running VAQ | bits=${TOTAL_BITS}, subspaces=${N_SUBSPACES}, min=${MIN_BITS}, max=${MAX_BITS}, var=${VAR_TO_USE}"
                    echo "  train=${TRAIN_SIZE}, refine=${REFINE}, k=${K}, learn_ratio=${LEARN_RATIO}"
                    echo "--------------------------------------------"

                    python3 "${PY_FILE}" \
                      --dataset_path "${DATASET_PATH}" \
                      --query_path "${QUERY_PATH}" \
                      --dim ${DIM} \
                      --dataset_name "${DATASET_NAME}" \
                      --data_root "${DATA_ROOT}" \
                      --total_bits ${TOTAL_BITS} \
                      --n_subspaces ${N_SUBSPACES} \
                      --min_bits ${MIN_BITS} \
                      --max_bits ${MAX_BITS} \
                      --variance ${VAR_TO_USE} \
                      --train_size ${TRAIN_SIZE} \
                      --refine "${REFINE}" \
                      --k ${K} \
                      --learn_ratio ${LEARN_RATIO} \
                      --sample_db ${SAMPLE_DB} \
                      --sample_queries ${SAMPLE_QUERIES} \
                      --results_dir "${RESULTS_DIR}" \
                      --vaq_binary "${VAQ_BINARY}"

                    echo "✅ Completed VAQ (bits=${TOTAL_BITS}, subspaces=${N_SUBSPACES}, min=${MIN_BITS}, max=${MAX_BITS}, var=${VAR_TO_USE})"
                  done
                done
              done
            done
          fi
        done
      done
    done
  done
else
  # Use method strings
  echo "Using method string grids..."
  for METHOD in "${METHODS[@]}"; do
    for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do
      for REFINE in "${REFINE_LIST[@]}"; do
        for K in "${K_LIST[@]}"; do
          for LEARN_RATIO in "${LEARN_RATIOS[@]}"; do

            echo ""
            echo "--------------------------------------------"
            echo "Running VAQ | method=${METHOD}, train=${TRAIN_SIZE}, refine=${REFINE}, k=${K}, learn_ratio=${LEARN_RATIO}"
            echo "--------------------------------------------"

            python3 "${PY_FILE}" \
              --dataset_path "${DATASET_PATH}" \
              --query_path "${QUERY_PATH}" \
              --dim ${DIM} \
              --dataset_name "${DATASET_NAME}" \
              --data_root "${DATA_ROOT}" \
              --method "${METHOD}" \
              --train_size ${TRAIN_SIZE} \
              --refine "${REFINE}" \
              --k ${K} \
              --learn_ratio ${LEARN_RATIO} \
              --sample_db ${SAMPLE_DB} \
              --sample_queries ${SAMPLE_QUERIES} \
              --results_dir "${RESULTS_DIR}" \
              --vaq_binary "${VAQ_BINARY}"

            echo "✅ Completed VAQ (method=${METHOD}, train=${TRAIN_SIZE}, refine=${REFINE}, k=${K}, learn_ratio=${LEARN_RATIO})"
          done
        done
      done
    done
  done
fi

echo "============================================"
echo "All VAQ experiments completed!"
echo "Results at: ${DATA_ROOT}/${RESULTS_DIR}/${DATASET_NAME}_adc_vs_exact_eval.csv"
echo "============================================"

