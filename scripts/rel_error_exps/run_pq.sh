#!/bin/bash
# ==========================================================
#  Multi-experiment launcher — dynamically calls ${METHOD}.py
# ==========================================================

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
conda activate dtwrl_env2

# ---- Paths ----
DATA_FP="/mnthdd/cpanourg/2-hdvc"
# Three separate files required: database (db), training (train_db), and queries (qr)
DATASET_PATH="${DATA_FP}/data/deep1b/dataset/test_1m.bin"  # Database/test data dim = 96
TRAIN_PATH="${DATA_FP}/data/deep1b/dataset/learn_100m.bin"  # Training data (required)
QUERY_PATH="${DATA_FP}/data/deep1b/dataset/query_10k.bin"  # Query data (required)

# DATASET_PATH="${DATA_FP}/data/gist/gist_base.fvecs"  # Database/test data dim = 960
# TRAIN_PATH="${DATA_FP}/data/gist/gist_learn.fvecs"  # Training data (required)
# QUERY_PATH="${DATA_FP}/data/gist/gist_query.fvecs"  # Query data (required)

# DATASET_PATH="${DATA_FP}/data/glove/splits/test.bin"  # Database/test data dim=200
# TRAIN_PATH="${DATA_FP}/data/glove/splits/train.bin"  # Training data (required)
# QUERY_PATH="${DATA_FP}/data/glove/splits/queries.bin"  # Query data (required)


DATASET_NAME="deep"
DIM=96
DATA_ROOT="${DATA_FP}"
RESULTS_DIR="${DATA_FP}/results/relerr"

# ---- Hyperparameter grids ----
METHODS=("PQ")
# archived hp test : 
N_SUBQUANTIZERS_LIST=(1)
NBITS_LIST=(10)
TRAIN_SIZES=(1000000)

#hp test : 
# N_SUBQUANTIZERS_LIST=(1 2 3 4 5 6 8 10 12 15 16 20 24 30 32 40 48 60 64 80 96 120 160 192 240 320 480 960)
# NBITS_LIST=(10)
# TRAIN_SIZES=(100000000)

SAMPLE_DB=10000
SAMPLE_QUERIES=1000

# ---- Model loading options (optional) ----
# Set LOAD_MODEL=true to load pre-trained models instead of training
LOAD_MODEL=false
# MODEL_PATH: path to the saved PQ model file
#   - Can be a single path used for all runs: "/path/to/pq_model.index"
#   - Can use variables that will be expanded per configuration:
#     "${DATA_ROOT}/results/relerr/pq/${DATASET_NAME}/subq${N_SUBQ}_nbits${NBITS}_train${TRAIN_SIZE}_*/pq_model.index"
#   - Wildcards (*) are supported and will match the first found file
#   - Leave empty to train new models
MODEL_PATH=""

# ==========================================================
# Run experiments
# ==========================================================
for METHOD in "${METHODS[@]}"; do
  PY_FILE="${METHOD}.py"   # Dynamically form the Python filename

  if [ ! -f "$PY_FILE" ]; then
    echo "❌ Error: ${PY_FILE} not found!"
    continue
  fi

  for N_SUBQ in "${N_SUBQUANTIZERS_LIST[@]}"; do
    for NBITS in "${NBITS_LIST[@]}"; do
      for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do

        echo ""
        echo "--------------------------------------------"
        if [ "$LOAD_MODEL" = true ]; then
          echo "Running ${METHOD} | subq=${N_SUBQ}, nbits=${NBITS}, train=${TRAIN_SIZE} [LOADING MODEL]"
        else
          echo "Running ${METHOD} | subq=${N_SUBQ}, nbits=${NBITS}, train=${TRAIN_SIZE} [TRAINING]"
        fi
        echo "--------------------------------------------"

        # Build command arguments
        CMD_ARGS=(
          --dataset_path "${DATASET_PATH}"
          --train_path "${TRAIN_PATH}"
          --query_path "${QUERY_PATH}"
          --dim ${DIM}
          --dataset_name "${DATASET_NAME}"
          --data_root "${DATA_ROOT}"
          --n_subquantizers ${N_SUBQ}
          --nbits ${NBITS}
          --train_size ${TRAIN_SIZE}
          --sample_db ${SAMPLE_DB}
          --sample_queries ${SAMPLE_QUERIES}
          --results_dir "${RESULTS_DIR}"
        )

        # Add model loading arguments if enabled
        if [ "$LOAD_MODEL" = true ]; then
          if [ -n "$MODEL_PATH" ]; then
            # If MODEL_PATH contains variables, expand them
            EXPANDED_MODEL_PATH=$(eval echo "${MODEL_PATH}")
            # If path contains wildcards, try to find the first match
            if [[ "$EXPANDED_MODEL_PATH" == *"*"* ]]; then
              # Use find or ls to get the first matching file
              FOUND_MODEL=$(ls -1 ${EXPANDED_MODEL_PATH} 2>/dev/null | head -n 1)
              if [ -n "$FOUND_MODEL" ]; then
                CMD_ARGS+=(--load_model --model_path "${FOUND_MODEL}")
              else
                echo "⚠️  Warning: No model found matching pattern: ${EXPANDED_MODEL_PATH}"
                echo "   Skipping model loading for this configuration"
              fi
            else
              CMD_ARGS+=(--load_model --model_path "${EXPANDED_MODEL_PATH}")
            fi
          else
            echo "⚠️  Warning: LOAD_MODEL=true but MODEL_PATH is empty. Skipping model loading."
          fi
        fi

        python3 "${PY_FILE}" "${CMD_ARGS[@]}"

        echo "✅ Completed ${METHOD} (${N_SUBQ}x${NBITS}, train=${TRAIN_SIZE})"
      done
    done
  done
done

echo "============================================"
echo "All ${#METHODS[@]} method experiments completed!"
echo "Results at: ${DATA_ROOT}/${RESULTS_DIR}/${DATASET_NAME}_adc_vs_exact_eval.csv"
echo "============================================"
