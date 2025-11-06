#!/bin/bash
# ==========================================================
#  Temporary script to run remaining LSQ++ experiments
#  Only runs experiments with train_size = 1M
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

# ---- Check what experiments are missing (train_size = 1M only) ----
PY_FILE="LSQpp.py"
RESULTS_CSV="${RESULTS_DIR}/${DATASET_NAME}_LSQpp_adc_vs_exact_eval.csv"

if [ ! -f "$PY_FILE" ]; then
  echo "❌ Error: ${PY_FILE} not found!"
  exit 1
fi

# Check what's missing
echo "Checking for missing experiments (train_size = 1M)..."
MISSING_EXPS=$(python3 << PYEOF
import pandas as pd
import sys
import os

results_csv = "${RESULTS_CSV}"

if not os.path.exists(results_csv):
    print("CSV not found, all experiments need to run")
    sys.exit(1)

df = pd.read_csv(results_csv)

# All possible combinations
n_subq_list = [4, 8, 16, 32]
nbits_list = [8, 9, 10]
train_size_1m = [1000000]  # Only 1M

# Get completed combinations
completed = set()
for _, row in df.iterrows():
    completed.add((int(row['n_subquantizers']), int(row['nbits']), int(row['train_size'])))

# Find missing combinations (only 1M)
missing = []
for n_subq in n_subq_list:
    for nbits in nbits_list:
        for train_size in train_size_1m:
            if (n_subq, nbits, train_size) not in completed:
                missing.append((n_subq, nbits, train_size))

if not missing:
    print("NONE")
    sys.exit(0)

# Print missing experiments in format: subq,nbits,train_size
for n_subq, nbits, train_size in sorted(missing):
    print(f"{n_subq},{nbits},{train_size}")
PYEOF
)

if [ "$MISSING_EXPS" = "NONE" ]; then
  echo "✅ All experiments with train_size = 1M are already completed!"
  echo "No experiments to run."
  exit 0
fi

# Parse missing experiments
SAMPLE_DB=10000
SAMPLE_QUERIES=1000

echo "============================================"
echo "Running remaining LSQ++ experiments"
echo "Only train_size = 1M"
echo "============================================"

# Count missing experiments
MISSING_COUNT=$(echo "$MISSING_EXPS" | wc -l)
echo "Total missing: ${MISSING_COUNT} experiments"
echo ""

# Run each missing experiment
echo "$MISSING_EXPS" | while IFS=',' read -r N_SUBQ NBITS TRAIN_SIZE; do
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

  if [ $? -eq 0 ]; then
    echo "✅ Completed LSQ++ (${N_SUBQ}x${NBITS}, train=${TRAIN_SIZE})"
  else
    echo "❌ Failed LSQ++ (${N_SUBQ}x${NBITS}, train=${TRAIN_SIZE})"
  fi
done

echo ""
echo "============================================"
echo "Remaining experiments (train_size = 1M) completed!"
echo "Results at: ${DATA_ROOT}/${RESULTS_DIR}/${DATASET_NAME}_LSQpp_adc_vs_exact_eval.csv"
echo "============================================"

