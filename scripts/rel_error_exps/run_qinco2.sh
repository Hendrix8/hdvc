#!/usr/bin/env bash
# Launch QINCo2 experiments over a grid of hyperparameters.
# Mirrors the structure used by other quantization run scripts.

# set -euo pipefail

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

if [ -z "${CONDA_PREFIX:-}" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # Fallback for systems where conda isn't on PATH yet
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "❌ Unable to locate conda. Please install or initialize conda before running this script."
    exit 1
  fi
fi

conda activate dtwrl_env2

###############################
# User configuration section #
###############################
DATA_ROOT="/data/cpanourg/2-hdvc"
PYTHON_SCRIPT="/home/cpanourg/projects/2-hdvc/scripts/rel_error_exps/QINCo2.py"
# DATASET_PATH="${DATA_ROOT}/data/deep1b/base.1B.fbin"
# QUERY_PATH="${DATA_ROOT}/data/deep1b/query.public.10K.fbin"
DATASET_PATH="${DATA_ROOT}/data/deep1b/dataset/deep1b-96-1m.bin"
QUERY_PATH="${DATA_ROOT}/data/deep1b/queries/queries-hard10p-deep1b-len96-1000.bin"
DATASET_NAME="deep1b"
DIM=96

# Output / temp directories
TEMP_DIR="${DATA_ROOT}/temp/qinco2"
MODEL_DIR="${DATA_ROOT}/results/qinco2/models"
RESULTS_SUBDIR="results/relerr"

# Hyperparameter grids
M_LIST=(8)
K_LIST=(256)
TRAIN_SIZES=(10000)
SAMPLE_DB_LIST=(10000)
SAMPLE_QR_LIST=(1000)

# Fixed QINCo2 defaults (override as needed)
L_LIST=(16)
DH_LIST=(384)
DE_LIST=(384)
A_LIST=(16)
B_LIST=(32)
IVF_K=1048576
EPOCHS=3
BATCH=1024
LR_LIST=(8e-4)
WD_LIST=(0.1)
GRAD_CLIP=0.1
VAL_RATIO=0.1
TEST_SIZE=1000000

USE_GPU_FLAG="--use_gpu"

###################################
# Iterate over experiment configs #
###################################

mkdir -p "${TEMP_DIR}" "${MODEL_DIR}"

for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do
  for M in "${M_LIST[@]}"; do
    for K in "${K_LIST[@]}"; do
      for SAMPLE_DB in "${SAMPLE_DB_LIST[@]}"; do
        for SAMPLE_QR in "${SAMPLE_QR_LIST[@]}"; do
          for L in "${L_LIST[@]}"; do
            for DH in "${DH_LIST[@]}"; do
              for DE in "${DE_LIST[@]}"; do
                for A in "${A_LIST[@]}"; do
                  for B in "${B_LIST[@]}"; do
                    for LR in "${LR_LIST[@]}"; do
                      for WD in "${WD_LIST[@]}"; do
                        echo ""
                        echo "============================================================"
                        echo "Running QINCo2 | M=${M}, K=${K}, train=${TRAIN_SIZE}, sample_db=${SAMPLE_DB}, sample_qr=${SAMPLE_QR}"
                        echo "============================================================"

                        python3 "${PYTHON_SCRIPT}" \
                          --dataset_path "${DATASET_PATH}" \
                          --query_path "${QUERY_PATH}" \
                          --dim "${DIM}" \
                          --dataset_name "${DATASET_NAME}" \
                          --data_root "${DATA_ROOT}" \
                          --results_dir "${RESULTS_SUBDIR}" \
                          --temp_dir "${TEMP_DIR}" \
                          --model_output_dir "${MODEL_DIR}" \
                          --train_size "${TRAIN_SIZE}" \
                          --val_ratio "${VAL_RATIO}" \
                          --test_size "${TEST_SIZE}" \
                          --sample_db "${SAMPLE_DB}" \
                          --sample_queries "${SAMPLE_QR}" \
                          --M "${M}" \
                          --K "${K}" \
                          --L "${L}" \
                          --dh "${DH}" \
                          --de "${DE}" \
                          --A "${A}" \
                          --B "${B}" \
                          --ivf_K "${IVF_K}" \
                          --epochs "${EPOCHS}" \
                          --batch "${BATCH}" \
                          --lr "${LR}" \
                          --weight_decay "${WD}" \
                          --grad_clip "${GRAD_CLIP}" \
                          --verbose \
                          ${USE_GPU_FLAG}

                        echo "✅ Completed QINCo2 (M=${M}, K=${K}, train=${TRAIN_SIZE})"
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo ""
echo "All QINCo2 experiments finished."

