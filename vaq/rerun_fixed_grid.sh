#!/usr/bin/env bash
# Rerun VAQ grid experiments after the XTrainSlice uninitialized-buffer fix.
# Results go to a new experiment folder so corrupt old runs are preserved separately.
#
# Launch:
#   nohup bash vaq/rerun_fixed_grid.sh > /data/cpanourg/2-hdvc/results/vaq/fixed_grid_nohup.out 2>&1 &
#   echo "PID: $!"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}"}/.." && pwd)"
cd "${REPO_ROOT}"

export EXPERIMENT_FOLDER="vaq_bpv_grid_FIXED_$(date +%Y%m%d_%H%M%S)"
export DATASETS="bigann gist openai deep msmarco"
export MIN_BITS_LIST="2 4 6"
export MAX_BITS_LIST="8 16"
export BPV_MULTIPLIERS="4 8 12"
export VARIANCE="1.0"
export SKIP_SEARCH="1"
export TRAIN_SIZE="100000"
export SAMPLE_DB="10000"
export SAMPLE_QUERIES="1000"
export MAX_DATASET_JOBS="5"
export OMP_NUM_THREADS="16"
export MKL_NUM_THREADS="16"

echo "Experiment folder: ${EXPERIMENT_FOLDER}"
echo "Launching grid..."

nohup bash vaq/run_all_datasets_bpv_grid_nohup.sh \
  > "/data/cpanourg/2-hdvc/results/vaq/${EXPERIMENT_FOLDER}_inner.out" 2>&1
