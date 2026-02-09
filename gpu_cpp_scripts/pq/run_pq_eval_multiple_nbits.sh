#!/usr/bin/env bash
# Wrapper script to run run_pq_eval_parallel_mult.sh sequentially for different datasets and nbits values
#
# Usage:
#   ./run_pq_eval_multiple_nbits.sh
#
# This script runs experiments in nested loops:
#   1. Outer loop: datasets (deep, msmarco, gist, openai, bigann)
#   2. Inner loop: nbits values [4, 6, 8, 10, 12]
#
# For each dataset, it automatically uses the appropriate M_VALUES from run_pq_eval_parallel_mult.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Array of datasets to run sequentially
# DATASETS=("deep" "msmarco" "gist" "openai" "bigann")
DATASETS=("msmarco")

# Array of nbits values to run sequentially
NBITS_VALUES=(10 12)

# Optional: Override GPU devices (default: use what's in run_pq_eval_parallel_mult.sh)
GPU_DEVICES="0"

# Optional: Override datasets or nbits via command line
# Usage: ./run_pq_eval_multiple_nbits.sh [dataset1 dataset2 ...] [--nbits 4 6 8]
TEMP_DATASETS=()
TEMP_NBITS=()
PARSE_NBITS=false

for arg in "$@"; do
    if [ "$arg" == "--nbits" ]; then
        PARSE_NBITS=true
    elif [ "$PARSE_NBITS" == true ]; then
        TEMP_NBITS+=("$arg")
    else
        TEMP_DATASETS+=("$arg")
    fi
done

if [ ${#TEMP_DATASETS[@]} -gt 0 ]; then
    DATASETS=("${TEMP_DATASETS[@]}")
fi

if [ ${#TEMP_NBITS[@]} -gt 0 ]; then
    NBITS_VALUES=("${TEMP_NBITS[@]}")
fi

echo "=========================================="
echo "Running PQ experiments for multiple datasets and nbits"
echo "=========================================="
echo "Datasets: ${DATASETS[*]}"
echo "nbits values: ${NBITS_VALUES[*]}"
echo "Running sequentially: datasets -> nbits"
echo "=========================================="
echo ""

# Track total time
START_TIME=$(date +%s)

# Outer loop: datasets
for DATASET_NAME in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo ">>> DATASET: ${DATASET_NAME}"
    echo "=========================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Set dataset-specific M_VALUES based on dataset name
    case "${DATASET_NAME}" in
        bigann)
            M_VALUES_ENV="1 2 4 8 16 32 64 128"
            ;;
        gist)
            M_VALUES_ENV="1 3 5 8 12 20 40 60 80 120 320 480 960"
            ;;
        msmarco)
            M_VALUES_ENV="1 2 4 8 16 32 64 128 256 512 1024"
            ;;
        openai)
            M_VALUES_ENV="1 4 8 16 32 64 128 256 512 768 1536"
            ;;
        deep)
            M_VALUES_ENV="1 2 3 4 6 8 12 16 24 32 48 96"
            ;;
        *)
            echo "Warning: Unknown dataset '${DATASET_NAME}', using default M_VALUES"
            M_VALUES_ENV="1 4 8 16 32 64 128"
            ;;
    esac
    
    echo "M_VALUES for ${DATASET_NAME}: ${M_VALUES_ENV}"
    echo ""
    
    # Inner loop: nbits values
    for NBITS in "${NBITS_VALUES[@]}"; do
        echo ""
        echo "  ----------------------------------------"
        echo "  >>> nbits=${NBITS} (dataset: ${DATASET_NAME})"
        echo "  ----------------------------------------"
        echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        # Export environment variables for run_pq_eval_parallel_mult.sh
        export DATASET_NAME
        export NBITS
        export M_VALUES_ENV
        
        # Optionally export GPU_DEVICES if set
        if [[ -n "${GPU_DEVICES:-}" ]]; then
            export GPU_DEVICES
        fi
        
        # Run the parallel script for this dataset and nbits value
        # It will run all M values in parallel, but we wait for it to complete
        # before moving to the next nbits value
        if ./run_pq_eval_parallel_mult.sh; then
            echo ""
            echo "  ✓ Completed nbits=${NBITS} for ${DATASET_NAME} successfully"
        else
            echo ""
            echo "  ✗ Failed for nbits=${NBITS} on ${DATASET_NAME}"
            echo "  Continuing with next nbits value..."
            # Continue with next nbits even if one fails
        fi
        
        echo ""
        echo "  Waiting a moment before starting next nbits..."
        sleep 2
    done
    
    echo ""
    echo "=========================================="
    echo "✓ Completed all nbits for dataset: ${DATASET_NAME}"
    echo "=========================================="
    echo ""
    echo "Waiting before starting next dataset..."
    sleep 5
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Completed datasets: ${DATASETS[*]}"
echo "Completed nbits values: ${NBITS_VALUES[*]}"
echo "=========================================="
