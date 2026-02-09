#!/usr/bin/env bash
# Run ADC timing experiments on CSV file
# This script runs ADC timing measurements for each row in the CSV
# and adds adc_pp and distance_table_time_pp columns

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH=/home/cpanourg/projects/2-hdvc/local/openblas/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:$LD_LIBRARY_PATH

DATA_ROOT="/data/cpanourg/2-hdvc"
RESULTS_DIR="/data/cpanourg/2-hdvc/results/relerr_cpp"

# Default parameters
N_SAMPLE_Q=1000
N_SAMPLE_DB=10000

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <csv_file> [n_sample_q] [n_sample_db]"
    echo ""
    echo "Arguments:"
    echo "  csv_file    - Path to CSV file (e.g., deep_PQ_adc_vs_exact_eval.csv)"
    echo "  n_sample_q  - Number of queries to use (default: 1000)"
    echo "  n_sample_db - Number of database vectors to use (default: 10000)"
    echo ""
    echo "Example:"
    echo "  $0 \${RESULTS_DIR}/deep_PQ_adc_vs_exact_eval.csv"
    echo "  $0 \${RESULTS_DIR}/gist_PQ_adc_vs_exact_eval.csv 1000 10000"
    exit 1
fi

CSV_FILE="$1"
if [ $# -ge 2 ]; then
    N_SAMPLE_Q="$2"
fi
if [ $# -ge 3 ]; then
    N_SAMPLE_DB="$3"
fi

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

# Check if executable exists
if [ ! -f "./pq_adc_timing" ]; then
    echo "Error: pq_adc_timing executable not found. Building..."
    make pq_adc_timing
    if [ ! -f "./pq_adc_timing" ]; then
        echo "Error: Failed to build pq_adc_timing"
        exit 1
    fi
fi

echo "=========================================="
echo "ADC Timing Experiment"
echo "=========================================="
echo "CSV file: $CSV_FILE"
echo "Sample queries: $N_SAMPLE_Q"
echo "Sample database: $N_SAMPLE_DB"
echo "Data root: $DATA_ROOT"
echo "=========================================="
echo ""

# Set process priority to reduce interference
# Use nice to lower priority (less likely to be preempted by system processes)
# Use chrt to set real-time priority (requires root or appropriate permissions)
# For now, we'll use nice which doesn't require root
echo "Setting process priority..."
nice -n -10 ./pq_adc_timing "$CSV_FILE" "$DATA_ROOT" "$N_SAMPLE_Q" "$N_SAMPLE_DB"

echo ""
echo "=========================================="
echo "Done! Results saved to: $CSV_FILE"
echo "=========================================="
