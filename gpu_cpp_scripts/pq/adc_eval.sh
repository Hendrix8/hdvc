#!/usr/bin/env bash
# ADC Timing Evaluation Script
# Runs ADC timing experiments on CSV files and adds adc_pp and distance_table_time_pp columns


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH=/home/cpanourg/projects/2-hdvc/local/openblas/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:$LD_LIBRARY_PATH

DATA_ROOT="/data/cpanourg/2-hdvc"
RESULTS_DIR="/data/cpanourg/2-hdvc/results/relerr_cpp"

# ============================================================================
# CONFIGURATION: Edit these defaults to match your needs
# ============================================================================
# Default CSV file path (edit this to your CSV file)
CSV_FILE="${RESULTS_DIR}/deep_PQ_adc_vs_exact_eval.csv"

# Default parameters
N_SAMPLE_Q=1000
N_SAMPLE_DB=10000
# ============================================================================

# Parse arguments (CSV file can be overridden via command line)
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then
    # First argument is CSV file (if it doesn't start with --)
    CSV_FILE="$1"
    shift
fi

# Show usage if help requested
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [csv_file] [options]"
    echo ""
    echo "Arguments:"
    echo "  csv_file          - Path to CSV file (optional, default: see script)"
    echo ""
    echo "Options:"
    echo "  --n_sample_q N    - Number of queries to use (default: 1000)"
    echo "  --n_sample_db N   - Number of database vectors to use (default: 10000)"
    echo "  --data_root PATH  - Data root directory (default: /data/cpanourg/2-hdvc)"
    echo "  --help            - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Uses default CSV from script"
    echo "  $0 \${RESULTS_DIR}/gist_PQ_adc_vs_exact_eval.csv"
    echo "  $0 --n_sample_q 500 --n_sample_db 5000"
    echo "  $0 \${RESULTS_DIR}/bigann_PQ_adc_vs_exact_eval.csv --data_root /custom/path"
    echo ""
    echo "Current default CSV: $CSV_FILE"
    exit 0
fi

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_sample_q)
            N_SAMPLE_Q="$2"
            shift 2
            ;;
        --n_sample_db)
            N_SAMPLE_DB="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 <csv_file> [options]"
            echo ""
            echo "Options:"
            echo "  --n_sample_q N    - Number of queries to use (default: 1000)"
            echo "  --n_sample_db N   - Number of database vectors to use (default: 10000)"
            echo "  --data_root PATH  - Data root directory (default: /data/cpanourg/2-hdvc)"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

# Check if executable exists, build if needed
if [ ! -f "./pq_adc_timing" ]; then
    echo "Building pq_adc_timing executable..."
    make pq_adc_timing
    if [ ! -f "./pq_adc_timing" ]; then
        echo "Error: Failed to build pq_adc_timing"
        echo "Please check the Makefile and ensure all dependencies are installed"
        exit 1
    fi
    echo "✓ Build successful"
fi

echo "=========================================="
echo "ADC Timing Evaluation"
echo "=========================================="
echo "CSV file:        $CSV_FILE"
echo "Sample queries:  $N_SAMPLE_Q"
echo "Sample database: $N_SAMPLE_DB"
echo "Data root:       $DATA_ROOT"
echo "=========================================="
echo ""
echo "Tip: Edit CSV_FILE in the script to change default CSV"
echo ""

# Set process priority to reduce interference
# Use nice to give higher priority (less likely to be preempted)
# Note: nice -n -10 requires appropriate permissions
echo "Setting process priority (nice -n -10)..."
if nice -n -10 ./pq_adc_timing "$CSV_FILE" "$DATA_ROOT" "$N_SAMPLE_Q" "$N_SAMPLE_DB"; then
    echo ""
    echo "=========================================="
    echo "✓ Success! Results saved to: $CSV_FILE"
    echo "=========================================="
    echo ""
    echo "New columns added:"
    echo "  - adc_pp: ADC time per pair (seconds)"
    echo "  - distance_table_time_pp: Distance table time per pair (seconds)"
else
    echo ""
    echo "=========================================="
    echo "✗ Error occurred during execution"
    echo "=========================================="
    exit 1
fi
