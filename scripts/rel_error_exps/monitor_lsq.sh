#!/bin/bash
# Monitor LSQ++ job progress
# Usage: ./monitor_lsq.sh

PID=3537454
CSV_FILE="/data/cpanourg/2-hdvc/results/relerr/deep_LSQpp_adc_vs_exact_eval.csv"
EXPECTED_TOTAL=36

echo "=========================================="
echo "LSQ++ Job Monitor"
echo "=========================================="
echo ""

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Process is RUNNING (PID: $PID)"
    echo ""
    ps -p $PID -o etime,cmd --no-headers | awk '{print "Runtime: " $1 " | " substr($0, index($0,$2))}'
else
    echo "❌ Process is NOT running (may have completed or crashed)"
    exit 1
fi

echo ""
echo "----------------------------------------"
echo "Progress:"
echo "----------------------------------------"

# Count completed experiments
if [ -f "$CSV_FILE" ]; then
    COMPLETED=$(tail -n +2 "$CSV_FILE" | wc -l)
    REMAINING=$((EXPECTED_TOTAL - COMPLETED))
    PERCENT=$((COMPLETED * 100 / EXPECTED_TOTAL))
    
    echo "Completed: $COMPLETED / $EXPECTED_TOTAL ($PERCENT%)"
    echo "Remaining: $REMAINING"
    echo ""
    
    # Show last completed experiment
    echo "Last completed experiment:"
    tail -1 "$CSV_FILE" | awk -F',' '{printf "  subq=%s, nbits=%s, train=%s\n", $7, $8, $10}'
    
    echo ""
    echo "Current experiment (in progress):"
    echo "  subq=16, nbits=10, train=1000000"
else
    echo "CSV file not found yet"
fi

echo ""
echo "=========================================="
echo "To monitor in real-time, run:"
echo "  watch -n 5 ./monitor_lsq.sh"
echo "  or"
echo "  tail -f $CSV_FILE"
echo "=========================================="

