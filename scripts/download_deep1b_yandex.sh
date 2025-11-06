#!/bin/bash
# Script to download DEEP1B dataset from Yandex Research
# Downloads: 1B base set, 350M learning set, and 10K query set

TARGET_DIR="/data/cpanourg/2-hdvc/data/yandex"
BASE_URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP"

# Create target directory
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

echo "=========================================="
echo "Downloading DEEP1B Dataset from Yandex"
echo "=========================================="
echo "Target directory: ${TARGET_DIR}"
echo ""

# Function to download with retry
download_with_retry() {
    local url=$1
    local filename=$2
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        echo "Downloading ${filename}... (attempt $((retry + 1))/$max_retries)"
        if wget --continue --progress=bar:force "${url}" -O "${filename}" 2>&1 | tail -1; then
            echo "✅ Successfully downloaded ${filename}"
            return 0
        else
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo "⚠️  Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
    echo "❌ Failed to download ${filename} after ${max_retries} attempts"
    return 1
}

# Download base set (1 billion vectors) - single file
echo "--- Downloading Base Set (1B vectors) ---"
download_with_retry "${BASE_URL}/base.1B.fbin" "base.1B.fbin"

# Download learning set (350 million vectors) - single file
echo ""
echo "--- Downloading Learning Set (350M vectors) ---"
download_with_retry "${BASE_URL}/learn.350M.fbin" "learn.350M.fbin"

# Download query set (10K vectors) - single file
echo ""
echo "--- Downloading Query Set (10K vectors) ---"
download_with_retry "${BASE_URL}/query.public.10K.fbin" "query.public.10K.fbin"

# Summary
echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="
ls -lh *.u8bin 2>/dev/null | awk '{print $9, "(" $5 ")"}'
echo ""
echo "Files are located in: ${TARGET_DIR}"
echo ""

