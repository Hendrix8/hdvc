#!/bin/bash
# Script to download and extract DEEP1B files from Sync.com
# Target directory: /data/cpanourg/2-hdvc/data/deep1b/all_data

TARGET_DIR="/data/cpanourg/2-hdvc/data/deep1b/all_data"
SYNC_URL="https://ln5.sync.com/dl/0b8135230/39vxx8su-tkfi7t2s-dgsvh8rp-k8ixcs8p?sync_id=13317693170004"

echo "=========================================="
echo "DEEP1B Files Download Script"
echo "=========================================="
echo ""
echo "Target directory: ${TARGET_DIR}"
echo ""

# Create target directory if it doesn't exist
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

echo "Method 1: If you have downloaded the ZIP file manually, place it here and run:"
echo "  unzip -q <zip_filename> -d ${TARGET_DIR}"
echo ""
echo "Method 2: Try downloading with wget (may require authentication):"
echo "  wget --no-check-certificate --content-disposition \"${SYNC_URL}\" -O deep1b_files.zip"
echo ""
echo "Method 3: If you have Sync.com credentials, you can:"
echo "  1. Install rclone: sudo apt-get install rclone"
echo "  2. Configure Sync.com: rclone config"
echo "  3. Download: rclone copy sync:folder/path ${TARGET_DIR}"
echo ""
echo "Method 4: Use browser to download, then extract:"
echo "  1. Open: ${SYNC_URL}"
echo "  2. Download the folder as ZIP"
echo "  3. Extract to: ${TARGET_DIR}"
echo ""

# Check if there's already a ZIP file in the directory
if ls *.zip 1> /dev/null 2>&1; then
    echo "Found ZIP file(s) in directory:"
    ls -lh *.zip
    echo ""
    read -p "Do you want to extract the ZIP file(s)? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for zipfile in *.zip; do
            echo "Extracting ${zipfile}..."
            unzip -q "${zipfile}" -d "${TARGET_DIR}"
            echo "Done extracting ${zipfile}"
        done
        echo ""
        echo "Files extracted to: ${TARGET_DIR}"
        ls -lh "${TARGET_DIR}" | head -20
    fi
else
    echo "No ZIP files found in ${TARGET_DIR}"
    echo "Please download the files first using one of the methods above."
fi

