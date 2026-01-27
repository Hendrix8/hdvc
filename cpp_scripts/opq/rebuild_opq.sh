#!/usr/bin/env bash
# Rebuild script for opq_eval binary

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Cleaning previous build..."
make clean

echo "Building opq_eval..."
make

echo "✓ opq_eval rebuilt successfully"
