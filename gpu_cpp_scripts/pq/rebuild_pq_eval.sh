#!/usr/bin/env bash
# Rebuild pq_eval (GPU version)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
make clean && make
