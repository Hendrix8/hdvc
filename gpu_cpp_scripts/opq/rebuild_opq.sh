#!/usr/bin/env bash
# Rebuild opq_eval (GPU version) – mirror PQ rebuild behaviour

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run make clean and then make. If either fails, the script exits with a non‑zero status.
make clean && make
