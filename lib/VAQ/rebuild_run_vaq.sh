#!/usr/bin/env bash
# Reconfigure and build run_vaq (needed after pull for --skip-query / vaq.eval --skip_search).
# Run from repo root or this directory; activate conda first if you use conda OpenBLAS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${ROOT}/lib/VAQ/build"
mkdir -p "${BUILD}"
cd "${BUILD}"
# Drop stale BLAS paths if libraries moved between conda envs
if [[ -f CMakeCache.txt ]]; then
  cmake "${ROOT}/lib/VAQ" 2>/dev/null || true
fi
cmake "${ROOT}/lib/VAQ"
cmake --build . --target run_vaq -j "$(nproc 2>/dev/null || echo 8)"
echo "OK: ${BUILD}/examples/run_vaq"
strings "${BUILD}/examples/run_vaq" | grep -F skip-query >/dev/null && echo "Binary includes skip-query flag." || echo "WARNING: skip-query string not found in binary."
