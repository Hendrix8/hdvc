#!/usr/bin/env bash
# Extended-RaBitQ BIGANN runner wrapper.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
"${PYTHON}" -m rabitq.eval "$@"
