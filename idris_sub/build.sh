#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${ROOT}/project_description.pdf"
TMPDIR="${ROOT}/project_description/build"

mkdir -p "${TMPDIR}"
cd "${ROOT}/project_description"

pdflatex -interaction=nonstopmode -halt-on-error -output-directory="${TMPDIR}" main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="${TMPDIR}" main.tex

cp "${TMPDIR}/main.pdf" "${OUT}"
echo "Wrote ${OUT}"
