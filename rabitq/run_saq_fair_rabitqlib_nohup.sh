#!/usr/bin/env bash
# Run RaBitQ-Library using the exact SAQ-prepared base/query vectors and IVF
# centroid/cluster-id files. This is the fair SAQ comparison path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CLUSTERS="${CLUSTERS:-4096}"
BITS="${BITS:-1 2 4 8}"
TOPK="${TOPK:-100}"
NPROBE="${NPROBE:-200}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OMP_NUM_THREADS
SKIP_EXISTING="${SKIP_EXISTING:-1}"

OUT_ROOT="${OUT_ROOT:-/data/cpanourg/2-hdvc/results/rabitq_saqfair}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/rabitq_saqfair_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

run_one() {
  local dataset="$1"
  local prepared_dir="${ROOT}/lib/SAQ/data/${dataset}"
  local base="${prepared_dir}/${dataset}_base.fvecs"
  local query="${prepared_dir}/${dataset}_query.fvecs"
  local out_csv="${OUT_ROOT}/${dataset}_RaBitQ_adc_vs_exact_eval.csv"
  local log="${LOG_DIR}/${dataset}.log"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${out_csv}" ]]; then
    echo "Skipping ${dataset}: existing ${out_csv}" | tee -a "${LOG_DIR}/skipped.log"
    return 0
  fi

  {
    echo "[$(date --iso-8601=seconds)] START ${dataset}"
    echo "prepared_dir=${prepared_dir}"
    echo "clusters=${CLUSTERS}; bits=${BITS}; topk=${TOPK}; nprobe=${NPROBE}; OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    "${PYTHON_BIN}" -m rabitq.rabitqlib_relerr \
      --dataset "${dataset}" \
      --base "${base}" \
      --query "${query}" \
      --vec_type fvecs \
      --clusters "${CLUSTERS}" \
      --bits ${BITS} \
      --topk "${TOPK}" \
      --nprobe "${NPROBE}" \
      --skip_build \
      --reuse_prepared_dir "${prepared_dir}" \
      --out_csv "${out_csv}"
    echo "[$(date --iso-8601=seconds)] EXIT ${dataset} rc=0"
  } >"${log}" 2>&1
}

for dataset in deep bigann msmarco openai gist; do
  if [[ ! -f "${ROOT}/lib/SAQ/data/${dataset}/${dataset}_centroid_${CLUSTERS}.fvecs" ]]; then
    echo "Skipping ${dataset}: missing SAQ centroid artifacts" | tee -a "${LOG_DIR}/missing.log"
    continue
  fi
  run_one "${dataset}"
done

echo "Logs: ${LOG_DIR}"
echo "Results: ${OUT_ROOT}"
