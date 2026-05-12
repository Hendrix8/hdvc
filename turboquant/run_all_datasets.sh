#!/usr/bin/env bash
# Run TurboQuant-MSE eval across all five datasets, bit_per_dim in {1..10}.
# Outputs one CSV per dataset under tqmse/results/, plus a combined tqmse_all.csv.

set -euo pipefail

PY=/home/qwang/software/miniforge3/envs/vllm/bin/python
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
OUT=${HERE}/results
mkdir -p "${OUT}"

DATASETS=("deep" "bigann" "gist" "msmarco" "openai")
BITS="1 2 3 4 5 6 7 8 9 10"
DEVICE=${DEVICE:-cuda:1}

cd "${HERE}"

for ds in "${DATASETS[@]}"; do
    csv="${OUT}/tqmse_${ds}_eval.csv"
    rm -f "${csv}"
    echo
    echo "================================================================"
    echo "==== Dataset: ${ds}   device=${DEVICE}   bits=${BITS}"
    echo "================================================================"
    ${PY} eval_turboquant_mse.py \
        --dataset "${ds}" \
        --bits ${BITS} \
        --n_base 10000 \
        --n_query 1000 \
        --device "${DEVICE}" \
        --output_csv "${csv}"
done

echo
echo "==== Combining all per-dataset CSVs ===="
COMBO="${OUT}/tqmse_all.csv"
# Take header from first file then append data rows of all
head -1 "${OUT}/tqmse_${DATASETS[0]}_eval.csv" > "${COMBO}"
for ds in "${DATASETS[@]}"; do
    tail -n +2 "${OUT}/tqmse_${ds}_eval.csv" >> "${COMBO}"
done
echo "Wrote ${COMBO}"
wc -l "${OUT}"/*.csv
