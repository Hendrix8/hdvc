#!/usr/bin/env bash
# Launch VAQ grid experiments for Deep, BigANN, GIST, MSMARCO, and OpenAI.
#
# Grid:
#   min_bits         in {2, 4, 6}
#   max_bits         in {8, 16}
#   bits_per_vector  in {4*dim, 8*dim, 12*dim}
#   variance         = 1.0
#
# We set n_subspaces = dim so the requested bit budgets correspond to average
# bits/subspace of 4, 8, and 12 respectively. Infeasible combinations where
# average bits/subspace is outside [min_bits, max_bits] are recorded and skipped.
#
# Usage:
#   nohup bash vaq/run_all_datasets_bpv_grid_nohup.sh > /data/cpanourg/2-hdvc/results/vaq/<experiment>/nohup.out 2>&1 &
#
# Useful env overrides:
#   EXPERIMENT_FOLDER=vaq_bpv_grid_20260501_120000
#   DATASETS="bigann gist openai deep msmarco"
#   MAX_DATASET_JOBS=5
#   TRAIN_SIZE=100000
#   SAMPLE_DB=10000
#   SAMPLE_QUERIES=1000
#   DRY_RUN=1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/data/cpanourg/2-hdvc}"
DATA_ROOT="${DATA_ROOT%/}"
DATA_PREFIX="${DATA_PREFIX:-${DATA_ROOT}/data}"
EXPERIMENT_FOLDER="${EXPERIMENT_FOLDER:-vaq_bpv_grid_min2-4-6_max8-16_bpv4-8-12_$(date +%Y%m%d_%H%M%S)}"
RESULTS_SUBDIR="results/vaq/${EXPERIMENT_FOLDER}"
EXPERIMENT_ROOT="${DATA_ROOT}/${RESULTS_SUBDIR}"
LOG_DIR="${EXPERIMENT_ROOT}/logs"

DATASETS="${DATASETS:-bigann gist openai deep msmarco}"
MIN_BITS_LIST="${MIN_BITS_LIST:-2 4 6}"
MAX_BITS_LIST="${MAX_BITS_LIST:-8 16}"
BPV_MULTIPLIERS="${BPV_MULTIPLIERS:-4 8 12}"
VARIANCE="${VARIANCE:-1.0}"
SEARCH="${SEARCH:-HEAP}"
TRAIN_SIZE="${TRAIN_SIZE:-100000}"
SAMPLE_DB="${SAMPLE_DB:-10000}"
SAMPLE_QUERIES="${SAMPLE_QUERIES:-1000}"
REFINE="${REFINE:-100,200}"
K="${K:-100}"
LEARN_RATIO="${LEARN_RATIO:-0.05}"
SKIP_SEARCH="${SKIP_SEARCH:-1}"
MAX_DATASET_JOBS="${MAX_DATASET_JOBS:-5}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python3}"
VAQ_BINARY="${VAQ_BINARY:-${REPO_ROOT}/lib/VAQ/build/examples/run_vaq}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${VAQ_BINARY}" ]]; then
  echo "ERROR: VAQ binary not found: ${VAQ_BINARY}" >&2
  exit 1
fi

metadata_json="${EXPERIMENT_ROOT}/experiment_metadata.json"
planned_csv="${EXPERIMENT_ROOT}/planned_runs.csv"
skipped_csv="${EXPERIMENT_ROOT}/skipped_runs.csv"
failed_csv="${EXPERIMENT_ROOT}/failed_runs.csv"

cat > "${metadata_json}" <<META
{
  "experiment_folder": "${EXPERIMENT_FOLDER}",
  "experiment_root": "${EXPERIMENT_ROOT}",
  "results_dir_argument": "${RESULTS_SUBDIR}",
  "datasets": "${DATASETS}",
  "min_bits_list": "${MIN_BITS_LIST}",
  "max_bits_list": "${MAX_BITS_LIST}",
  "bits_per_vector_multipliers": "${BPV_MULTIPLIERS}",
  "variance": "${VARIANCE}",
  "search": "${SEARCH}",
  "n_subspaces_policy": "n_subspaces = dim",
  "train_size": ${TRAIN_SIZE},
  "sample_db": ${SAMPLE_DB},
  "sample_queries": ${SAMPLE_QUERIES},
  "refine": "${REFINE}",
  "k": ${K},
  "learn_ratio": ${LEARN_RATIO},
  "skip_search": "${SKIP_SEARCH}",
  "omp_num_threads": "${OMP_NUM_THREADS}",
  "mkl_num_threads": "${MKL_NUM_THREADS}",
  "vaq_binary": "${VAQ_BINARY}"
}
META

if [[ ! -f "${planned_csv}" ]]; then
  printf 'dataset,dim,bpv_multiplier,bits_per_vector,n_subspaces,min_bits,max_bits,variance,method,train_size,status\n' > "${planned_csv}"
fi
if [[ ! -f "${skipped_csv}" ]]; then
  printf 'dataset,dim,bpv_multiplier,bits_per_vector,n_subspaces,min_bits,max_bits,variance,reason\n' > "${skipped_csv}"
fi
if [[ ! -f "${failed_csv}" ]]; then
  printf 'dataset,dim,bpv_multiplier,bits_per_vector,n_subspaces,min_bits,max_bits,variance,method,train_size,exit_code,reason\n' > "${failed_csv}"
fi

dataset_paths() {
  local name="$1"
  DIM_OPT=()
  case "${name}" in
    bigann)
      DIM=128
      DIM_OPT=(--dim "${DIM}")
      DATASET_PATH="${BIGANN_DATASET_PATH:-${DATA_PREFIX}/bigann/SIFT1M/bigann_base.bvecs}"
      QUERY_PATH="${BIGANN_QUERY_PATH:-${DATA_PREFIX}/bigann/SIFT1M/bigann_query.bvecs}"
      TRAIN_PATH="${BIGANN_TRAIN_PATH:-${DATA_PREFIX}/bigann/SIFT1M/bigann_learn.bvecs}"
      ;;
    gist)
      DIM=960
      DIM_OPT=(--dim "${DIM}")
      DATASET_PATH="${GIST_DATASET_PATH:-${DATA_PREFIX}/gist/gist_base.fvecs}"
      QUERY_PATH="${GIST_QUERY_PATH:-${DATA_PREFIX}/gist/gist_query.fvecs}"
      TRAIN_PATH="${GIST_TRAIN_PATH:-${DATA_PREFIX}/gist/gist_learn.fvecs}"
      ;;
    msmarco)
      DIM=1024
      DIM_OPT=(--dim "${DIM}")
      DATASET_PATH="${MSMARCO_DATASET_PATH:-${DATA_PREFIX}/msmarco/base1m.fvecs}"
      QUERY_PATH="${MSMARCO_QUERY_PATH:-${DATA_PREFIX}/msmarco/query10k.fvecs}"
      TRAIN_PATH="${MSMARCO_TRAIN_PATH:-${DATA_PREFIX}/msmarco/train1m.fvecs}"
      ;;
    openai)
      DIM=1536
      DIM_OPT=(--dim "${DIM}")
      DATASET_PATH="${OPENAI_DATASET_PATH:-${DATA_PREFIX}/openai/openai_base1m.fvecs}"
      QUERY_PATH="${OPENAI_QUERY_PATH:-${DATA_PREFIX}/openai/openai_query10k.fvecs}"
      TRAIN_PATH="${OPENAI_TRAIN_PATH:-${DATA_PREFIX}/openai/openai_train1m.fvecs}"
      ;;
    deep)
      DIM=96
      DATASET_PATH="${DEEP_DATASET_PATH:-${DATA_PREFIX}/deep1b/dataset/test_1m.bin}"
      QUERY_PATH="${DEEP_QUERY_PATH:-${DATA_PREFIX}/deep1b/dataset/query_10k.bin}"
      TRAIN_PATH="${DEEP_TRAIN_PATH:-${DATA_PREFIX}/deep1b/dataset/learn_100m.bin}"
      DIM_OPT=(--dim "${DIM}")
      ;;
    *)
      echo "Unknown dataset: ${name}" >&2
      return 1
      ;;
  esac
}

run_dataset() {
  local ds="$1"
  local ds_log="${LOG_DIR}/${ds}.log"
  if [[ "${APPEND_LOGS:-0}" == "1" ]]; then
    exec >> "${ds_log}" 2>&1
  else
    exec > "${ds_log}" 2>&1
  fi
  local agg_csv="${EXPERIMENT_ROOT}/${ds}_VAQ_adc_vs_exact_eval.csv"
  {
    echo "[$(date --iso-8601=seconds)] START dataset=${ds}"
    dataset_paths "${ds}"

    for f in "${DATASET_PATH}" "${TRAIN_PATH}" "${QUERY_PATH}"; do
      if [[ ! -f "${f}" ]]; then
        echo "ERROR: missing file for ${ds}: ${f}" >&2
        return 1
      fi
    done

    local n_subspaces="${DIM}"
    for min_bits in ${MIN_BITS_LIST}; do
      for max_bits in ${MAX_BITS_LIST}; do
        if (( min_bits > max_bits )); then
          continue
        fi
        for mult in ${BPV_MULTIPLIERS}; do
          local total_bits=$((mult * DIM))
          local method="VAQ${total_bits}m${n_subspaces}min${min_bits}max${max_bits}var${VARIANCE},${SEARCH}"
          local feasible=1
          local reason=""
          if (( mult < min_bits )); then
            feasible=0
            reason="avg_bits_per_subspace_${mult}_lt_min_bits_${min_bits}"
          elif (( mult > max_bits )); then
            feasible=0
            reason="avg_bits_per_subspace_${mult}_gt_max_bits_${max_bits}"
          fi

          if (( feasible == 0 )); then
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
              "${ds}" "${DIM}" "${mult}" "${total_bits}" "${n_subspaces}" \
              "${min_bits}" "${max_bits}" "${VARIANCE}" "${reason}" >> "${skipped_csv}"
            echo "SKIP ${ds}: ${method} (${reason})"
            continue
          fi

          if [[ -f "${agg_csv}" ]] && grep -Fq "\"${method}\"" "${agg_csv}"; then
            printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,completed_existing\n' \
              "${ds}" "${DIM}" "${mult}" "${total_bits}" "${n_subspaces}" \
              "${min_bits}" "${max_bits}" "${VARIANCE}" "${method}" "${TRAIN_SIZE}" >> "${planned_csv}"
            echo "SKIP ${ds}: ${method} (already completed)"
            continue
          fi

          printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,planned\n' \
            "${ds}" "${DIM}" "${mult}" "${total_bits}" "${n_subspaces}" \
            "${min_bits}" "${max_bits}" "${VARIANCE}" "${method}" "${TRAIN_SIZE}" >> "${planned_csv}"

          echo "[$(date --iso-8601=seconds)] RUN ${ds}: ${method} train=${TRAIN_SIZE}"
          cmd=(
            "${PYTHON}" -m vaq.eval
            --dataset_path "${DATASET_PATH}"
            --train_path "${TRAIN_PATH}"
            --query_path "${QUERY_PATH}"
            "${DIM_OPT[@]}"
            --dataset_name "${ds}"
            --data_root "${DATA_ROOT}"
            --method "${method}"
            --train_size "${TRAIN_SIZE}"
            --sample_db "${SAMPLE_DB}"
            --sample_queries "${SAMPLE_QUERIES}"
            --results_dir "${RESULTS_SUBDIR}"
            --vaq_binary "${VAQ_BINARY}"
            --refine "${REFINE}"
            --k "${K}"
            --learn_ratio "${LEARN_RATIO}"
            --sample_mode first
          )
          if [[ "${SKIP_SEARCH}" == "1" ]]; then
            cmd+=(--skip_search)
          fi

          if [[ "${DRY_RUN}" == "1" ]]; then
            printf 'DRY_RUN:'
            printf ' %q' "${cmd[@]}"
            printf '\n'
          else
            set +e
            "${cmd[@]}"
            rc=$?
            set -e
            if (( rc != 0 )); then
              printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s\n' \
                "${ds}" "${DIM}" "${mult}" "${total_bits}" "${n_subspaces}" \
                "${min_bits}" "${max_bits}" "${VARIANCE}" "${method}" \
                "${TRAIN_SIZE}" "${rc}" "vaq_eval_failed" >> "${failed_csv}"
              echo "FAILED ${ds}: ${method} (exit=${rc}); continuing"
              continue
            fi
          fi
        done
      done
    done
    echo "[$(date --iso-8601=seconds)] DONE dataset=${ds}"
  }
}

pids=()
for ds in ${DATASETS}; do
  while (( ${#pids[@]} >= MAX_DATASET_JOBS )); do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        wait "${pids[$i]}" || true
        unset 'pids[i]'
      fi
    done
    pids=("${pids[@]}")
    sleep 2
  done
  run_dataset "${ds}" &
  pids+=("$!")
  echo "Launched ${ds} as PID ${pids[-1]} (log: ${LOG_DIR}/${ds}.log)"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Metadata: ${metadata_json}"
echo "Planned runs: ${planned_csv}"
echo "Skipped runs: ${skipped_csv}"
echo "Logs: ${LOG_DIR}"
exit "${status}"
