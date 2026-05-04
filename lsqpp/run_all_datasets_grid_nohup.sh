#!/usr/bin/env bash
# LSQ++ grid for Deep, BigANN, GIST, MSMARCO, and OpenAI.
# Resumable: completed (dataset, M, nbits, train_size) rows are skipped.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/data/cpanourg/2-hdvc}"
DATA_PREFIX="${DATA_ROOT}/data"
EXPERIMENT_FOLDER="${EXPERIMENT_FOLDER:-lsqpp_grid_M2-7-15_nbits4-8-12_train10k-100k_$(date +%Y%m%d_%H%M%S)}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/lsqpp/${EXPERIMENT_FOLDER}}"
ROOT_OUT="${DATA_ROOT}/${RESULTS_SUBDIR}"

DATASETS="${DATASETS:-deep bigann gist msmarco openai}"
N_SUBQUANTIZERS_LIST="${N_SUBQUANTIZERS_LIST:-2 7 15}"
NBITS_LIST="${NBITS_LIST:-4 8 12}"
TRAIN_SIZES="${TRAIN_SIZES:-10000 100000}"
SAMPLE_DB="${SAMPLE_DB:-10000}"
SAMPLE_QUERIES="${SAMPLE_QUERIES:-1000}"
SEED="${SEED:-123}"
SAMPLE_MODE="${SAMPLE_MODE:-first}"
MAX_DATASET_JOBS="${MAX_DATASET_JOBS:-5}"
PYTHON="${PYTHON:-python3}"
RUN_FOREGROUND="${RUN_FOREGROUND:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONPATH="${REPO_ROOT}"

mkdir -p "${ROOT_OUT}/logs"

metadata_json="${ROOT_OUT}/metadata.json"
planned_csv="${ROOT_OUT}/planned_runs.csv"
failed_csv="${ROOT_OUT}/failed_runs.csv"

cat > "${metadata_json}" <<META
{
  "method": "LSQpp",
  "created_at": "$(date --iso-8601=seconds)",
  "repo_root": "${REPO_ROOT}",
  "data_root": "${DATA_ROOT}",
  "results_subdir": "${RESULTS_SUBDIR}",
  "datasets": "${DATASETS}",
  "n_subquantizers_list": "${N_SUBQUANTIZERS_LIST}",
  "nbits_list": "${NBITS_LIST}",
  "train_sizes": "${TRAIN_SIZES}",
  "sample_db": ${SAMPLE_DB},
  "sample_queries": ${SAMPLE_QUERIES},
  "seed": ${SEED},
  "sample_mode": "${SAMPLE_MODE}",
  "omp_num_threads": "${OMP_NUM_THREADS}",
  "mkl_num_threads": "${MKL_NUM_THREADS}",
  "resume_skip_sources": [
    "${ROOT_OUT}/{dataset}_LSQpp_adc_vs_exact_eval.csv",
    "${DATA_ROOT}/results/lsqpp/{dataset}_LSQpp_adc_vs_exact_eval.csv",
    "${DATA_ROOT}/results/{dataset}_LSQpp_adc_vs_exact_eval.csv"
  ]
}
META

if [[ ! -f "${planned_csv}" ]]; then
  printf 'dataset,dim,n_subquantizers,nbits,bits_per_vector,train_size,status\n' > "${planned_csv}"
fi
if [[ ! -f "${failed_csv}" ]]; then
  printf 'dataset,dim,n_subquantizers,nbits,bits_per_vector,train_size,exit_code,reason\n' > "${failed_csv}"
fi

resolve_dataset() {
  local ds="$1"
  DIM=""
  case "${ds}" in
    deep)
      DATASET_PATH="${DATA_PREFIX}/deep1b/dataset/test_1m.bin"
      TRAIN_PATH="${DATA_PREFIX}/deep1b/dataset/learn_100m.bin"
      QUERY_PATH="${DATA_PREFIX}/deep1b/dataset/query_10k.bin"
      DIM="96"
      ;;
    bigann)
      DATASET_PATH="${DATA_PREFIX}/bigann/SIFT1M/bigann_base.bvecs"
      TRAIN_PATH="${DATA_PREFIX}/bigann/SIFT1M/bigann_learn.bvecs"
      QUERY_PATH="${DATA_PREFIX}/bigann/SIFT1M/bigann_query.bvecs"
      DIM="128"
      ;;
    gist)
      DATASET_PATH="${DATA_PREFIX}/gist/gist_base.fvecs"
      TRAIN_PATH="${DATA_PREFIX}/gist/gist_learn.fvecs"
      QUERY_PATH="${DATA_PREFIX}/gist/gist_query.fvecs"
      DIM="960"
      ;;
    msmarco)
      DATASET_PATH="${DATA_PREFIX}/msmarco/base1m.fvecs"
      TRAIN_PATH="${DATA_PREFIX}/msmarco/train1m.fvecs"
      QUERY_PATH="${DATA_PREFIX}/msmarco/query10k.fvecs"
      DIM="1024"
      ;;
    openai)
      DATASET_PATH="${DATA_PREFIX}/openai/openai_base1m.fvecs"
      TRAIN_PATH="${DATA_PREFIX}/openai/openai_train1m.fvecs"
      QUERY_PATH="${DATA_PREFIX}/openai/openai_query10k.fvecs"
      DIM="1536"
      ;;
    *)
      echo "Unknown dataset: ${ds}" >&2
      return 1
      ;;
  esac
}

already_done() {
  local ds="$1" M="$2" nbits="$3" train_size="$4"
  "${PYTHON}" - "$ROOT_OUT" "$DATA_ROOT" "$ds" "$M" "$nbits" "$train_size" <<'PY'
import csv, pathlib, sys
root_out=pathlib.Path(sys.argv[1])
data_root=pathlib.Path(sys.argv[2])
ds=sys.argv[3]
M=int(sys.argv[4]); nbits=int(sys.argv[5]); train=int(sys.argv[6])
paths=[
    root_out / f"{ds}_LSQpp_adc_vs_exact_eval.csv",
    data_root / "results" / "lsqpp" / f"{ds}_LSQpp_adc_vs_exact_eval.csv",
    data_root / "results" / f"{ds}_LSQpp_adc_vs_exact_eval.csv",
]
for p in paths:
    if not p.exists():
        continue
    try:
        with p.open(newline="") as f:
            for r in csv.DictReader(f):
                try:
                    if int(float(r.get("n_subquantizers", -1))) == M and int(float(r.get("nbits", -1))) == nbits and int(float(r.get("train_size", -1))) == train:
                        raise SystemExit(0)
                except ValueError:
                    pass
    except Exception:
        continue
raise SystemExit(1)
PY
}

run_dataset() {
  local ds="$1"
  resolve_dataset "${ds}"

  for f in "${DATASET_PATH}" "${TRAIN_PATH}" "${QUERY_PATH}"; do
    if [[ ! -f "${f}" ]]; then
      echo "SKIP ${ds}: missing file: ${f}" >&2
      return 0
    fi
  done

  echo "[$(date --iso-8601=seconds)] START dataset=${ds} dim=${DIM}"
  for M in ${N_SUBQUANTIZERS_LIST}; do
    for NBITS in ${NBITS_LIST}; do
      for TRAIN_SIZE in ${TRAIN_SIZES}; do
        BPV=$((M * NBITS))
        set +e
        already_done "${ds}" "${M}" "${NBITS}" "${TRAIN_SIZE}"
        done_rc=$?
        set -e
        if [[ ${done_rc} -eq 0 ]]; then
          printf '%s,%s,%s,%s,%s,%s,skipped_done\n' "${ds}" "${DIM}" "${M}" "${NBITS}" "${BPV}" "${TRAIN_SIZE}" >> "${planned_csv}"
          echo "[$(date --iso-8601=seconds)] SKIP ${ds}: M=${M} nbits=${NBITS} train=${TRAIN_SIZE} already done"
          continue
        fi

        printf '%s,%s,%s,%s,%s,%s,running\n' "${ds}" "${DIM}" "${M}" "${NBITS}" "${BPV}" "${TRAIN_SIZE}" >> "${planned_csv}"
        echo "[$(date --iso-8601=seconds)] RUN ${ds}: M=${M} nbits=${NBITS} train=${TRAIN_SIZE} bpv=${BPV}"
        set +e
        "${PYTHON}" -m lsqpp.eval \
          --dataset_path "${DATASET_PATH}" \
          --train_path "${TRAIN_PATH}" \
          --query_path "${QUERY_PATH}" \
          --dim "${DIM}" \
          --dataset_name "${ds}" \
          --data_root "${DATA_ROOT}" \
          --n_subquantizers "${M}" \
          --nbits "${NBITS}" \
          --train_size "${TRAIN_SIZE}" \
          --sample_db "${SAMPLE_DB}" \
          --sample_queries "${SAMPLE_QUERIES}" \
          --results_dir "${RESULTS_SUBDIR}" \
          --seed "${SEED}" \
          --sample_mode "${SAMPLE_MODE}"
        rc=$?
        set -e
        if [[ ${rc} -ne 0 ]]; then
          printf '%s,%s,%s,%s,%s,%s,%s,lsqpp_eval_failed\n' "${ds}" "${DIM}" "${M}" "${NBITS}" "${BPV}" "${TRAIN_SIZE}" "${rc}" >> "${failed_csv}"
          echo "[$(date --iso-8601=seconds)] FAIL ${ds}: M=${M} nbits=${NBITS} train=${TRAIN_SIZE} rc=${rc}" >&2
        fi
      done
    done
  done
  echo "[$(date --iso-8601=seconds)] DONE dataset=${ds}"
}

pids=()
if [[ "${RUN_FOREGROUND}" == "1" ]]; then
  status=0
  for ds in ${DATASETS}; do
    if ! run_dataset "${ds}"; then
      status=1
    fi
  done
  echo "[$(date --iso-8601=seconds)] foreground dataset run finished status=${status}"
  echo "Results root: ${ROOT_OUT}"
  exit "${status}"
fi

for ds in ${DATASETS}; do
  (
    run_dataset "${ds}"
  ) > "${ROOT_OUT}/logs/${ds}.log" 2>&1 &
  pids+=("$!")
  echo "Launched ${ds} as PID $! (log: ${ROOT_OUT}/logs/${ds}.log)"
  while (( ${#pids[@]} >= MAX_DATASET_JOBS )); do
    new_pids=()
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        new_pids+=("${pid}")
      fi
    done
    pids=("${new_pids[@]}")
    if (( ${#pids[@]} >= MAX_DATASET_JOBS )); then
      sleep 10
    fi
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[$(date --iso-8601=seconds)] all dataset branches finished status=${status}"
echo "Results root: ${ROOT_OUT}"
exit "${status}"
