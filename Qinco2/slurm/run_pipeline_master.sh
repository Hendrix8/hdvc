#!/bin/bash
# Full QINCo2 AQ training + ADC evaluation + CSV aggregation for both experiment grids.
#
# Uses bundled extended fork at Qinco2/Qinco (wired via Qinco2/run.py + qinco symlink).
#
# Optional background:
#   nohup bash "$(readlink -f "$0")" > "$(dirname "$0")/pipeline_master.log" 2>&1 &
#
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="${CODE:-/home/cpanourg/projects/2-hdvc/Qinco2}"
PYTHON="${PYTHON:-$HOME/.miniconda3/envs/tsfm/bin/python}"
RESULT="${RESULT:-/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/compressed_qinco_res/unzip}"
export PATH="$(dirname "$PYTHON"):$PATH"
export QINCO2_DATA_ROOT="${QINCO2_DATA_ROOT:-/mnthdd/cpanourg/2-hdvc/results/urania_results/results/qinco2/.adc_data_symlinks}"

# Concurrent jobs: oversubscribe CPUs/RAM across GPUs (AQ train is mostly CPU-bound).
#   PARALLEL_WORKERS (default 16) = max concurrent subprocesses
#   GPU_IDS=e.g. 0,1 for round-robin CUDA_VISIBLE_DEVICES
# Serial: PIPELINE_SERIAL=1  OR  PARALLEL_GPUS=1  OR  PARALLEL_WORKERS=1

PARALLEL_WORKERS="${PARALLEL_WORKERS:-16}"

log() { echo "[$(date -Is)] $*"; }

run_jobs() {
  local script_path="$1"
  if [[ "${PIPELINE_SERIAL:-0}" == "1" ]] || [[ "${PARALLEL_GPUS:-}" == "1" ]] || [[ "${PARALLEL_WORKERS:-16}" == "1" ]]; then
    log "Serial mode (PIPELINE_SERIAL / PARALLEL_GPUS=1 / PARALLEL_WORKERS=1)"
    bash "$script_path"
    return
  fi
  log "Parallel mode: PARALLEL_WORKERS=${PARALLEL_WORKERS}${GPU_IDS:+ GPU_IDS=${GPU_IDS}}"
  CMD=( "$PYTHON" "$CODE/slurm/run_bash_jobs_parallel_gpus.py" "$script_path" --workers "$PARALLEL_WORKERS" )
  [[ -n "${GPU_IDS:-}" ]] && CMD+=( --gpu-ids "$GPU_IDS" )
  "${CMD[@]}"
}

if ! (cd "$CODE" && "$PYTHON" -c "import qinco.search.search_tasks as m; assert hasattr(m, 'TrainQincoAQTask') and hasattr(m, 'QincoAQComputeDistancesTask')"); then
  log "ERROR: qinco does not expose TrainQincoAQTask / QincoAQComputeDistancesTask (run from Qinco2 so bundled Qinco2/Qinco wins)."
  exit 2
fi

prepend_env_header() {
  local f="$1"
  local t
  t="$(mktemp)"
  {
    echo '#!/bin/bash'
    echo 'set -eo pipefail'
    echo "export PATH=\"$(dirname "$PYTHON"):\$PATH\""
    echo "export QINCO2_DATA_ROOT=\"$QINCO2_DATA_ROOT\""
    tail -n +2 "$f"
  } >"$t"
  mv "$t" "$f"
  chmod +x "$f"
}

regen_adc_main_grid() {
  local BASH_ADC="$HERE/run_adc_main_grid.sh"
  local COMMON=(
    --mode qinco_aq_compute_distances
    --ds_loop 100000
    --encode_first_n 10000
    --n_db 10000
    --n_queries 1000
    --tqa_n_train 10000
    --tqa_n_val 10000
    --min_M 1
    --result_path "$RESULT"
    --code_path "$CODE"
    --bash_full_fname "$BASH_ADC"
    --print_anomalies
  )
  "$PYTHON" "$CODE/slurm/generate_slurm_cmds.py" "${COMMON[@]}" --overwrite_existing_bash --max_M 32 --all_datasets bigann >/dev/null
  "$PYTHON" "$CODE/slurm/generate_slurm_cmds.py" "${COMMON[@]}" --max_M 64 --all_datasets deep gist msmarco openai >/dev/null
  prepend_env_header "$BASH_ADC"
}

regen_adc_train_size() {
  local BASH_ADC="$HERE/run_adc_train_size_M8_K256.sh"
  local COMMON=(
    --mode qinco_aq_compute_distances
    --ds_loop 100000
    --encode_first_n 10000
    --n_db 10000
    --n_queries 1000
    --tqa_n_train 10000
    --tqa_n_val 10000
    --print_anomalies
    --all_M 8
    --all_K 256
    --result_path "$RESULT"
    --code_path "$CODE"
    --bash_full_fname "$BASH_ADC"
  )
  for nt in 0.05 0.1 0.25 0.5 0.75; do
    local OW=()
    if [[ "$nt" == "0.05" ]]; then OW=(--overwrite_existing_bash); fi
    "$PYTHON" "$CODE/slurm/generate_slurm_cmds.py" "${COMMON[@]}" "${OW[@]}" --n_train "$nt" --n_val 10000 >/dev/null
  done
  prepend_env_header "$BASH_ADC"
}

log "=== Main grid: train_qinco_aq ==="
run_jobs "$HERE/run_train_qinco_aq_main_grid.sh"

log "=== Main grid: regenerate ADC bash ==="
regen_adc_main_grid

log "=== Main grid: qinco_aq_compute_distances ==="
run_jobs "$HERE/run_adc_main_grid.sh"

# Multi-root ADC CSV discovery: "$CODE/aggregate_qinco_adc_results.py" --help
# Run missing ADC evals (train_qinco_aq_setting_*.npz + encode_test *_0-9999): QINCO2_DATA_ROOT=... "$CODE/run_discovered_adc_jobs.py"
log "=== Main grid: CSV ==="
"$PYTHON" "$CODE/save_results_to_csv.py" --mode qinco_aq_compute_distances \
  --ds_loop 100000 --encode_first_n 10000 --n_db 10000 --n_queries 1000 \
  --tqa_n_train 10000 --tqa_n_val 10000 \
  --max_M 64 --min_M 1 --result_path "$RESULT" --csv_save_path "$HERE/main_grid_csv"

log "=== Train-size sweep: train_qinco_aq ==="
run_jobs "$HERE/run_train_qinco_aq_train_size_M8_K256.sh"

log "=== Train-size sweep: regenerate ADC bash ==="
regen_adc_train_size

log "=== Train-size sweep: qinco_aq_compute_distances ==="
run_jobs "$HERE/run_adc_train_size_M8_K256.sh"

log "=== Train-size sweep: CSV ==="
"$PYTHON" "$CODE/save_results_to_csv.py" --mode qinco_aq_compute_distances \
  --ds_loop 100000 --encode_first_n 10000 --n_db 10000 --n_queries 1000 \
  --tqa_n_train 10000 --tqa_n_val 10000 \
  --all_M 8 --all_K 256 --all_n_train 0.05 0.1 0.25 0.5 0.75 1. \
  --n_val 10000 --max_M 8 --min_M 8 \
  --result_path "$RESULT" --csv_save_path "$HERE/train_size_M8_K256_csv"

log "All phases finished."
