#!/usr/bin/env bash
set -u
cd /home/cpanourg/projects/2-hdvc
TARGET=290
LOG=/home/cpanourg/projects/2-hdvc/pq_quality_status/pq_quality_metrics.log
while true; do
  COUNT=$(find /mnthdd/cpanourg/2-hdvc/results/pq/quality_metrics/row_results -name 'row_*.json' 2>/dev/null | wc -l)
  echo "[supervisor] $(date -Is) count=${COUNT}/${TARGET}" >> "$LOG"
  if [ "$COUNT" -ge "$TARGET" ]; then
    /home/cpanourg/.conda/envs/dtwrl_env2/bin/python -u scripts/evals/pq_quality_metrics.py \
      --summary /mnthdd/cpanourg/2-hdvc/results/pq/pq_faiss_adc_summary.csv \
      --out_dir /mnthdd/cpanourg/2-hdvc/results/pq/quality_metrics \
      --clean_res_dir /home/cpanourg/projects/2-hdvc/results/clean_res \
      --max_workers 1 \
      --max_spearman_queries 500 \
      >> "$LOG" 2>&1
    exit 0
  fi
  /home/cpanourg/.conda/envs/dtwrl_env2/bin/python -u scripts/evals/pq_quality_metrics.py \
    --summary /mnthdd/cpanourg/2-hdvc/results/pq/pq_faiss_adc_summary.csv \
    --out_dir /mnthdd/cpanourg/2-hdvc/results/pq/quality_metrics \
    --clean_res_dir /home/cpanourg/projects/2-hdvc/results/clean_res \
    --max_workers 1 \
    --max_spearman_queries 500 \
    >> "$LOG" 2>&1 || true
  sleep 2
done
