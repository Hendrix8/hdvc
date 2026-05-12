#!/usr/bin/env bash
set -euo pipefail
cd /home/cpanourg/projects/2-hdvc
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/cpanourg/projects/2-hdvc
export HDVC_RESULTS_ROOT=/home/cpanourg/projects/2-hdvc/results_timing
exec /home/cpanourg/.conda/envs/dtwrl_env2/bin/python -u -m saq.saq_ivf_grid \
  --output_root "$RUN_ROOT" \
  --datasets deep10k bigann10k gist10k msmarco10k openai10k \
  --clusters 64 128 256 \
  --bits 0.25 1 2 3 4 6 8 \
  --caq_adj_rd_lmt 6 \
  --searcher_vars_bound_m 4 \
  --n_runs 10 \
  --warmup_runs 5 \
  --num_threads 24 \
  --rand_rotate true \
  --use_fastscan true \
  --skip_build
