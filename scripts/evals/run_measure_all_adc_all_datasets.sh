#!/bin/bash
# Run measure_all_adc.py for all datasets in parallel.
# Each dataset writes to all_adc_timing_{dataset}.csv; then --merge combines them.

cd "$(dirname "$0")/../.."

python scripts/evals/measure_all_adc.py --dataset deep   --n_subq 8 --n_runs 10 --output_suffix deep   &
python scripts/evals/measure_all_adc.py --dataset bigann --n_subq 8 --n_runs 10 --output_suffix bigann &
python scripts/evals/measure_all_adc.py --dataset gist   --n_subq 480 --n_runs 10 --output_suffix gist   &
python scripts/evals/measure_all_adc.py --dataset msmarco --n_subq 512 --n_runs 10 --output_suffix msmarco &
python scripts/evals/measure_all_adc.py --dataset openai --n_subq 8 --n_runs 10 --output_suffix openai &

wait

python scripts/evals/measure_all_adc.py --merge
