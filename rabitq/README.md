# RaBitQ evaluation

Reference RaBitQ (Faiss test) + ADC vs exact **squared Euclidean** relative error, aligned with `scripts/rel_error_exps/PQ.py` where it matters.

## Results layout (default)

Under `data_root` (default `/data/cpanourg/2-hdvc`):

- **Aggregate CSV (times, errors, metadata):**  
  `{data_root}/results/rabitq/{dataset}_RaBitQ_adc_vs_exact_eval.csv`
- **Per run:**  
  `{data_root}/results/rabitq/{dataset}/bits{B}_train{train}_{timestamp}/summary.csv`  
  plus `rel_error_bits*_db*_qr*.bin` (full rel-error matrix).

## PQ-comparable protocol

1. **`--train_path`** (recommended): train RaBitQ on the **first `train_size` vectors** of the learn file.
2. **Base / test:** **first 1M vectors** after the train slice from `--dataset_path` (or from the same file with a non-overlapping offset if `train_path` and `dataset_path` resolve to the same file).
3. **Queries:** load **first `sample_queries` vectors** from `--query_path` (default 1000 for Deep).
4. **Evaluation:** **first `sample_db` base vectors × first `sample_queries` queries** (`--sample_mode first`, default), matching PQ’s “first N × first M” ADC vs exact block.
5. **Relative error:** same recipe as PQ — `epsilon=1e-6`, float64 intermediates, clip, finite mask — see `rabitq/eval.py`.
6. **Timing columns:** same names as PQ (`train_time_s`, `encoding_time_s`, `distance_table_time_s`, `cdist_time_s`, `adc_time_s`). RaBitQ has no PQ-style distance tables; `distance_table_time_s` is `0`.

CSV columns align with PQ/OPQ summaries: `n_subquantizers=0`, `nbits` = bits per query dimension, `bits_per_vector` = `nbits * dim`.

## Deep example (bits 1–12)

```bash
export DATA_FP=/data/cpanourg/2-hdvc
# Optional: separate learn file; if unset, script defaults train+base from one 100m .bin with non-overlapping ranges
export TRAIN_PATH="${DATA_FP}/data/deep1b/dataset/deep1b-96-100m.bin"
./rabitq/run_deep_bits_1_to_12.sh
```

## BigANN (SIFT), GIST, MSMARCO, OpenAI (bits 1–12)

Uses the same layout as `gpu_cpp_scripts/opq/run_opq_learn_then_train.sh` under `DATA_ROOT` (default `/data/cpanourg/2-hdvc`): `.bvecs` for BigANN, `.fvecs` for the others.

```bash
cd /path/to/2-hdvc
export PYTHONPATH="$PWD"
chmod +x rabitq/run_all_datasets_bits_1_to_12.sh
./rabitq/run_all_datasets_bits_1_to_12.sh
```

Include Deep as well:

```bash
DATASETS="bigann gist msmarco openai deep" ./rabitq/run_all_datasets_bits_1_to_12.sh
```

Override paths with e.g. `BIGANN_DATASET_PATH`, `GIST_TRAIN_PATH`, or set `DATA_ROOT` if your vectors live under a different root.

Override `DATASET_PATH`, `QUERY_PATH`, `TRAIN_SIZE`, `SAMPLE_DB` (10000), `SAMPLE_QUERIES` (1000) as needed.

## Dependencies

NumPy, SciPy (same as before). No Faiss required for this package.
