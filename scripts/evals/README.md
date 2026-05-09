# PQ/OPQ Evaluation Scripts

> **Cross-method ADC timing:** use the **`distance_eval/`** package
> (`python -m distance_eval.manifest`, `python -m distance_eval.harness`).
> See [distance_eval/README.md](../../distance_eval/README.md).
>
> **Deprecated (PQ-only legacy):** `measure_adc_cpu_time.py` and `measure_all_adc.py`
> — kept for reference; prefer `distance_eval` for fair IP+norm timing across methods.

Run evaluation metrics on models from `*_adc_vs_exact_eval.csv` files.

## Metrics

| Metric | Output CSV | Description |
|--------|------------|--------------|
| **compression_rate** | `{dataset}_{method}_compression_rate.csv` | bits_compressed / bits_original = (M*nbits) / (dim*32) |
| **reconstruction_error** | `{dataset}_{method}_reconstruction_error.csv` | Mean Euclidean distance between original and reconstructed vectors (on database) |
| **spearman** | `{dataset}_{method}_spearman.csv` | Spearman rank correlation between exact and ADC distance rankings (queries vs database) |
| **recall** | `{dataset}_{method}_recall.csv` | Recall@1, Recall@10, Recall@100 (queries vs database) |

All metrics use **database** and **query** vectors only (not train/learn).

## Usage

```bash
# From project root
python scripts/evals/run_evals.py \
  --input_csv /data/cpanourg/2-hdvc/results/deep_PQ_adc_vs_exact_eval.csv \
  --output_dir /data/cpanourg/2-hdvc/results/relerr_cpp

# With custom data root
python scripts/evals/run_evals.py \
  --input_csv /path/to/deep_PQ_adc_vs_exact_eval.csv \
  --output_dir /path/to/output \
  --data_root /data/cpanourg/2-hdvc/data
```

## Configuration

Dataset paths are in `config.py` (`DATASET_CONFIG`). Add or edit entries for your datasets.

Experiment folders must contain `pq_model.index` (FAISS IndexPQ). The `experiment_folder` column in the input CSV should point to the model directory.
