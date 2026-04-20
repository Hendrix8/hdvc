# VAQ evaluation (self-contained)

This directory does **not** depend on `rabitq/`. It includes:

| File | Role |
|------|------|
| `data_io.py` | Read/write `.fvecs`, `.bin`, `.fbin`, `.bvecs`; PQ-style prefix loading |
| `metrics.py` | Relative error vs exact L2² (same recipe as PQ/OPQ scripts) |
| `eval.py` | CLI: calls `lib/VAQ/build/examples/run_vaq`, then ADC vs exact on a 10k×1k slice |
| `run_all_datasets.sh` | BigANN, GIST, MSMARCO, OpenAI (+ optional `deep` via `DATASETS`) |
| `run_deep_hyperparam_grid.sh` | Deep only: **PQ/OPQ-style grid** (M × nbits × train_size) → VAQ methods; see `experiments/README.md` |

Results default to `{data_root}/results/vaq/` and `{dataset}_VAQ_adc_vs_exact_eval.csv`.

```bash
export PYTHONPATH=/path/to/2-hdvc
python3 -m vaq.eval --dataset_path ... --train_path ... --query_path ... --dataset_name bigann
```

### Deep — choose hyperparameters (`run_deep.sh`)

Defaults: PQ-style files `test_1m.bin`, `learn_100m.bin`, `query_10k.bin`, `dim=96`, `train_size=100k`, `10k×1k` eval slice.

**Full method string:**

```bash
METHOD="VAQ256m32min7max8var1,HEAP" \
TRAIN_SIZE=500000 \
REFINE="100,200" K=100 LEARN_RATIO=0.05 \
./vaq/run_deep.sh
```

**Or build `VAQ{total}m{m}min{lo}max{hi}var{v},{SEARCH}` from parts** (omit `METHOD`):

```bash
TOTAL_BITS=256 N_SUBSPACES=32 MIN_BITS=7 MAX_BITS=8 VARIANCE=1 SEARCH=HEAP \
TRAIN_SIZE=100000 ./vaq/run_deep.sh
```

Override data locations: `DEEP_TEST_PATH`, `DEEP_LEARN_PATH`, `DEEP_QUERY_PATH`, or `DATA_PREFIX` / `DATA_ROOT`.

All `vaq.eval` flags are also available by calling `python3 -m vaq.eval` directly (`--total_bits`, `--n_subspaces`, … to rewrite the method string).

Requires a built VAQ binary at `lib/VAQ/build/examples/run_vaq`.

### Deep only — same *data* as PQ, same *train sizes* as OPQ

`run_deep_pq_opq_style.sh` uses **PQ-style paths**: `test_1m.bin` (base), `learn_100m.bin` (train), `query_10k.bin` (queries), `dim=96`.

It sweeps **`train_size` ∈ {10k, 100k, 1M}** like `scripts/rel_error_exps/run_opq.sh`.  
VAQ does **not** share the same `(M, nbits)` knobs as PQ/OPQ; the default method is `VAQ256m32min7max8var1,HEAP` (same default as legacy `VAQ.py`). Override with **`METHODS`** (space-separated) if you have other valid `run_vaq` method strings.

```bash
./vaq/run_deep_pq_opq_style.sh
```

Examples:

```bash
# Only 100k train, single method
TRAIN_SIZES="100000" ./vaq/run_deep_pq_opq_style.sh

# Two method strings × all train sizes
METHODS="VAQ256m32min7max8var1,HEAP VAQ128m16min7max8var1,HEAP" ./vaq/run_deep_pq_opq_style.sh
```
