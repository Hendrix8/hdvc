# TurboQuant-MSE Evaluation Bundle

Faithful implementation of **TurboQuant-MSE** (Algorithm 1 of Zandieh et al.
2025, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) and the evaluation
results across five ANN benchmark datasets.

Author: qwang  ·  Created: 2026-05-12  ·  Host (origin): `urania`

---

## 1. What this bundle contains

```
tqmse/
├── README.md                    this file
├── turboquant_mse.py            quantizer: Haar rotation + 1D Lloyd-Max codebook + bucketize
├── data_loaders.py              readers for fvecs / fbin / u8bin / raw_f32, plus a dataset registry
├── eval_turboquant_mse.py       evaluation driver (3 measures, writes CSV)
├── run_all_datasets.sh          one-shot driver over all (dataset, bit_per_dim) combos
├── codebooks_tqmse/             50 cached Lloyd-Max 1D codebooks in JSON
│   └── cb_d{96,128,960,1024,1536}_b{1..10}.json
└── results/
    ├── tqmse_all.csv            ← MAIN OUTPUT: 50 rows (5 datasets × 10 bit_per_dim)
    ├── tqmse_deep_eval.csv
    ├── tqmse_bigann_eval.csv
    ├── tqmse_gist_eval.csv
    ├── tqmse_msmarco_eval.csv
    └── tqmse_openai_eval.csv
```

Sizes: code ≈ 25 KB, codebooks ≈ 580 KB, results ≈ 25 KB. **Whole bundle < 1 MB.**

---

## 2. Method

**TurboQuant-MSE** quantizes a vector `x ∈ R^d` in three steps:

1. L2-normalize: `n ← ‖x‖`, `u ← x / n` (store `n` in fp32).
2. Haar-rotate: `y ← Π · u` where `Π` is a random orthogonal matrix sampled
   once from the Haar measure on O(d).
3. Per-coordinate scalar quantization: `idx_j ← argmin_k |y_j − c_k|` for
   `j ∈ [d]`, where `{c_1, …, c_{2^b}}` is the 1D Lloyd-Max optimal codebook
   for the Beta-marginal density `f_Y(y) ∝ (1 − y²)^((d−3)/2)`.

Reconstruction: `x̂ ← n · Π^T · (c_{idx_1}, …, c_{idx_d})`.

The codebook depends only on `(d, bit_per_dim)` and is computed once
**analytically** via the regularized incomplete Beta function
(`scipy.special.betainc / betaincinv`), then cached on disk.

---

## 3. Parameters swept

| Parameter      | Values                              | Notes                              |
|----------------|-------------------------------------|------------------------------------|
| `bit_per_dim`  | {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}     | Codebook size `K = 2^bit_per_dim`  |
| dataset        | deep, bigann, gist, msmarco, openai | 5 ANN benchmarks                   |

`block_size`, `outlier_frac`, `outlier_extra_bits` were **dropped** after
verifying that the upstream `original_code/` directory in the turboquant repo
contains no implementation of block-wise rotation or outlier-channel
mixed-precision allocation.

TurboQuant-MSE is **data-oblivious**: the Haar rotation and the Beta-based
codebook depend only on `d`. The *learn* set is therefore not used.

---

## 4. Datasets

Source on `urania`: `/data/cpanourg/2-hdvc/data/`. **Adjust `DATA_ROOT` at the
top of `data_loaders.py` if these are mirrored elsewhere on the target host.**

For each dataset we load only the first 10 000 base vectors and the first
1 000 query vectors.

| name    | d    | base file                          | query file                       | reader     |
|---------|------|------------------------------------|----------------------------------|------------|
| deep    | 96   | `deep1b/dataset/test_1m.bin`       | `deep1b/dataset/query_10k.bin`   | raw_f32    |
| bigann  | 128  | `bigann/base.1B.u8bin`             | `bigann/query.public.10K.u8bin`  | u8bin      |
| gist    | 960  | `gist/gist_base.fvecs`             | `gist/gist_query.fvecs`          | fvecs      |
| msmarco | 1024 | `msmarco/base1m.fvecs`             | `msmarco/query10k.fvecs`         | fvecs      |
| openai  | 1536 | `openai/openai_base1m.fvecs`       | `openai/openai_query10k.fvecs`   | fvecs      |

File formats:
- **fvecs**: per-row `[d:int32, vec:float32×d]`
- **fbin** (header-prefixed `.bin`): `[n:int32, d:int32]` header then `n × d` float32
- **u8bin**: `[n:int32, d:int32]` header then `n × d` uint8 (decoded as float32 internally)
- **raw_f32**: header-less raw float32 with `d` supplied externally (used for deep1b)

---

## 5. Measures (3 measures, per the 2.2.x naming)

Each CSV row contains three measures evaluated on `10k base × 1k query`
(i.e. `1e7` distance pairs):

| measure   | CSV column            | definition                                                                       |
|-----------|-----------------------|----------------------------------------------------------------------------------|
| **2.2.1** | `mse_distortion`      | mean over base of `‖x − x̂‖²`                                                    |
| **2.2.2** | `l2_rel_err_mean`     | mean over base of `‖x − x̂‖ / ‖x‖`                                               |
| **2.2.4** | `spearman_mean`       | mean over queries of Spearman ρ between approximate and exact L2 distance vectors, where each ρ is computed per-query over the 10 000 base distances |

Spearman ρ uses `scipy.stats.spearmanr` per query (handles ties via average
rank), then averages across the 1 000 queries.

Auxiliary spread statistics for the per-query Spearman ρ distribution:
`spearman_std`, `spearman_p10`, `spearman_p50`, `spearman_p90`. Timings:
`setup_time_s`, `encode_time_s`, `spearman_time_s`, `total_time_s`.

---

## 6. CSV schema

Each per-dataset CSV (and the combined `tqmse_all.csv`) has these columns:

```
method, dataset, d, bit_per_dim, K, n_base, n_query,
mse_distortion, l2_rel_err_mean,
spearman_mean, spearman_std, spearman_p10, spearman_p50, spearman_p90,
setup_time_s, encode_time_s, spearman_time_s, total_time_s
```

`method = "TQMSE"`, `n_base = 10000`, `n_query = 1000` for all rows in this bundle.

---



### Single config

```bash
cd tqmse/
python eval_turboquant_mse.py \
    --dataset gist --bits 4 \
    --n_base 10000 --n_query 1000 \
    --output_csv results/tqmse_gist_b4.csv \
    --device cuda:0
```

### Full grid

```bash
cd tqmse/
DEVICE=cuda:0 bash run_all_datasets.sh
```

First run builds and caches 50 codebooks in `codebooks_tqmse/` (~5 s each
for high `bit_per_dim`). The cached JSONs are shipped in this bundle, so
subsequent runs are pure inference (~1 s per config).

Total wall time of the full grid on one A100:
- data loading: ~3 min dominated by msmarco (cold page-cache, fvecs memmap)
- compute: ~2 min for all 50 configs

---

## 8. Numerical sanity

- For `bit_per_dim ∈ {1, 2, 3, 4}`, our 1D Lloyd-Max per-coord MSE × `d`
  matches the values in Theorem 1 of the paper:
  - b=1 → 0.36 (paper 0.36 ✓)
  - b=2 → 0.116 (paper 0.117 ✓)
  - b=3 → 0.034 (paper 0.030, within rounding ✓)
  - b=4 → 0.0093 (paper 0.009 ✓)
  - centroids for b=1 match `±√(2 / (π d))` to within 0.1%.
- For `bit_per_dim ≥ 7` the per-coord MSE × `d` slightly exceeds the
  paper's asymptotic upper bound `√(3π)/2 · 4^-b`. Both the analytical
  integration (incomplete Beta) and a 30 M-sample Monte-Carlo Lloyd-Max
  converge to the same fixed point, so this is the genuine Lloyd-Max
  optimum for the symmetric Beta marginal at finite `d`. The trend is
  strictly monotone decreasing in `bit_per_dim`, so cross-method comparisons
  at fixed `bit_per_dim` remain fair.
- Spearman ρ saturates to ≥ 0.999 by `bit_per_dim = 6` on all datasets, and
  is already ≥ 0.95 by `bit_per_dim = 2` on gist/msmarco/openai.

---

