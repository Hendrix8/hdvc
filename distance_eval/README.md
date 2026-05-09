# distance_eval — unified IP + norm ADC

All supported methods estimate squared Euclidean distance to a reconstruction \(\hat{x}\) using the same decomposition:

\[
\|q - \hat{x}\|^2 = \|q\|^2 + \|\hat{x}\|^2 - 2\langle q, \hat{x}\rangle
\]

Each **backend** only implements:

1. **`encode(db)`** — database codes (opaque per method).
2. **`db_norms_sq(codes)`** — per-vector \(\|\hat{x}\|^2\), precomputed once (timed separately as `db_prep_time_s` in the harness).
3. **`prepare_query(q)`** — per-query LUT / state (timed as `lut_time_s`).
4. **`ip_estimate(qstate, codes)`** — matrix of inner products \(\langle q, \hat{x}\rangle\) (timed as `ip_scan_time_s`).

The harness then applies shared **`compose_l2`** and reports CPU `process_time` over `n_runs`.

## Layout

| Path | Role |
|------|------|
| `distance_eval/backends/pq.py` | FAISS `ProductQuantizer`, IP LUT + Numba IP scan |
| `distance_eval/backends/opq.py` | `IndexPreTransform` + IP LUT on rotated queries |
| `distance_eval/backends/lsq.py` | LSQ++ dot tables + `alpha` norms |
| `distance_eval/backends/vaq.py` | VAQ `codes.fvecs` / `centroids.fvecs` + per-subspace IP LUT |
| `distance_eval/backends/qinco2.py` | QINCo2 decode + GEMM (requires `lib.Qinco`, torch) |
| `distance_eval/backends/rabitq.py` | **Placeholder** — wire `lib/RaBitQ-Library` multi-bit kernel here |
| `distance_eval/manifest.py` | Build `results/unified_manifest.csv` from `relerr_cpp/*_adc_vs_exact_eval.csv` |
| `distance_eval/harness.py` | Run timings → `results/unified_adc_timing.csv` (+ by BPV / by method) |
| `distance_eval/figures.py` | Pareto scatter: `rel_error_mean` vs `adc_total_time_pp_mean` |

## Usage

From the **repository root** (so `hdvc_paths`, `scripts.evals.config`, and datasets resolve):

```bash
# 1) Build manifest (default: results in HDVC_RESULTS_ROOT or ./results/relerr_cpp)
python -m distance_eval.manifest \
  --results_dir /path/to/results/relerr_cpp \
  --output distance_eval/results/unified_manifest.csv

# 2) Time unified ADC (CPU process time; default 1 OpenMP / Numba thread)
python -m distance_eval.harness \
  --manifest distance_eval/results/unified_manifest.csv \
  --output distance_eval/results/unified_adc_timing.csv \
  --n_runs 5 \
  --num_threads 1

# 3) Plot (optional)
python -m distance_eval.figures \
  --timing_csv distance_eval/results/unified_adc_timing.csv \
  --output distance_eval/results/unified_adc_pareto.png
```

`python scripts/figure_generators/all_methods_relerr_figures.py --unified-adc` will emit the Pareto PNG if `unified_adc_timing.csv` exists.

## Threading

Use **`--num_threads 1`** (default) when comparing `time.process_time()` across runs; otherwise multi-threaded Numba/FAISS kernels can make process-time misleading.

## QINCo2 manifest fields

Rows need **`model_path`** (from eval CSV), **`codes_npz`** (globbed under the results root), and **`config_path`** (defaults to `lib/Qinco/config/qinco_cfg.yaml` when present). The harness writes a temporary `.fvecs` slice of the evaluation database so `QincoEvalTask` can load the model.

## RaBitQ

`RaBitQBackend` is a stub until `lib/RaBitQ-Library` is built and exposed (ctypes / pybind11). The harness **skips** `RaBitQ` rows with a short message.

## Adding a method

1. Subclass `ADCBackend` in `distance_eval/backends/yourmethod.py`.
2. Register in `distance_eval/registry.py` → `_ensure_registry()`.
3. Extend `distance_eval/manifest.py` to fill `artifact_index` / extra columns from your CSV.
4. Extend `distance_eval/harness._make_backend` if you need non-standard kwargs.

## Legacy PQ-only timing

PQ-only scripts under `scripts/evals/measure_adc_cpu_time.py` and `measure_all_adc.py` are **deprecated**; use this package for cross-method timing.
