# VAQ experiment layouts (Deep)

## Scripts

| Script | Role |
|--------|------|
| `run_deep.sh` | Single run; set `METHOD` or `TOTAL_BITS` / `N_SUBSPACES` / … |
| `run_deep_pq_opq_style.sh` | Same data + train-size sweep as OPQ; one or more full `METHOD` strings |
| `run_deep_hyperparam_grid.sh` | **PQ/OPQ-style grid**: `M × nbits × train_size` → builds VAQ `METHOD` per cell |

## Alignment with OPQ (`run_opq.sh`)

| OPQ / PQ | VAQ grid script |
|----------|-----------------|
| `n_subquantizers` M ∈ {4, 8, 16, 32} | `N_SUBQUANTIZERS_LIST` (same default) |
| `nbits` ∈ {8, 9, 10} | `NBITS_LIST` |
| `train_size` ∈ {10k, 100k, 1M} | `TRAIN_SIZES` |
| `bits_per_vector` = M × nbits | Used as target for VAQ’s leading `TOTAL` (floored to at least `MIN_TOTAL_VAQ_BITS`, default 128) |
| `M` must divide dim (96) | Skips `M` if `96 % M != 0` |

VAQ still uses its own **per-subspace bit range** in the method string. Defaults (`MIN_BITS=7`, `MAX_BITS=8`) match the stable recipe from legacy `VAQ.py`; they are **not** set equal to OPQ’s `nbits` (the GLP step can fail on tight `min=nbits max=nbits` at low totals). Override `MIN_BITS`, `MAX_BITS` if your build tolerates other ranges.

## Quick start

```bash
cd /path/to/2-hdvc
chmod +x vaq/run_deep_hyperparam_grid.sh
./vaq/run_deep_hyperparam_grid.sh
```

Dry-run (print commands only):

```bash
DRY_RUN=1 ./vaq/run_deep_hyperparam_grid.sh
```

Smaller grid:

```bash
N_SUBQUANTIZERS_LIST="8 32" NBITS_LIST="8 10" TRAIN_SIZES="100000" ./vaq/run_deep_hyperparam_grid.sh
```

Wider M list (like a heavy PQ sweep):

```bash
PQ_GRID=1 ./vaq/run_deep_hyperparam_grid.sh
```
