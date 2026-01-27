# PQ Evaluation C++ Implementation

This directory contains a C++ implementation of Product Quantization (PQ) evaluation that replicates the functionality of `scripts/rel_error_exps/PQ.py`.

## Files

- `pq_eval.cpp` - Main program implementing PQ training, encoding, and distance computation
- `io_utils.h` / `io_utils.cpp` - File I/O utilities for reading .fvecs and .bin datasets
- `Makefile` - Build configuration

## Building

### Prerequisites

1. **FAISS C++ library** - Must be built and available (see `INSTALL_FAISS.md` for instructions)
2. **OpenBLAS** - Required by FAISS (can be installed via conda: `conda install -c conda-forge openblas`)
3. **C++17 compiler** - GCC 7+ or Clang 5+

### Build Steps

1. Verify FAISS installation:
   ```bash
   make check
   ```

2. Build the program:
   ```bash
   make
   ```

3. If FAISS is installed in a different location, override paths:
   ```bash
   make FAISS_INCLUDE=/path/to/faiss/include \
        FAISS_LIB_PATH=/path/to/faiss/lib
   ```

## Usage

The program has the same command-line interface as the Python version:

```bash
./pq_eval \
    --dataset_path /path/to/database.fvecs \
    --query_path /path/to/queries.fvecs \
    --train_path /path/to/training.fvecs \
    [--dim DIM] \
    [--dataset_name NAME] \
    [--data_root ROOT] \
    [--n_subquantizers M] \
    [--nbits BITS] \
    [--train_size SIZE] \
    [--sample_db SIZE] \
    [--sample_queries SIZE] \
    [--results_dir DIR] \
    [--load_model] \
    [--model_path PATH]
```

### Required Arguments

- `--dataset_path`: Path to database/test dataset (.fvecs or .bin)
- `--query_path`: Path to query dataset (.fvecs or .bin)
- `--train_path`: Path to training dataset (.fvecs or .bin)

### Optional Arguments

- `--dim`: Dimensionality (required for .bin files if not auto-detected)
- `--dataset_name`: Dataset name for results (default: "custom")
- `--data_root`: Root path for saving results (default: "/data/cpanourg/2-hdvc/")
- `--n_subquantizers`: Number of subquantizers M (default: 32)
- `--nbits`: Number of bits per subquantizer (default: 8)
- `--train_size`: Number of training vectors (default: 100000)
- `--sample_db`: Number of database vectors to sample for evaluation (default: 10000)
- `--sample_queries`: Number of queries to sample for evaluation (default: 1000)
- `--results_dir`: Results subdirectory under data_root (default: "results/relerr")
- `--load_model`: Load a pre-trained model instead of training
- `--model_path`: Path to saved PQ model file (required if --load_model is set)

### Example

```bash
./pq_eval \
    --dataset_path /data/datasets/sift1M/sift_base.fvecs \
    --query_path /data/datasets/sift1M/sift_query.fvecs \
    --train_path /data/datasets/sift1M/sift_learn.fvecs \
    --dataset_name sift1M \
    --n_subquantizers 32 \
    --nbits 8 \
    --train_size 100000 \
    --sample_db 10000 \
    --sample_queries 1000
```

## Output

The program generates the same outputs as the Python version:

1. **PQ Model** (if training): `pq_model.index` in the experiment folder
2. **Relative Error Binary**: `rel_error_subq{M}_nbits{BITS}_db{N}k_qr{M}k.bin`
3. **Summary CSV**: `summary.csv` in the experiment folder
4. **Main Results CSV**: `{dataset_name}_PQ_adc_vs_exact_eval.csv` in the results directory

## Differences from Python Version

- Uses FAISS C++ API directly (no Python bindings)
- Faster execution due to compiled code
- Same file formats supported (.fvecs and .bin)
- Same output format and structure

## Troubleshooting

### "Cannot find libfaiss"
- Ensure FAISS is built: `cd lib/faiss && make -C build -j$(nproc)`
- Check paths: `make check`

### "Cannot find OpenBLAS"
- Install via conda: `conda install -c conda-forge openblas`
- Or set `CONDA_PREFIX` in Makefile if using a different conda installation

### Runtime errors with file I/O
- Ensure input files exist and are readable
- Check file format (.fvecs vs .bin)
- For .bin files, dimension must be specified with `--dim`

