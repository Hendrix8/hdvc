# ADC Timing Experiments

This directory contains tools to measure Asymmetric Distance Computation (ADC) timing for trained PQ models and add the results to CSV files.

## Overview

The `pq_adc_timing` program:
1. Reads a CSV file with PQ experiment results
2. For each row, loads the trained PQ model
3. Runs ADC experiments using the first 1k queries and first 10k base vectors
4. Measures computation time and normalizes to **time per pair** (query-database pair)
5. Adds two new columns to the CSV:
   - `adc_pp`: ADC computation time per pair (seconds)
   - `distance_table_time_pp`: Distance table computation time per pair (seconds)

## Fair Timing Measurement

To ensure fair and accurate timing measurements, the program uses several techniques:

### 1. CPU Time Measurement (`CLOCK_PROCESS_CPUTIME_ID`)

Instead of wall-clock time, we use **CPU time** which measures only the time the process spends executing on the CPU. This excludes:
- Time waiting for I/O
- Time the process is suspended (sleep, context switches)
- Interference from other processes competing for CPU

**Why this is fair:**
- CPU time gives a consistent measure of actual computation time
- It's not affected by system load or other processes running on the machine
- Multiple runs will give similar results even if the system is busy

**Limitation:** CPU time doesn't account for memory bandwidth contention, but for CPU-bound operations like ADC, this is the most accurate measure.

### 2. Process Priority (`nice`)

The run script uses `nice -n -10` to give the process higher priority:
- Reduces likelihood of being preempted by lower-priority processes
- Still allows system processes to take priority when needed
- Doesn't require root privileges

### 3. Normalization to Time Per Pair

Results are normalized by dividing total time by the number of query-database pairs:
- `adc_pp = adc_time / (n_sample_q * n_sample_db)`
- `distance_table_time_pp = distance_table_time / (n_sample_q * n_sample_db)`

This makes results comparable across different sample sizes and allows you to estimate time for any number of pairs.

## Building

```bash
cd /home/cpanourg/projects/2-hdvc/gpu_cpp_scripts/pq
make pq_adc_timing
```

Or build everything:
```bash
make all
```

## Usage

### Basic Usage

```bash
./run_adc_timing.sh <csv_file> [n_sample_q] [n_sample_db]
```

**Arguments:**
- `csv_file`: Path to CSV file (e.g., `deep_PQ_adc_vs_exact_eval.csv`)
- `n_sample_q`: Number of queries to use (default: 1000)
- `n_sample_db`: Number of database vectors to use (default: 10000)

### Examples

```bash
# Use default 1k queries and 10k database vectors
./run_adc_timing.sh /data/cpanourg/2-hdvc/results/relerr_cpp/deep_PQ_adc_vs_exact_eval.csv

# Custom sample sizes
./run_adc_timing.sh /data/cpanourg/2-hdvc/results/relerr_cpp/gist_PQ_adc_vs_exact_eval.csv 500 5000
```

### Direct Program Usage

You can also run the program directly:

```bash
./pq_adc_timing <csv_path> <data_root> <n_sample_q> <n_sample_db>
```

Example:
```bash
./pq_adc_timing \
  /data/cpanourg/2-hdvc/results/relerr_cpp/deep_PQ_adc_vs_exact_eval.csv \
  /data/cpanourg/2-hdvc \
  1000 \
  10000
```

## CSV Format

The program expects a CSV file with the following columns:
- `method`: Method name (e.g., "PQ")
- `dataset`: Dataset name (e.g., "deep", "gist", "bigann")
- `experiment_folder`: Folder name containing the trained model
- `n_subquantizers`: Number of subquantizers (M)
- `nbits`: Bits per subvector

The program will add (or update) these columns:
- `adc_pp`: ADC time per pair (seconds)
- `distance_table_time_pp`: Distance table computation time per pair (seconds)

## How It Works

1. **Read CSV**: Loads the CSV file and parses all rows
2. **Check existing results**: Skips rows that already have `adc_pp` values
3. **For each row**:
   - Constructs paths to model, queries, and database based on dataset name
   - Loads the trained PQ model
   - Loads query and database vectors
   - Encodes database vectors using the PQ model
   - Computes distance tables for queries (measures time)
   - Computes ADC distances (measures time)
   - Normalizes times to per-pair values
   - Updates CSV row with results
   - Saves CSV after each successful measurement

4. **Incremental updates**: The CSV is saved after each measurement, so you can stop and resume the process

## Supported Datasets

- `deep`: Uses `deep1b/dataset/fvecs/query_10k.fvecs` and `test_1m.fvecs`
- `gist`: Uses `gist/gist_query.fvecs` and `gist_base.fvecs`
- `bigann`: Uses `bigann/sift1m/sift_query.fvecs` and `sift_base.fvecs`

To add more datasets, edit `pq_adc_timing.cpp` and add the dataset paths in the `main()` function.

## Notes

- The program uses CPU time (`CLOCK_PROCESS_CPUTIME_ID`) for accurate measurement
- Results are saved incrementally, so you can stop and resume
- Rows with existing `adc_pp` values are skipped (set to empty string to re-run)
- Failed measurements are marked as "N/A" in the CSV

## Troubleshooting

**Error: "Model not found"**
- Check that the `experiment_folder` path is correct
- Verify the model file exists at: `{data_root}/results/relerr_cpp/pq/{dataset}/{experiment_folder}/pq_model.index`

**Error: "Failed to load queries/database"**
- Check that dataset paths are correct in the code
- Verify the `.fvecs` files exist

**Timing seems inconsistent**
- CPU time should be consistent, but if you see large variations:
  - Check system load (use `htop` or `top`)
  - Ensure no other heavy processes are running
  - Consider running with `taskset` to pin to specific CPU cores
  - For even more isolation, use `chrt` with real-time scheduling (requires root)

## Advanced: Even More Isolation

For maximum isolation from other processes, you can:

1. **Pin to specific CPU cores**:
```bash
taskset -c 0-3 ./pq_adc_timing ...
```

2. **Use real-time scheduling** (requires root):
```bash
sudo chrt -f 50 ./pq_adc_timing ...
```

3. **Disable CPU frequency scaling**:
```bash
sudo cpupower frequency-set -g performance
```

However, for most use cases, CPU time measurement (`CLOCK_PROCESS_CPUTIME_ID`) is sufficient and doesn't require these additional steps.
