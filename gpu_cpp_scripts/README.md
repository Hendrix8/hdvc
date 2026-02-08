# GPU C++ Scripts

GPU-accelerated versions of the PQ and OPQ evaluation scripts. These use FAISS's GPU-accelerated k-means assignment during Product Quantizer training for faster training.

## Prerequisites

1. **FAISS built with GPU support**: Rebuild FAISS with CUDA enabled:
   ```bash
   cd lib/faiss
   cmake -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF .
   make -C build -j$(nproc)
   ```

2. **CUDA toolkit** installed (e.g. `/usr/local/cuda`)

3. **OpenBLAS** (same as cpp_scripts)

## Build

### PQ
```bash
cd gpu_cpp_scripts/pq
make
# Or with custom paths:
make CUDA_HOME=/path/to/cuda FAISS_INCLUDE=/path/to/faiss FAISS_LIB_PATH=/path/to/faiss/build/faiss
```

### OPQ
```bash
cd gpu_cpp_scripts/opq
make
```

## Usage

Same as `cpp_scripts` with an additional optional `--gpu_device N` flag (default: 0):

```bash
# PQ
./pq_eval --dataset_path ... --query_path ... --train_path ... --gpu_device 0

# OPQ
./opq_eval --dataset_path ... --query_path ... --train_path ... --gpu_device 0
```

Set `GPU_DEVICE` env var when using the run scripts: `GPU_DEVICE=1 ./run_pq_eval.sh`

## What uses GPU?

- **PQ**: The Product Quantizer training uses `GpuIndexFlatL2` for k-means assignment, accelerating the nearest-centroid searches during clustering.
- **OPQ**: The inner PQ training (after OPQ rotation) uses the same GPU-accelerated assignment. OPQ matrix training remains on CPU (FAISS has no GPU implementation for that step).

Encoding, distance tables, and evaluation remain on CPU to keep outputs identical to the original cpp_scripts.
