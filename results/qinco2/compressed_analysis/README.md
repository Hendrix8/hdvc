# Qinco2 Compressed Analysis

This folder contains analysis generated from compressed Qinco2 archives only.

Metric types:
- `distance_relerr`: paper-style relative distance error when exact/approx distances are available in archive npz.
- `mse_proxy`: fallback proxy from run logs (`validation MSE` / `min_MSE`) when distance arrays are unavailable.
- `bits_per_vector`: `M * log2(K)`.

Coverage summary:

- Archives processed: 217
- Settings processed: 242
- Distance-level relerr available: 1 settings
- Rows using mse_proxy metric: 0
- bigann: 1/47 settings distance-ready (2.1%)
- deep: 0/45 settings distance-ready (0.0%)
- gist: 0/50 settings distance-ready (0.0%)
- msmarco: 0/50 settings distance-ready (0.0%)
- openai: 0/50 settings distance-ready (0.0%)
