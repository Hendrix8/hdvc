# TurboQuant Quantization Benchmark

Benchmark comparing two TurboQuant quantization variants — **TQ-MSE** and **TQ-Prod** — on vector similarity search datasets, evaluating quantization quality from 1 to 12 bits per dimension.

## Methods

Both methods share the same preprocessing pipeline:

1. **Normalize:** decompose each vector as $x = \|x\| \cdot \hat{x}$, storing the norm $\|x\|$ separately.
2. **Random rotation:** apply a Haar-distributed random orthogonal matrix $\Pi$ so that each coordinate of $y = \Pi \hat{x}$ becomes approximately i.i.d. $\mathcal{N}(0, 1/d)$ for large $d$.
3. **Scalar quantization:** quantize each rotated coordinate $y_i$ independently using Lloyd-Max optimal centroids derived from the known Gaussian distribution (data-oblivious — no training data needed).

### TQ-MSE (MSE-optimal)

All $b$ bits per dimension are allocated to MSE-optimal scalar quantization.

**Encoding:** Each coordinate $y_i$ is mapped to the nearest centroid $c_{k_i}$ from a $2^b$-level codebook, producing a quantized vector $\hat{y}$.

**Inner product estimation** via full reconstruction:

$$\langle q, x \rangle_{\text{approx}} = \|x\| \cdot \langle q, \Pi^T \hat{y} \rangle = \|x\| \cdot \langle \Pi q, \hat{y} \rangle$$

where $\hat{y}$ is the vector of centroids looked up from quantization indices. This is equivalent to computing $\langle q, \tilde{x} \rangle$ where $\tilde{x} = \|x\| \cdot \Pi^T \hat{y}$ is the reconstructed vector.

**Properties:**
- Minimizes reconstruction MSE $\mathbb{E}[\|x - \tilde{x}\|^2]$.
- The IP estimate is **biased** — it systematically under-estimates inner products because quantization loses information.
- Storage: $b \cdot d$ bits (indices) + 32 bits (norm).

### TQ-Prod (IP-unbiased via QJL correction)

Allocates $(b-1)$ bits to MSE quantization and 1 bit to a **Quantized Johnson-Lindenstrauss (QJL)** residual correction, achieving an unbiased inner product estimator.

**Encoding:**
1. Quantize $\hat{x}$ with $(b-1)$-bit TQ-MSE, yielding reconstruction $\tilde{x}_{\text{mse}}$.
2. Compute the residual $r = \hat{x} - \tilde{x}_{\text{mse}}$ and store its norm $\|r\|$.
3. Project the residual through a random Gaussian matrix $S \in \mathbb{R}^{d \times d}$ and store only the signs: $s = \text{sign}(S \cdot r)$.

**Inner product estimation** with two-stage correction:

$$\langle q, x \rangle_{\text{approx}} = \|x\| \cdot \left( \underbrace{\langle \Pi q, \hat{y} \rangle}_{\text{MSE part}} + \underbrace{\sqrt{\frac{\pi}{2}} \cdot \frac{\|r\|}{d} \cdot \langle S q, s \rangle}_{\text{QJL residual correction}} \right)$$

The QJL term provides an unbiased estimate of $\langle q, r \rangle$ using only 1 bit per dimension, based on the Johnson-Lindenstrauss lemma: the sign of a random projection preserves inner product information in expectation.

**Properties:**
- Achieves $\mathbb{E}[\langle q, x \rangle_{\text{approx}}] = \langle q, x \rangle$ (unbiased IP estimation).
- Storage: $(b-1) \cdot d$ bits (MSE indices) + $d$ bits (QJL signs) + 32 bits (residual norm) + 32 bits (norm).
---

## Evaluation Metrics

All experiments use 10,000 base vectors and 1,000 query vectors (10M distance pairs). L2 distance estimation uses exact norms: $\hat{L}_2^2(q, x) = \|q\|^2 + \|x\|^2 - 2 \cdot \widehat{\langle q, x \rangle}$, so that the only source of approximation error is the inner product estimator.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Distortion** | $\frac{1}{n}\sum_i \|x_i - \tilde{x}_i\|^2$ | Mean squared reconstruction error. Measures how well the quantized vector approximates the original. Lower is better. |
| **L2-RelErr** (mean) | $\frac{1}{n \cdot m}\sum_{i,j} \frac{\lvert \hat{L}_2^2(q_j, x_i) - L_2^2(q_j, x_i) \rvert}{L_2^2(q_j, x_i)}$ | Mean relative error of squared L2 distance estimation. Lower means more accurate distance computation. |
| **IP-RelErr** (mean) | $\frac{1}{n \cdot m}\sum_{i,j} \frac{\lvert \widehat{\langle q_j, x_i \rangle} - \langle q_j, x_i \rangle \rvert}{\lvert \langle q_j, x_i \rangle \rvert}$ | Mean relative error of inner product estimation. Lower means more accurate IP computation. |
| **L2-R@K** | $P(\text{true L2-NN} \in \text{top-}K \text{ by approx L2})$ | Recall@K under L2 distance: probability that the exact nearest neighbor appears in the top-K results from approximate ranking. Higher is better. |
| **IP-R@K** | $P(\text{true IP-NN} \in \text{top-}K \text{ by approx IP})$ | Recall@K under inner product: probability that the exact maximum-IP neighbor appears in the top-K from approximate ranking. Higher is better. |

Median variants of RelErr are also reported in the CSV for robustness against outliers.

---

## Datasets

| Dataset | Dimensionality | Domain |
|---------|---------------|--------|
| **SIFT** | 128 | SIFT local image descriptors |
| **Deep** | 96 | Deep learning embeddings (Yandex Deep1B) |
| **BigANN** | 128 | SIFT descriptors (BigANN benchmark) |
| **GIST** | 960 | GIST global image descriptors |
| **MS MARCO** | 1024 | Passage retrieval embeddings |
| **OpenAI** | 1536 | OpenAI text embeddings |

---

## Results Summary

### Key Findings

**1. TQ-MSE dominates at low bit-widths (1–6 bit/dim).**

At the same total bits per dimension, TQ-MSE consistently achieves lower distortion, lower relative error, and higher recall than TQ-Prod. This is because TQ-Prod sacrifices 1 bit to QJL correction, leaving only $(b-1)$ bits for MSE quantization — a significant penalty when the budget is small.

For example, on SIFT at 4 bit/dim:
- TQ-MSE: L2-R@1 = 0.505, Distortion = 2537
- TQ-Prod: L2-R@1 = 0.373, Distortion = 14468

**2. TQ-Prod catches up at high bit-widths (≥ 8 bit/dim).**

As the bit budget increases, the 1-bit QJL overhead becomes relatively minor, and TQ-Prod's unbiased IP estimation provides better ranking accuracy. On SIFT:
- At 9 bit/dim: TQ-Prod L2-R@1 = 0.933 vs TQ-MSE L2-R@1 = 0.906
- At 12 bit/dim: TQ-Prod L2-R@1 = 0.980 vs TQ-MSE L2-R@1 = 0.973

**3. Higher dimensionality helps both methods.**

The Gaussian approximation (which underlies the data-oblivious codebook) becomes more accurate as $d$ increases, leading to better quantization quality at the same bit-width. Comparing L2-R@1 at 4 bit/dim:
- Deep (d=96): 0.770
- SIFT (d=128): 0.505
- GIST (d=960): 0.612
- MS MARCO (d=1024): 0.888
- OpenAI (d=1536): 0.833

Note: SIFT/BigANN are unnormalized integer-valued descriptors with different distributional properties, so the dimension effect is not purely monotonic across all datasets.

**4. TQ-Prod's distortion is always higher, but its Recall@1 can be higher.**

This apparent paradox arises because distortion measures per-vector reconstruction quality, while recall measures ranking quality across the database. TQ-Prod's unbiased IP estimation produces better relative ordering of candidates despite larger per-element noise, especially at higher bit-widths where the noise variance is small enough that unbiasedness matters more than variance.

**5. Recall@10 and Recall@100 saturate quickly.**

For most datasets, Recall@10 reaches 1.0 by 6–7 bit/dim, and Recall@100 reaches 1.0 by 4–5 bit/dim. The main differentiator between methods is Recall@1, which continues to improve up to 12 bit/dim.

---

## Usage

```bash
python benchmark_quant.py --dataset sift --bits 1 2 3 4 5 6 7 8 9 10 11 12
```

Options:
- `--dataset`: one of `sift`, `deep`, `bigann`, `gist`, `msmarco`, `openai`
- `--bits`: list of bit-widths to test (e.g., `1 2 3 4`)
- `--nb`: number of base vectors (default: 10000)
- `--nq`: number of query vectors (default: 1000)
- `--no-prod`: skip TQ-Prod (only run TQ-MSE)
- `--no-rabitq`: skip RaBitQ
- `--output`: custom CSV output path (default: `result/quant_benchmark_{dataset}_nb{nb}_nq{nq}.csv`)

## Output

Results are saved as CSV files in `result/` 
