# Analysis of Product Quantization (PQ) Hyperparameter Effects

## Executive Summary

Based on the experimental results, here are the key findings:

1. **`nbits` (bits per subspace) is THE most critical parameter** - Strong negative correlation (r=-0.40, p<0.01)
2. **`n_subquantizers` has weak direct correlation** but strong **interaction effects** with `nbits`
3. **Training size effect is complex** - helps most when quantization is poor
4. **Best configuration**: `n_subquantizers=16`, `nbits=8` (error=0.09)
5. **Worst configuration**: `n_subquantizers=16`, `nbits=4`, `train_size=100K` (error=2.00)

---

## 1. Correlation Analysis

| Hyperparameter | Correlation | p-value | Interpretation |
|----------------|-------------|---------|----------------|
| `nbits` | -0.40 | 0.009 | **Strong negative** - most important single parameter |
| `bits_per_vector` | -0.30 | 0.057 | Moderate negative |
| `train_size` | -0.13 | 0.40 | Weak, not significant |
| `n_subquantizers` | -0.007 | 0.96 | Very weak (but has interactions) |

---

## 2. Intuitive Understanding

### A. The Role of `nbits` (Bits per Subspace)

**This is the MOST CRITICAL parameter.**

- **nbits=4**: Mean error = 0.55 (VERY HIGH)
- **nbits=8**: Mean error = 0.16 (GOOD - sweet spot)
- **nbits=12/16**: Mean error = 0.18 (similar to nbits=8)

**Intuition**: 
- Product Quantization splits each vector into `n_subquantizers` subspaces
- Each subspace is quantized independently with `2^nbits` codewords
- With only 4 bits (16 codewords per subspace), the quantization is **too coarse** and loses critical information
- With 8+ bits (256+ codewords), there's **sufficient resolution** to represent the data distribution well
- Going beyond 8 bits provides **diminishing returns** - the subspace is already well-represented

**Think of it like**: 
- 4 bits = trying to represent a complex shape with only 16 points → too few, loses detail
- 8 bits = 256 points → sufficient to capture the shape
- 12+ bits = even more points → minimal improvement

---

### B. The Role of `n_subquantizers` (Number of Subquantizers)

**Alone, weak correlation, but STRONG interaction with `nbits`.**

**When nbits is LOW (4)** - more subquantizers can help:
- n_subquantizers=4, nbits=4: error=0.68
- n_subquantizers=8, nbits=4: error=0.31 (better!)
- n_subquantizers=12, nbits=4: error=0.33
- n_subquantizers=16, nbits=4: error=0.87 (worse - too fine splitting)

**When nbits is HIGH (8+)** - n_subquantizers matters less:
- n_subquantizers=4, nbits=8: error=0.23
- n_subquantizers=8, nbits=8: error=0.18
- n_subquantizers=12, nbits=8: error=0.13
- n_subquantizers=16, nbits=8: error=0.09 (best!)

**Intuition**:
- More subquantizers = **finer partitioning** of the vector space
- When each subspace has **few bits** (4), you need more subspaces to compensate
- When each subspace has **enough bits** (8+), the partitioning is less critical
- **Extreme case**: Too many subquantizers (16) with too few bits (4) → **instability**
  - The vector is split too finely, each subspace has too few codewords
  - Creates quantization artifacts and poor reconstruction

**Think of it like**:
- Splitting a pizza into 4 slices with 16 toppings each vs 16 slices with 4 toppings each
- The latter is too fragmented - each slice is too small to represent well

---

### C. The Role of `train_size` (Training Data Size)

**Complex effect that depends on other parameters.**

**For nbits=4 (poor quantization)**:
- train_size=10K: error=0.53
- train_size=100K: error=0.78 (worse!)
- train_size=100M: error=0.34 (better with lots of data)

**For nbits=8 (good quantization)**:
- train_size=10K: error=0.15
- train_size=100K: error=0.16
- train_size=100M: error=0.16 (stable across training sizes)

**Intuition**:
- When quantization is **coarse** (nbits=4), the codebook needs **more training data** to learn the data distribution
- With insufficient data (10K-100K), the codebook may overfit or underfit
- With **lots of data** (100M), even coarse quantization can learn better
- When quantization is **fine** (nbits=8+), the codebook is already well-learned with **minimal data** (10K is enough)
- This is why training size has minimal effect when nbits is high

**Think of it like**:
- Learning a language with 16 words (nbits=4) → need lots of examples to understand usage
- Learning a language with 256 words (nbits=8) → can learn patterns with fewer examples

---

### D. The Pathological Outlier

**Configuration**: `n_subquantizers=16`, `nbits=4`, `train_size=100K`  
**Error**: 2.00 (extremely high!)

This is a **pathological case** that demonstrates the interaction:
- 16 subquantizers × 4 bits = 16 subspaces × 16 codewords = 256 total codewords
- BUT: The vector is split **too finely** (16 pieces)
- Each piece is **too small** to quantize well with only 16 codewords
- Creates **instability** and poor quantization
- The intermediate training size (100K) may not be enough to stabilize this difficult configuration

---

## 3. Key Takeaways & Recommendations

### Most Important Findings:

1. **`nbits=8` is the sweet spot** - provides excellent quantization with minimal overhead
2. **Avoid `nbits=4`** - too coarse, causes high error regardless of other settings
3. **`n_subquantizers` matters most when `nbits` is low** - use more subquantizers to compensate
4. **Training size helps most when quantization is poor** - invest in more data if stuck with low nbits
5. **Best overall**: `n_subquantizers=16`, `nbits=8` achieves error=0.09
6. **Avoid extremes**: Very low nbits (4) or extreme combinations (many subquantizers + few bits)

### Practical Recommendations:

- **For best accuracy**: Use `nbits=8` or higher, `n_subquantizers=12-16`
- **For memory efficiency**: `nbits=8` is optimal (going higher has diminishing returns)
- **For limited training data**: Use `nbits=8+` (less sensitive to training size)
- **For abundant training data**: Can use `nbits=4` with more subquantizers, but `nbits=8` is still better

---

## 4. Mathematical Intuition

Product Quantization approximates a vector **v** as:
```
v ≈ [q₁(v₁), q₂(v₂), ..., qₘ(vₘ)]
```
where:
- **m** = `n_subquantizers` (number of subspaces)
- **qᵢ** = quantizer for subspace i with **2^nbits** codewords
- Total codewords = **m × 2^nbits**

**Key insight**: The **effective resolution** depends on both m and nbits, but:
- Increasing **nbits** (exponentially more codewords per subspace) is more effective
- Increasing **m** (linearly more subspaces) helps but has diminishing returns
- There's an **optimal balance** - too many subspaces with too few bits → instability
- The **sweet spot** is nbits=8 (256 codewords per subspace) with m=12-16

---

## 5. Why These Results Make Sense

1. **Information theory**: Each subspace needs sufficient bits to encode its information content
2. **Curse of dimensionality**: Splitting too finely (high m, low nbits) creates quantization artifacts
3. **Learning theory**: Coarse quantization (low nbits) needs more data to learn the distribution
4. **Practical trade-offs**: nbits=8 provides 256 codewords - enough for most distributions, not too many to overfit

The results show that **quality of quantization per subspace** (nbits) matters more than **number of subspaces** (n_subquantizers), which aligns with information-theoretic principles.

