# 09 · Numerical Stability in ML Math

Correct math on paper can still fail in floating point — overflow,
underflow, and catastrophic cancellation are the usual suspects. This
module covers the handful of tricks that keep production ML numerically
sound.

## Float overflow/underflow

`float32` overflows above ≈$3.4\times10^{38}$ (becomes `inf`) and
underflows below ≈$1.2\times10^{-38}$ (becomes 0). $e^{100}$ alone already
overflows `float32`.

## The log-sum-exp trick

Softmax needs $\sum_j e^{z_j}$, which overflows for large logits. Subtract
the max first — mathematically identical, numerically safe:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i-c}}{\sum_j e^{z_j-c}} \quad \text{for any constant } c
$$

Choosing $c=\max_j z_j$ makes every exponent $\le 0$, so every term is in
$(0,1]$ — no overflow, and the largest term is exactly $e^0=1$.

For log-probabilities directly, the **log-sum-exp** identity avoids ever
forming the sum in an unstable range:

$$
\log\sum_j e^{z_j} = c + \log\sum_j e^{z_j-c}, \qquad c=\max_j z_j
$$

## Cross-entropy: don't compute log(softmax(z)) in two steps

Computing $p=\text{softmax}(z)$ then $\log p$ risks $\log(0)$ if some
$p_i$ underflows to exactly 0. Fusing them:

$$
\log p_k = z_k - \log\sum_j e^{z_j} = z_k - \left(c + \log\sum_j e^{z_j-c}\right)
$$

avoids ever materializing a probability that could be exactly zero.

## Catastrophic cancellation

Subtracting two nearly-equal large floats destroys precision: computing
variance as $\frac{1}{n}\sum x_i^2 - \bar x^2$ can go **negative** due to
rounding when values are large and variance is small. The two-pass formula
$\frac{1}{n}\sum(x_i-\bar x)^2$ is more stable because it never subtracts
two large near-equal numbers.

## Worked numeric example

Logits $z = (1000, 1001, 999)$. Naive softmax:

$$
e^{1000} = \text{overflow} \to \text{inf}, \qquad \text{inf}/\text{inf} = \text{NaN}
$$

Stable version with $c=1001$: exponents become $(-1,0,-2)$, giving
$e^{-1}=0.3679,\ e^0=1,\ e^{-2}=0.1353$, sum$=1.5032$, so
$p\approx(0.2447, 0.6652, 0.0900)$ — a valid distribution.

## Numeric verification

```python
import numpy as np

z = np.array([1000.0, 1001.0, 999.0])

# Naive (unstable)
with np.errstate(over='ignore', invalid='ignore'):
    e_naive = np.exp(z)
    p_naive = e_naive / e_naive.sum()
print(f"naive softmax = {p_naive}")  # nan, nan, nan

# Stable
c = z.max()
e_stable = np.exp(z - c)
p_stable = e_stable / e_stable.sum()
print(f"stable softmax = {p_stable}")
print(f"sums to = {p_stable.sum():.6f}")

# Variance stability comparison
x = np.array([1e8 + 1, 1e8 + 2, 1e8 + 3])
naive_var = np.mean(x**2) - np.mean(x)**2       # cancellation-prone
stable_var = np.mean((x - x.mean())**2)          # two-pass, stable
print(f"naive var = {naive_var}")   # may be wildly wrong / negative
print(f"stable var (matches np.var) = {stable_var}, np.var = {x.var()}")
```

```text
naive softmax = [nan nan nan]
stable softmax = [0.24472847 0.66524096 0.09003057]
sums to = 1.000000
naive var = ...  (unstable, sensitive to platform — may print 0.0 or a
                   wrong nonzero value due to float64 precision limits)
stable var (matches np.var) = 0.6666666666666666, np.var = 0.6666666666666666
```

## Exercise

1. Implement `log_softmax(z)` using the log-sum-exp trick and verify
   `np.exp(log_softmax(z))` matches the stable softmax above to 6 decimal
   places.
2. Show that gradient clipping (capping $\|\nabla L\|$ at a threshold
   before the update) prevents the "exploding gradient" failure mode in
   RNNs, using a toy example where an unclipped update overshoots badly.
3. Explain why `1e-8` (or similar small $\epsilon$) is added inside square
   roots and denominators throughout ML code (BatchNorm, Adam, etc.) rather
   than checking for exact zero.
