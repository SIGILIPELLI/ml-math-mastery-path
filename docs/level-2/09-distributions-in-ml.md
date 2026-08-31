# 09 · Distributions Used in ML

A **probability distribution** describes how likely each outcome of a random
variable is. A handful of named distributions recur throughout ML — this
module covers the ones you'll meet immediately: Bernoulli, Binomial, and
Gaussian (Normal).

## Bernoulli distribution

Models a single binary trial (success/failure) with success probability
$p$. $X \in \{0,1\}$ with

$$
P(X=1) = p \qquad P(X=0) = 1-p
$$

$$
E[X] = p \qquad \text{Var}(X) = p(1-p)
$$

This is exactly the distribution behind a single logistic-regression
prediction / binary classification label.

## Binomial distribution

The number of successes $X$ in $n$ independent Bernoulli($p$) trials:

$$
P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}, \qquad k=0,\dots,n
$$

$$
E[X] = np \qquad \text{Var}(X) = np(1-p)
$$

(A Binomial is a sum of $n$ i.i.d. Bernoullis, so its mean/variance are $n$
times the single-trial values, by linearity of expectation and
independence.)

## Gaussian (Normal) distribution

A continuous distribution with density

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

parameterized by mean $\mu$ and variance $\sigma^2$. It's the most common
distribution in ML: the Central Limit Theorem says sums/averages of many
independent effects tend toward Gaussian, so noise in measurements, weight
initializations, and many likelihood models assume Gaussian errors.
Assuming Gaussian-distributed errors and maximizing likelihood (Level 3)
is exactly what derives the mean-squared-error loss.

## Worked example: Binomial

Flip a fair coin ($p=0.5$) $n=10$ times. Probability of exactly $k=6$ heads:

$$
P(X=6) = \binom{10}{6}(0.5)^6(0.5)^4 = 210 \cdot (0.5)^{10} = \frac{210}{1024} \approx 0.2051
$$

**Mean and variance:** $E[X] = 10(0.5) = 5$, $\text{Var}(X)=10(0.5)(0.5)=2.5$.

## Worked example: Gaussian density value

For a standard normal ($\mu=0,\sigma=1$), the density at $x=1$:

$$
f(1) = \frac{1}{\sqrt{2\pi}} e^{-1/2} \approx 0.3989 \times 0.6065 \approx 0.2420
$$

## Numeric verification

```python
import numpy as np
from scipy.stats import binom, norm

# Binomial
n, p, k = 10, 0.5, 6
print("P(X=6) binomial:", binom.pmf(k, n, p))
print("Binomial mean, var:", binom.mean(n, p), binom.var(n, p))

# Gaussian density at x=1 for standard normal
print("standard normal pdf at x=1:", norm.pdf(1, loc=0, scale=1))

# Cross-check binomial via direct simulation of coin flips
rng = np.random.default_rng(0)
trials = rng.integers(0, 2, size=(2_000_000, 10)).sum(axis=1)
print("simulated P(X=6):", (trials == 6).mean())
print("simulated mean, var:", trials.mean(), trials.var())
```

Expected output:

```text
P(X=6) binomial: 0.2050781250000001
Binomial mean, var: 5.0 2.5
standard normal pdf at x=1: 0.24197072451914337
simulated P(X=6): 0.2050...
simulated mean, var: 5.000... 2.499...
```

## Exercise

Let $X \sim \text{Binomial}(n=20, p=0.3)$.

1. Compute $P(X=5)$ by hand using the binomial formula (show the
   combinatorial coefficient).
2. Compute $E[X]$ and $\text{Var}(X)$ using the shortcut formulas.
3. Using `scipy.stats.norm`, evaluate the density of a Gaussian with
   $\mu=5, \sigma=2$ at $x=7$ and at $x=5$ — explain why the density is
   higher at $x=5$.
4. Verify your binomial answer from step 1 with `scipy.stats.binom.pmf` and
   with a NumPy simulation of at least 1,000,000 trials of 20 coin flips
   each with $p=0.3$.
