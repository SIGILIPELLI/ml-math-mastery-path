# 07 · Expectation & Variance

Expectation and variance summarize a random variable with two numbers: where
it's centered, and how spread out it is. Loss functions are expectations
over data; variance underlies regularization, bias-variance tradeoff, and
weight-initialization schemes.

## Expectation

For a discrete random variable $X$ taking values $x_i$ with probabilities
$p_i$, the **expected value** (mean) is

$$
E[X] = \sum_i x_i\, p_i
$$

Expectation is **linear**: for constants $a,b$ and random variables $X,Y$:

$$
E[aX+b] = aE[X]+b \qquad E[X+Y] = E[X]+E[Y]
$$

(the second holds even when $X,Y$ are dependent).

## Variance

Variance measures spread around the mean:

$$
\text{Var}(X) = E\big[(X-E[X])^2\big] = E[X^2] - (E[X])^2
$$

The second form (expand the square and use linearity of $E$) is usually
easier to compute. Standard deviation is $\sigma = \sqrt{\text{Var}(X)}$.

**Scaling rule:** $\text{Var}(aX+b) = a^2\text{Var}(X)$ (shifting by $b$
doesn't change spread; scaling by $a$ scales variance by $a^2$).

## Worked example

A fair die roll $X \in \{1,2,3,4,5,6\}$, each with probability $1/6$.

**Expectation:**

$$
E[X] = \frac{1+2+3+4+5+6}{6} = \frac{21}{6} = 3.5
$$

**$E[X^2]$:**

$$
E[X^2] = \frac{1+4+9+16+25+36}{6} = \frac{91}{6} \approx 15.1\overline{6}
$$

**Variance:**

$$
\text{Var}(X) = \frac{91}{6} - (3.5)^2 = 15.1\overline{6} - 12.25 = 2.91\overline{6}
$$

**Standard deviation:** $\sigma = \sqrt{2.9167} \approx 1.7078$.

## Why this matters for ML

* Training loss is $E[\ell(\hat y, y)]$ over the data distribution —
  minimizing expected loss is the entire point of training.
* **Bias–variance decomposition** (Level 3) splits expected test error into
  bias$^2$ + variance + irreducible noise, directly using these definitions.
* Feature scaling/normalization uses mean and variance (standardize to mean
  0, variance 1) so gradient descent converges evenly across features.

## Numeric verification

```python
import numpy as np

outcomes = np.array([1, 2, 3, 4, 5, 6])
probs = np.full(6, 1/6)

E_X = np.sum(outcomes * probs)
E_X2 = np.sum(outcomes**2 * probs)
Var_X = E_X2 - E_X**2

print("E[X]      :", E_X)
print("E[X^2]    :", E_X2)
print("Var(X)    :", Var_X)
print("std(X)    :", np.sqrt(Var_X))

# Cross-check with a large simulation
rng = np.random.default_rng(0)
samples = rng.integers(1, 7, size=5_000_000)
print("simulated mean:", samples.mean())
print("simulated var :", samples.var())
```

Expected output:

```text
E[X]      : 3.5
E[X^2]    : 15.166666666666666
Var(X)    : 2.916666666666668
std(X)    : 1.707825127659933
simulated mean: 3.500...
simulated var : 2.916...
```

## Exercise

A biased coin lands heads ($X=1$) with probability $0.7$ and tails ($X=0$)
with probability $0.3$.

1. Compute $E[X]$ and $E[X^2]$ by hand (note for a 0/1 variable,
   $X^2 = X$, so this simplifies).
2. Compute $\text{Var}(X)$ using the shortcut formula.
3. Let $Y = 10X + 5$. Compute $E[Y]$ and $\text{Var}(Y)$ using the linearity
   and scaling rules — don't recompute from scratch.
4. Verify all four results with a NumPy simulation of 1,000,000 coin flips.
