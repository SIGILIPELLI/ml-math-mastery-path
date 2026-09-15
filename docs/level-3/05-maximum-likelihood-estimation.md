---
description: "Maximum Likelihood Estimation — Most loss functions in ML aren't arbitrary — they fall out of maximum likelihood estimation (MLE): choose the parameters…"
---

# 05 · Maximum Likelihood Estimation

Most loss functions in ML aren't arbitrary — they fall out of maximum
likelihood estimation (MLE): choose the parameters that make the observed
data most probable under an assumed distribution.

## The likelihood

Given data $\{x_i\}_{i=1}^n$ assumed i.i.d. from a distribution with
parameters $\theta$, the likelihood is:

$$
\mathcal{L}(\theta) = \prod_{i=1}^n p(x_i \mid \theta)
$$

Products underflow numerically, so we maximize the **log-likelihood**
instead (log is monotonic, same maximizer):

$$
\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)
$$

MLE: $\hat\theta = \arg\max_\theta \ell(\theta)$, equivalently
$\arg\min_\theta -\ell(\theta)$ — **negative log-likelihood (NLL)** is a
loss function.

## MSE is MLE under a Gaussian noise model

Assume $y_i = f_\theta(x_i) + \varepsilon_i$ with $\varepsilon_i \sim
\mathcal{N}(0,\sigma^2)$. Then $p(y_i\mid x_i,\theta) =
\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\right)$.

$$
-\ell(\theta) = \sum_i \left[\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2} + \tfrac12\log(2\pi\sigma^2)\right]
$$

Dropping constants independent of $\theta$, minimizing $-\ell$ is exactly
minimizing $\sum_i (y_i - f_\theta(x_i))^2$ — **squared error loss**.

## Cross-entropy is MLE under a Bernoulli/categorical model

For binary labels $y_i \in \{0,1\}$ with predicted probability $\hat
p_i=f_\theta(x_i)$: $p(y_i\mid x_i,\theta) = \hat p_i^{y_i}(1-\hat
p_i)^{1-y_i}$.

$$
-\ell(\theta) = -\sum_i \left[y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)\right]
$$

That sum is exactly binary cross-entropy — the standard classification
loss is negative log-likelihood of a Bernoulli model.

## Worked numeric example

Estimate the MLE of $\mu$ for Gaussian data $\{2, 4, 4, 6\}$ with known
$\sigma^2=1$. Setting $d\ell/d\mu = \sum_i(x_i-\mu) = 0$ gives the familiar
result $\hat\mu = \bar x$:

$$
\hat\mu = \frac{2+4+4+6}{4} = 4.0
$$

## Numeric verification

```python
import numpy as np
from scipy.optimize import minimize_scalar

data = np.array([2.0, 4.0, 4.0, 6.0])
sigma = 1.0

def neg_log_likelihood(mu):
    return -np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - (data - mu)**2 / (2 * sigma**2))

result = minimize_scalar(neg_log_likelihood)
print(f"closed-form MLE mu = {data.mean():.4f}")
print(f"numerically optimized mu = {result.x:.4f}")

# Cross-entropy = NLL of Bernoulli, sanity check
y = np.array([1, 0, 1, 1])
p_hat = np.array([0.9, 0.2, 0.8, 0.6])
bce = -np.mean(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
print(f"binary cross-entropy = {bce:.4f}")
```

```text
closed-form MLE mu = 4.0000
numerically optimized mu = 4.0000
binary cross-entropy = 0.3199
```

## How It Actually Works

Maximizing the likelihood $L(\theta) = \prod_i p(x_i;\theta)$ directly is a
numerical non-starter for any real dataset: multiplying even a few hundred
probabilities (each $\leq 1$, typically much less than 1 for continuous
densities) underflows float64 to exactly `0.0` almost immediately, at
which point the "likelihood" carries zero information about which $\theta$
is better. This is precisely why every practical implementation maximizes
the **log-likelihood** instead,
$\ell(\theta) = \sum_i \log p(x_i;\theta)$ — mathematically equivalent
(since $\log$ is monotonic, the maximizing $\theta$ is identical) but
numerically stable, because summing many log-probabilities (each a
moderate negative number) stays comfortably within float64's range where
their product would not.

This also determines *how* MLE is actually solved computationally: for
most models (anything beyond simple closed-form cases like a Gaussian's
mean and variance), there is no algebraic solution to
$\nabla_\theta \ell(\theta) = 0$, so real code runs gradient-based
optimization — computing $\nabla_\theta \ell(\theta)$ via autodiff through
the log-density expression directly (never through $\log(p(\cdot))$ after
$p$ has been computed, for the underflow reasons from Level 2 Module 09),
and applying gradient ascent (or, equivalently, gradient descent on the
negative log-likelihood, which is the actual "loss function" minimized when
you train a classifier with cross-entropy — cross-entropy loss *is*
negative log-likelihood under a categorical model, computed via the fused,
numerically stable softmax-cross-entropy kernel from Module 06.

## Exercise

1. Derive the MLE for the variance $\sigma^2$ of a Gaussian given data
   $\{2,4,4,6\}$ with $\mu$ fixed at the sample mean. Verify against
   `np.var(data)`.
2. Show that minimizing categorical cross-entropy over $C$ classes is MLE
   under a categorical (multinoulli) distribution — write out $-\ell$ for
   one-hot labels.
3. Explain, using the log-likelihood view, why adding an L2 penalty
   $\lambda\|\theta\|^2$ to the loss corresponds to placing a Gaussian
   *prior* on $\theta$ (this is MAP estimation, not plain MLE).
