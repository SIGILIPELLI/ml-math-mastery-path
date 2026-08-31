# 07 · Bayesian Inference Foundations

Bayesian inference treats parameters as random variables with a
distribution that gets updated as data arrives — contrasted with MLE
(Level 3 Module 05), which finds a single "best" point estimate.

## Bayes' theorem for parameters

$$
p(\theta \mid D) = \frac{p(D\mid\theta)\,p(\theta)}{p(D)}
$$

- $p(\theta)$: **prior** — belief about $\theta$ before seeing data.
- $p(D\mid\theta)$: **likelihood** — same quantity MLE maximizes.
- $p(\theta\mid D)$: **posterior** — updated belief after seeing data.
- $p(D) = \int p(D\mid\theta)p(\theta)\,d\theta$: **evidence**, a
  normalizing constant.

## MAP vs. MLE vs. full Bayesian

**MLE:** $\hat\theta = \arg\max_\theta p(D\mid\theta)$ — ignores prior.

**MAP (maximum a posteriori):** $\hat\theta = \arg\max_\theta
p(D\mid\theta)p(\theta)$ — a single point, but prior-informed. This is
where L2 regularization comes from: a Gaussian prior on weights, when you
take $-\log$ of $p(D\mid\theta)p(\theta)$, adds exactly a $\lambda\|w\|^2$
term to the loss (Level 3 Module 04's regularization, derived properly).

**Full Bayesian:** keep the entire posterior distribution $p(\theta\mid
D)$, not just its peak — captures uncertainty about $\theta$ itself.

## Worked example: Beta-Bernoulli conjugate update

Coin flips are Bernoulli($\theta$). A **Beta($\alpha,\beta$)** prior on
$\theta$ is *conjugate*: the posterior after observing $k$ heads in $n$
flips is again Beta, with a simple update:

$$
p(\theta) = \text{Beta}(\alpha,\beta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

$$
p(\theta\mid D) \propto \theta^k(1-\theta)^{n-k}\cdot\theta^{\alpha-1}(1-\theta)^{\beta-1} = \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}
$$

$$
p(\theta\mid D) = \text{Beta}(\alpha+k,\ \beta+n-k)
$$

Starting with a uniform prior Beta(1,1) and observing 7 heads in 10 flips
gives posterior Beta(8,4), with posterior mean
$\frac{8}{8+4}=0.6\overline{6}$ — close to, but pulled slightly toward 0.5
from, the raw MLE estimate $7/10=0.7$.

## Numeric verification

```python
import numpy as np
from scipy import stats

alpha_prior, beta_prior = 1, 1  # uniform prior
n, k = 10, 7  # 10 flips, 7 heads

alpha_post = alpha_prior + k
beta_post = beta_prior + (n - k)

mle_estimate = k / n
map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2) if alpha_post > 1 and beta_post > 1 else None
posterior_mean = alpha_post / (alpha_post + beta_post)

print(f"MLE estimate = {mle_estimate:.4f}")
print(f"posterior Beta({alpha_post},{beta_post})")
print(f"posterior mean = {posterior_mean:.4f}")

# Verify via direct numerical integration of Bayes' theorem
theta_grid = np.linspace(0.001, 0.999, 100000)
prior = stats.beta.pdf(theta_grid, alpha_prior, beta_prior)
likelihood = theta_grid**k * (1 - theta_grid)**(n - k)
unnormalized_posterior = prior * likelihood
posterior = unnormalized_posterior / np.trapz(unnormalized_posterior, theta_grid)
numeric_posterior_mean = np.trapz(theta_grid * posterior, theta_grid)
print(f"numerically integrated posterior mean = {numeric_posterior_mean:.4f}")
```

```text
MLE estimate = 0.7000
posterior Beta(8,4)
posterior mean = 0.6667
numerically integrated posterior mean = 0.6667
```

## Exercise

1. Repeat with a stronger prior Beta(10,10) (representing a strong belief
   the coin is fair) and the same data (7 heads/10 flips). How much does
   the posterior mean shift toward 0.5 compared to the uniform-prior case?
2. Derive, via $-\log$ of MAP's objective $p(D\mid\theta)p(\theta)$ for
   Gaussian likelihood and Gaussian prior on a weight $w$, that MAP
   estimation reduces exactly to L2-regularized least squares. Identify
   $\lambda$ in terms of the likelihood and prior variances.
3. Plot the posterior distribution as more coin flips arrive (5, 20, 100
   flips, same true $\theta=0.7$) and describe how it narrows —
   this is Bayesian uncertainty quantification in action.
