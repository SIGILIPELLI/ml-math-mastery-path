# 10 · Capstone — Probability & Optimization Mini-Project

This capstone combines everything from Level 2: gradient descent, the chain
rule, and probability/distributions — by fitting a simple probabilistic
model (a Bernoulli/logistic model) to data using gradient descent, and
verifying every piece by hand and in NumPy.

## The setup

We observe $n$ coin flips $y_1,\dots,y_n \in \{0,1\}$ and want to estimate
the probability of heads, $p$. Model each flip as Bernoulli($p$)
(Module 9). Instead of the closed-form estimate (the sample mean), we'll
**find $p$ with gradient descent** on the negative log-likelihood — the same
strategy used to train logistic regression.

## Negative log-likelihood

The likelihood of the whole dataset (i.i.d. flips) is
$L(p) = \prod_i p^{y_i}(1-p)^{1-y_i}$. Taking logs turns the product into a
sum (much easier to differentiate):

$$
\ell(p) = \sum_i \big[y_i\ln p + (1-y_i)\ln(1-p)\big]
$$

We minimize the **negative** log-likelihood $J(p) = -\ell(p)$.

## Deriving the gradient

$$
\frac{dJ}{dp} = -\sum_i\left[\frac{y_i}{p} - \frac{1-y_i}{1-p}\right]
$$

Let $n_1 = \sum_i y_i$ (count of heads) and $n_0 = n-n_1$ (count of tails):

$$
\frac{dJ}{dp} = -\frac{n_1}{p} + \frac{n_0}{1-p}
$$

Setting this to zero (the exact, closed-form optimum) gives
$n_1(1-p) = n_0 p \Rightarrow p^\star = n_1/n$ — the sample mean, exactly as
expected. Gradient descent should converge to this same value.

## Worked example

Suppose $n=10$ flips with $n_1=7$ heads, $n_0=3$ tails. The true optimum is
$p^\star = 0.7$.

**Gradient at $p=0.5$:**

$$
\frac{dJ}{dp}\Big|_{p=0.5} = -\frac{7}{0.5}+\frac{3}{0.5} = -14+6=-8
$$

**One gradient descent step** with $\alpha=0.02$:

$$
p_1 = 0.5 - 0.02(-8) = 0.5+0.16=0.66
$$

Already close to $0.7$ after a single step; repeated steps converge to
$0.7$ (verified below).

## Full implementation and verification

```python
import numpy as np

n1, n0 = 7, 3
n = n1 + n0
p_star = n1 / n  # closed-form optimum

def dJ_dp(p):
    return -n1 / p + n0 / (1 - p)

p = 0.5
alpha = 0.02
history = [p]
for step in range(200):
    grad = dJ_dp(p)
    p = p - alpha * grad
    p = np.clip(p, 1e-6, 1 - 1e-6)  # keep p in valid (0,1) range
    history.append(p)

print("closed-form optimum p*:", p_star)
print("gradient descent p after 1 step :", history[1])
print("gradient descent p after 200 steps:", history[-1])
```

Expected output (gradient descent converges to the closed-form MLE):

```text
closed-form optimum p*: 0.7
gradient descent p after 1 step : 0.66
gradient descent p after 200 steps: 0.69999...
```

## Putting the level together

This one exercise touches every Level 2 module: probability and Bernoulli
(6, 9), expectation used implicitly in the likelihood (7), Bayes-style
reasoning about what the data implies about $p$ (8), the chain rule used
inside the derivative of $\ln p$ and $\ln(1-p)$ (2), and gradient descent
itself (1) — this is precisely the recipe Level 3 will scale up to full
logistic regression and neural networks.

## Exercise

Using $n_1=15, n_0=5$ (so $n=20$, $p^\star=0.75$):

1. Derive $dJ/dp$ symbolically (same form as above, different counts).
2. By hand, compute the gradient at $p=0.5$ and the value of $p$ after one
   gradient descent step with $\alpha=0.01$.
3. Implement the full gradient descent loop in NumPy for 300 steps and
   confirm it converges to $p^\star = 0.75$.
4. Plot (or print every 50 steps) how $p$ approaches $p^\star$, and describe
   in one sentence how the convergence rate would change if $\alpha$ were
   doubled.
