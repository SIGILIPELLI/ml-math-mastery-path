# 03 · Information Theory Basics

Entropy, cross-entropy, and KL divergence quantify "how surprising" or
"how different" distributions are — and they underlie the loss functions
used throughout classification and generative modeling.

## Entropy

Entropy measures the average "surprise" (information content) of a
distribution $p$:

$$
H(p) = -\sum_i p_i \log p_i
$$

$-\log p_i$ is the "surprise" of outcome $i$ (rare events, small $p_i$,
are more surprising). $H(p)$ is maximized by the uniform distribution
(maximum uncertainty) and is 0 for a distribution with all mass on one
outcome (no uncertainty at all).

## Cross-entropy

Cross-entropy measures the average surprise of outcomes drawn from true
distribution $p$, but measured using a *model's* distribution $q$:

$$
H(p,q) = -\sum_i p_i \log q_i
$$

This is exactly the classification loss from Module 06 of Level 3 — $p$ is
the one-hot true label, $q$ is the predicted distribution.

## KL divergence

The Kullback-Leibler divergence measures how much *extra* surprise you pay
for using $q$ instead of the true $p$:

$$
D_{KL}(p\|q) = \sum_i p_i \log\frac{p_i}{q_i} = H(p,q) - H(p)
$$

$D_{KL}\ge 0$ always (Gibbs' inequality), with equality iff $p=q$. Since
$H(p)$ doesn't depend on the model, minimizing cross-entropy loss is
*exactly* minimizing $D_{KL}(p\|q)$ — training a classifier is literally
minimizing the KL divergence from the true label distribution to the
model's predicted distribution.

## Worked numeric example

True distribution $p=(1,0,0)$ (one-hot), model prediction
$q=(0.7,0.2,0.1)$.

$$
H(p) = -1\log 1 - 0 - 0 = 0 \quad(\text{one-hot has zero entropy})
$$

$$
H(p,q) = -1\log(0.7) \approx 0.3567
$$

$$
D_{KL}(p\|q) = H(p,q)-H(p) = 0.3567 - 0 = 0.3567
$$

## Numeric verification

```python
import numpy as np

p = np.array([1.0, 0.0, 0.0])
q = np.array([0.7, 0.2, 0.1])

def entropy(p):
    # 0*log(0) := 0 by convention
    return -np.sum(np.where(p > 0, p * np.log(p), 0.0))

def cross_entropy(p, q):
    return -np.sum(np.where(p > 0, p * np.log(q), 0.0))

def kl_divergence(p, q):
    return np.sum(np.where(p > 0, p * np.log(p / q), 0.0))

Hp = entropy(p)
Hpq = cross_entropy(p, q)
KL = kl_divergence(p, q)

print(f"H(p) = {Hp:.4f}")
print(f"H(p,q) = {Hpq:.4f}")
print(f"KL(p||q) = {KL:.4f}")
print(f"H(p,q) - H(p) = {Hpq - Hp:.4f}  (should equal KL(p||q))")

# Compare a softer true distribution
p2 = np.array([0.7, 0.2, 0.1])
q2 = np.array([0.5, 0.3, 0.2])
print(f"\nH(p2)={entropy(p2):.4f}, H(p2,q2)={cross_entropy(p2,q2):.4f}, "
      f"KL(p2||q2)={kl_divergence(p2,q2):.4f}")
```

```text
H(p) = 0.0000
H(p,q) = 0.3567
KL(p||q) = 0.3567
H(p,q) - H(p) = 0.3567  (should equal KL(p||q))

H(p2)=0.8018, H(p2,q2)=0.9927, KL(p2||q2)=0.1909
```

## Exercise

1. Show numerically that $D_{KL}(p\|q) \ne D_{KL}(q\|p)$ in general (pick
   any $p\ne q$) — KL divergence is not a true distance metric because it
   isn't symmetric.
2. Prove $D_{KL}(p\|q)\ge 0$ for the 2-outcome case using calculus (minimize
   $D_{KL}$ over $q_1\in(0,1)$ with $p$ fixed, show the minimum is 0 at
   $q=p$).
3. Compute mutual information $I(X;Y) = \sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$
   for a small joint distribution table of your choosing, and explain what
   $I(X;Y)=0$ would mean about $X$ and $Y$.
