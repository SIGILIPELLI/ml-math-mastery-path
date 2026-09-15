---
description: "Bayes' Theorem — Bayes' theorem lets you flip a conditional probability around — update a belief given new evidence. It's the mathematical core of naive…"
---

# 08 · Bayes' Theorem

Bayes' theorem lets you flip a conditional probability around — update a
belief given new evidence. It's the mathematical core of naive Bayes
classifiers, Bayesian inference (Level 4), and spam filters.

## Derivation

From the definition of conditional probability (Module 6):

$$
P(A\mid B) = \frac{P(A\cap B)}{P(B)} \qquad P(B\mid A) = \frac{P(A\cap B)}{P(A)}
$$

Both equal $P(A\cap B)$ when solved for it: $P(A\cap B) = P(A\mid B)P(B) =
P(B\mid A)P(A)$. Rearranging gives **Bayes' theorem**:

$$
P(A\mid B) = \frac{P(B\mid A)\,P(A)}{P(B)}
$$

In ML terms, with $A$ = hypothesis/class, $B$ = observed data:

$$
\underbrace{P(A\mid B)}_{\text{posterior}} = \frac{\overbrace{P(B\mid A)}^{\text{likelihood}}\ \overbrace{P(A)}^{\text{prior}}}{\underbrace{P(B)}_{\text{evidence}}}
$$

## Worked example: medical test

A disease affects $1\%$ of a population ($P(D)=0.01$). A test correctly
detects it $99\%$ of the time ($P(+\mid D) = 0.99$, the true positive rate)
but has a $5\%$ false-positive rate ($P(+\mid D^c) = 0.05$). Given a
positive test, what's $P(D\mid +)$?

**Evidence (law of total probability):**

$$
P(+) = P(+\mid D)P(D) + P(+\mid D^c)P(D^c) = (0.99)(0.01)+(0.05)(0.99)
$$

$$
P(+) = 0.0099 + 0.0495 = 0.0594
$$

**Bayes' theorem:**

$$
P(D\mid +) = \frac{P(+\mid D)P(D)}{P(+)} = \frac{0.0099}{0.0594} \approx 0.1667
$$

Despite a "99% accurate" test, a positive result only means about a **16.7%**
chance of actually having the disease — because the disease is rare, false
positives from the large healthy population dominate. This is the classic
base-rate fallacy, and it's exactly what Bayes' theorem corrects for.

## Naive Bayes preview

A naive Bayes classifier picks the class $c$ maximizing
$P(c\mid \mathbf{x}) \propto P(\mathbf{x}\mid c)P(c)$, and assumes (naively)
that features are conditionally independent given the class, so
$P(\mathbf{x}\mid c) = \prod_i P(x_i\mid c)$ — turning an intractable joint
likelihood into a simple product, directly using Bayes' theorem plus the
independence idea from Module 6.

## Numeric verification

```python
import numpy as np

P_D = 0.01
P_pos_given_D = 0.99
P_pos_given_notD = 0.05

P_pos = P_pos_given_D * P_D + P_pos_given_notD * (1 - P_D)
P_D_given_pos = (P_pos_given_D * P_D) / P_pos
print("P(+)        :", P_pos)
print("P(D | +)    :", P_D_given_pos)

# Cross-check with a large simulation
rng = np.random.default_rng(0)
n = 5_000_000
has_disease = rng.random(n) < P_D
test_positive = np.where(
    has_disease,
    rng.random(n) < P_pos_given_D,
    rng.random(n) < P_pos_given_notD,
)
print("simulated P(D | +):", has_disease[test_positive].mean())
```

Expected output:

```text
P(+)        : 0.0594
P(D | +)    : 0.16666666666666669
simulated P(D | +): 0.1667...
```

## How It Actually Works

Bayes' theorem, $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$, looks like a single
division, but applying it repeatedly (as Naive Bayes does, multiplying
likelihoods across many features) multiplies many probabilities together —
each less than 1 — and the product **underflows to exactly 0.0** in
floating point after only a few dozen features, since float64 can't
represent numbers below roughly $10^{-308}$ and this product shrinks
geometrically. A classifier comparing two computed 0.0 values can no longer
tell which class was more likely, even though the true (infinitesimally
small but distinct) probabilities differed.

The fix used in every real implementation (scikit-learn's `GaussianNB`
included) is to work in **log-space**: since $\log$ is monotonic, comparing
$\log P(A|B)$ values ranks identically to comparing $P(A|B)$ values
directly, but $\log$ turns the underflowing product of likelihoods into a
sum, $\log P(B|A) = \sum_i \log P(b_i|A)$, which stays numerically
representable across hundreds of features. When you eventually do need an
actual probability (not just a ranking) recovered from log-probabilities,
you need the **log-sum-exp trick**:
$\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$ where $m=\max_i x_i$,
which prevents `exp` from overflowing on the largest term while keeping
the result exact — this exact trick reappears verbatim in the softmax and
cross-entropy module ahead.

## Exercise

A factory has two machines: Machine A makes 60% of products with a 2%
defect rate; Machine B makes 40% with a 5% defect rate.

1. Compute $P(\text{defect})$ using the law of total probability.
2. Given a random product is defective, compute $P(\text{Machine A}\mid
   \text{defect})$ and $P(\text{Machine B}\mid \text{defect})$ using Bayes'
   theorem (they should sum to 1).
3. Explain in one sentence why the posterior $P(A\mid\text{defect})$ is
   lower than the prior $P(A)=0.6$.
4. Verify both posteriors with a NumPy simulation of at least 2,000,000
   products.
