---
description: "Probability Basics — Calculus and linear algebra tell us how to optimize; probability tells us how to reason about uncertainty — noisy labels, random…"
---

# 06 · Probability Basics

Calculus and linear algebra tell us how to optimize; probability tells us how
to reason about uncertainty — noisy labels, random initialization, and
whole model families (naive Bayes, Bayesian methods, probabilistic loss
functions) are built directly on it.

## Sample spaces and events

A **sample space** $\Omega$ is the set of all possible outcomes of a random
process. An **event** $A$ is any subset of $\Omega$. The probability
function $P$ assigns each event a number in $[0,1]$ satisfying:

1. $P(\Omega) = 1$
2. $P(A) \geq 0$ for every event $A$
3. If $A$ and $B$ are mutually exclusive ($A \cap B = \emptyset$), then
   $P(A \cup B) = P(A) + P(B)$

## Key rules

**Complement:** $P(A^c) = 1 - P(A)$.

**Union (inclusion–exclusion):**

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

**Conditional probability:** the probability of $A$ given $B$ has occurred:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} \qquad (P(B) > 0)
$$

**Independence:** $A$ and $B$ are independent iff
$P(A \cap B) = P(A)P(B)$, equivalently $P(A\mid B) = P(A)$.

## Worked example

A fair six-sided die is rolled. Let $A$ = "roll is even" = $\{2,4,6\}$,
$B$ = "roll is $\geq 4$" = $\{4,5,6\}$.

$$
P(A) = 3/6 = 0.5 \qquad P(B) = 3/6 = 0.5 \qquad P(A\cap B) = |\{4,6\}|/6 = 2/6
$$

**Union:**

$$
P(A\cup B) = 0.5 + 0.5 - 2/6 = 1 - 1/3 = 2/3
$$

(Check: $A \cup B = \{2,4,5,6\}$, which has 4 outcomes out of 6, so
$4/6=2/3$. ✓)

**Conditional:**

$$
P(A\mid B) = \frac{2/6}{3/6} = \frac{2}{3}
$$

**Independence check:** $P(A)P(B) = 0.25 \neq P(A\cap B) = 1/3$, so $A$ and
$B$ are **not** independent.

## Why this matters for ML

Class labels, dropout masks, mini-batch sampling, and data-augmentation
choices are all modeled as random events. Naive Bayes classifiers apply the
conditional-probability and independence rules directly to features; the
"naive" in the name is literally the (often false, but useful)
assumption of feature independence.

## Numeric verification (Monte Carlo)

```python
import numpy as np

rng = np.random.default_rng(0)
n = 2_000_000
rolls = rng.integers(1, 7, size=n)

A = (rolls % 2 == 0)
B = (rolls >= 4)

print("P(A)      ~", A.mean())
print("P(B)      ~", B.mean())
print("P(A and B)~", (A & B).mean())
print("P(A or B) ~", (A | B).mean())
print("P(A|B)    ~", (A & B).sum() / B.sum())
```

Expected output (Monte Carlo estimates converge to the exact fractions
0.5, 0.5, 0.3333, 0.6667, 0.6667):

```text
P(A)      ~ 0.499948
P(B)      ~ 0.500222
P(A and B)~ 0.333174
P(A or B) ~ 0.666996
P(A|B)    ~ 0.665933
```

## How It Actually Works

The "random" numbers behind any Monte Carlo verification you run
(`np.random.rand()`, sampling from a distribution) are not random at all —
they come from a **deterministic pseudo-random number generator (PRNG)**,
typically a variant of the Mersenne Twister (NumPy's legacy default) or
PCG64 (NumPy's current default), seeded with an initial state and then
advanced by a fixed recurrence that produces a long, statistically
uniform-looking sequence of integers, which are then bit-manipulated into
floats uniformly distributed on $[0,1)$. Given the same seed, you get
*exactly* the same "random" sequence every time — which is precisely why
setting a seed makes ML experiments reproducible.

Sampling from a non-uniform distribution (Bernoulli, Gaussian, ...) from
this uniform stream uses specific numerical algorithms: **inverse transform
sampling** computes $F^{-1}(u)$ for uniform $u$ where $F$ is the target
CDF (exact when $F^{-1}$ has a closed form, e.g. exponential); Gaussian
sampling commonly uses the **Box-Muller transform** (turning two uniform
samples into two independent standard normal samples via
$z_1 = \sqrt{-2\ln u_1}\cos(2\pi u_2)$) or the **ziggurat algorithm** for
speed. Also worth knowing: probabilities are frequently computed and
combined in **log-space** ($\log p$ instead of $p$) throughout ML code,
because multiplying many probabilities less than 1 quickly underflows to
exactly 0.0 in floating point, while summing their logs stays numerically
representable — this single trick reappears in Bayes' theorem, MLE, and
softmax modules ahead.

## Exercise

A fair coin is flipped twice. Let $A$ = "first flip is heads",
$B$ = "at least one flip is heads".

1. List the sample space (4 equally likely outcomes) and compute $P(A)$,
   $P(B)$, $P(A\cap B)$ by counting outcomes.
2. Compute $P(A\cup B)$ using inclusion–exclusion and verify by direct
   counting.
3. Compute $P(A\mid B)$ and state whether $A,B$ are independent.
4. Verify all four numbers with a NumPy Monte Carlo simulation of at least
   1,000,000 trials.
