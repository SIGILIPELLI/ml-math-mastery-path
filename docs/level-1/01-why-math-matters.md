# 01 · Why Math Matters for ML

You can call `.fit()` on a scikit-learn model without knowing any of the math
underneath it, and it will happily run. But the moment something goes wrong —
training diverges, loss becomes `NaN`, the model won't learn, a paper's
notation looks like alphabet soup — the only way through is the math. This
module is the "why," in plain language, before we get mechanical in the
modules that follow.

## The three pillars, and where each one shows up

Every ML algorithm you'll ever meet is built from three mathematical
ingredients:

1. **Linear algebra** — data is stored as vectors and matrices (a row of a
   spreadsheet is a vector; a whole dataset is a matrix). Model parameters
   are vectors and matrices too. "Making a prediction" is almost always a
   matrix multiplication.
2. **Calculus (derivatives)** — "training" a model means adjusting its
   parameters to reduce error. To know *which direction* to adjust each
   parameter, you need the derivative of the error with respect to that
   parameter. This is not optional — it is the mechanism of learning.
3. **Probability & statistics** — data is noisy, models are uncertain, and
   most loss functions (like cross-entropy) come directly from probability
   theory ("what set of parameters makes the observed data most likely?").

Concretely, here's how they show up in an algorithm you may already know
informally — **gradient descent**, the workhorse that trains almost every
model from linear regression to GPT-scale networks:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \alpha \, \nabla_\theta J(\theta)
$$

Reading this formula left to right is itself a small linear-algebra and
calculus exercise: $\theta$ is a **vector** of parameters (linear algebra),
$\nabla_\theta J$ is the **gradient** — a vector of partial derivatives
(calculus) — of the cost function $J$, and $\alpha$ is a scalar learning
rate. By the end of Level 1 you will be able to read every symbol in that
line and compute it by hand for a small example.

## A concrete before/after

Suppose we want to fit a line $\hat{y} = wx + b$ to some data points, and we
measure error with mean squared error:

$$
J(w, b) = \frac{1}{n}\sum_{i=1}^{n} \left( wx_i + b - y_i \right)^2
$$

Without calculus, the only way to find a good $(w, b)$ is guessing — try a
value, check the error, try another. With calculus, we can compute the exact
direction that *decreases* $J$ fastest for any $(w, b)$: that's the gradient
$\nabla J = \left(\frac{\partial J}{\partial w}, \frac{\partial J}{\partial
b}\right)$, which we derive by hand in Module 9 and verify numerically.

## Worked numeric example: guessing vs. computing

Let's make "guessing is worse than computing" concrete with three data
points: $(1, 2), (2, 3), (3, 5)$, and a fixed intercept $b = 0$, so we only
need to find the best slope $w$.

**Guessing approach.** Try $w = 1$: predictions are $1, 2, 3$; squared errors
are $(1-2)^2=1$, $(2-3)^2=1$, $(3-5)^2=4$; mean squared error
$J(1) = \frac{1+1+4}{3} = 2.0$.

Try $w = 2$: predictions are $2, 4, 6$; squared errors are $0, 1, 1$;
$J(2) = \frac{0+1+1}{3} = 0.667$. Better — but which direction next? Guessing
gives no hint.

**Calculus approach.** The derivative of $J(w) = \frac{1}{3}\sum (wx_i -
y_i)^2$ with respect to $w$ is

$$
\frac{dJ}{dw} = \frac{2}{3}\sum_{i=1}^{3} x_i(wx_i - y_i)
$$

At $w = 2$: $x_i(wx_i - y_i)$ for each point is $1\cdot(2-2)=0$,
$2\cdot(4-3)=2$, $3\cdot(6-5)=3$, summing to $5$, so
$\frac{dJ}{dw}\Big|_{w=2} = \frac{2}{3}\cdot 5 \approx 3.33$. Positive slope
means increasing $w$ increases error, so we should **decrease** $w$ — one
derivative evaluation tells us exactly which way to move, with no more
guessing needed.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 5.0])

def cost(w):
    pred = w * x
    return np.mean((pred - y) ** 2)

def grad(w):
    pred = w * x
    return np.mean(2 * x * (pred - y))

for w in [1.0, 2.0]:
    print(f"J({w}) = {cost(w):.3f}")

print("dJ/dw at w=2:", grad(2.0))
```

Expected output (hand-computed above, matches NumPy):

```text
J(1.0) = 2.000
J(2.0) = 0.667
dJ/dw at w=2: 3.333...
```

The sign and rough magnitude of `grad(2.0)` match our hand derivation
($\approx 3.33$), confirming that increasing $w$ past 2 makes things worse —
exactly the signal gradient descent uses to know which way to step.

## What you'll be able to do by the end of Level 1

- Read $\mathbf{v}$, $\mathbf{A}$, $\nabla f$, $\partial f/\partial x$
  notation fluently.
- Compute dot products, norms, and matrix products by hand for small
  examples, and verify with NumPy.
- Take derivatives of polynomials and simple compositions using the power,
  sum, and chain rules.
- Compute a gradient for a two-variable cost function and confirm it
  numerically.
- Understand — mechanically, not just conceptually — why linear regression's
  cost function has the shape it does, and how it connects to the general ML
  training loop.

## Exercise

1. Using the three points $(1,2), (2,3), (3,5)$ above, compute $J(w)$ by hand
   for $w = 1.5$ and $w = 2.5$.
2. Compute $\frac{dJ}{dw}$ by hand at $w = 1.5$ using the formula given
   above.
3. Write the four-line NumPy snippet (`cost`, `grad`) from this module and
   confirm your hand-computed values match the code's output.
4. In one or two sentences, explain what the *sign* of `grad(1.5)` tells you
   about which direction to move $w$ next.
