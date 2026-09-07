# 01 · Gradient Descent Step-by-Step

Level 1 ended with the gradient $\nabla f$ as the direction of steepest
*increase*. Gradient descent is the algorithm that uses $-\nabla f$ to
iteratively walk **downhill** toward a minimum — this is literally how every
model in this course gets trained.

## The update rule

Given a cost function $J(\theta)$ over parameters $\theta$, gradient descent
repeats:

$$
\theta_{t+1} = \theta_t - \alpha \, \nabla J(\theta_t)
$$

where $\alpha$ is the **learning rate**, a small positive scalar controlling
step size. Each step moves $\theta$ a little in the direction that decreases
$J$ fastest, based on the local gradient.

## Why subtract the gradient

Near $\theta_t$, a first-order (linear) approximation gives

$$
J(\theta_t - \alpha \nabla J(\theta_t)) \approx J(\theta_t) - \alpha \lVert \nabla J(\theta_t) \rVert^2
$$

Since $\alpha > 0$ and $\lVert \nabla J \rVert^2 \geq 0$, this approximation is
always $\leq J(\theta_t)$ — moving opposite the gradient decreases the cost
(for small enough $\alpha$; too large an $\alpha$ can overshoot).

## Worked example

Let $J(\theta) = (\theta - 3)^2 + 2$, a simple 1-D parabola with minimum at
$\theta = 3$. Its derivative is

$$
J'(\theta) = 2(\theta - 3)
$$

Start at $\theta_0 = 0$ with learning rate $\alpha = 0.3$.

**Step 1:**

$$
J'(0) = 2(0-3) = -6 \qquad \theta_1 = 0 - 0.3(-6) = 1.8
$$

**Step 2:**

$$
J'(1.8) = 2(1.8-3) = -2.4 \qquad \theta_2 = 1.8 - 0.3(-2.4) = 2.52
$$

**Step 3:**

$$
J'(2.52) = 2(2.52-3) = -0.96 \qquad \theta_3 = 2.52 - 0.3(-0.96) = 2.808
$$

Each step gets closer to $\theta = 3$, and the distance to the minimum
shrinks by a constant factor ($1 - 2\alpha = 0.4$) every step, since this
particular $J$ is quadratic.

## Learning rate matters

* Too small $\alpha$: convergence is correct but painfully slow.
* Too large $\alpha$: for this quadratic, $\alpha > 1$ causes divergence
  (since the contraction factor $|1-2\alpha|$ exceeds 1); values between 0
  and 1 converge, with $\alpha$ near 0.5 converging in a single step here.

## Numeric verification

```python
import numpy as np

def J(theta):
    return (theta - 3) ** 2 + 2

def dJ(theta):
    return 2 * (theta - 3)

theta = 0.0
alpha = 0.3
history = [theta]
for step in range(3):
    theta = theta - alpha * dJ(theta)
    history.append(theta)

print("theta after each step:", history)
print("J at final theta:", J(theta))
```

Expected output (matches the hand computation):

```text
theta after each step: [0.0, 1.8, 2.52, 2.808]
J at final theta: 2.036864
```

## How It Actually Works

A gradient descent update, $\theta \leftarrow \theta - \alpha\nabla J(\theta)$,
is a single line of code but hides several floating-point failure modes
that this module's "learning rate matters" section only gestures at.
If $\alpha\nabla J(\theta)$ overflows float32's range (roughly
$3.4\times10^{38}$) — which happens when the gradient itself is huge
(exploding gradients) — the update produces `inf`, and the next loss
evaluation computes `inf - inf` or similar, yielding `NaN` that never
recovers, because every subsequent arithmetic operation touching a `NaN`
value stays `NaN` (IEEE-754's defined "poisoning" behavior). Too small an
$\alpha$ has the opposite numerical failure: if $\alpha\nabla J(\theta)$ is
smaller than float32's precision relative to $\theta$'s magnitude (roughly
$\theta \times 1.2\times10^{-7}$), the update rounds to exactly zero —
$\theta$ stops changing even though the mathematical gradient is nonzero,
a phenomenon distinct from mathematical convergence and purely an artifact
of finite precision.

This is also why real training loops rarely use plain gradient descent:
momentum-based optimizers (Level 3 Module 03) maintain an exponentially
weighted running average of past gradients specifically to smooth out the
per-step floating-point noise that arises from mini-batch sampling, and
mixed-precision training (float16 forward/backward, float32 master weights)
exists precisely because float16's ~3 decimal digits of precision would
otherwise let exactly this kind of "update rounds to zero" problem destroy
training — the master weights are kept in float32 so the accumulation of
many small updates doesn't get lost to rounding.

## Exercise

Let $J(\theta) = \theta^2 - 4\theta + 10$.

1. Find $J'(\theta)$ and the exact minimizer $\theta^*$ by setting
   $J'(\theta^*) = 0$.
2. Starting at $\theta_0 = 0$ with $\alpha = 0.2$, hand-compute $\theta_1$,
   $\theta_2$, $\theta_3$.
3. Implement the loop in NumPy and confirm your three values match.
4. Try $\alpha = 1.1$ for 5 steps and observe (print, don't just guess)
   whether the sequence diverges — explain why using the contraction factor
   $|1 - 2\alpha|$.
