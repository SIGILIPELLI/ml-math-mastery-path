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
