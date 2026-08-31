# 03 · Optimization Beyond Vanilla GD

Vanilla gradient descent (Level 2 Module 1) treats every step independently.
**Momentum** and **Adam** use the *history* of gradients to converge faster
and handle noisy or ill-conditioned loss surfaces — this is what optimizers
like `torch.optim.Adam` actually compute under the hood.

## Momentum

Momentum keeps an exponentially-decaying running average of past gradients,
$v_t$, and steps using that instead of the raw gradient:

$$
v_t = \beta v_{t-1} + (1-\beta)\nabla J(\theta_{t-1}) \qquad \theta_t = \theta_{t-1} - \alpha v_t
$$

with $\beta \in [0,1)$ (commonly $0.9$). Intuition: like a ball rolling
downhill, momentum smooths out oscillations across steep, narrow valleys and
keeps moving through small local bumps.

## Adam (Adaptive Moment Estimation)

Adam tracks **two** running averages: the first moment (mean, like momentum)
and the second moment (uncentered variance of the gradient), then uses both
to adapt the step size per-parameter:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

where $g_t = \nabla J(\theta_{t-1})$. Because $m_0=v_0=0$, early estimates
are biased toward zero, so Adam **bias-corrects**:

$$
\hat m_t = \frac{m_t}{1-\beta_1^t} \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

Common defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. Dividing
by $\sqrt{\hat v_t}$ shrinks the step for parameters with consistently large
gradients and grows it for parameters with small/sparse gradients —
adaptive per-parameter learning rates.

## Worked example: momentum, 2 steps

$J(\theta)=(\theta-3)^2+2$ (same as Level 2 Module 1),
$J'(\theta)=2(\theta-3)$. Start $\theta_0=0$, $v_0=0$, $\alpha=0.3$,
$\beta=0.9$.

**Step 1:** $g_1 = J'(0) = -6$.

$$
v_1 = 0.9(0)+0.1(-6) = -0.6 \qquad \theta_1 = 0-0.3(-0.6)=0.18
$$

**Step 2:** $g_2 = J'(0.18) = 2(0.18-3)=-5.64$.

$$
v_2 = 0.9(-0.6)+0.1(-5.64) = -0.54-0.564=-1.104
$$

$$
\theta_2 = 0.18-0.3(-1.104) = 0.18+0.3312=0.5112
$$

Momentum's first steps look smaller than vanilla GD's (which reached $1.8$
after step 1) because $v_t$ starts at zero and needs a few steps to "warm
up" — but it accelerates once the running average builds up speed in a
consistent direction.

## Numeric verification

```python
import numpy as np

def dJ(theta):
    return 2 * (theta - 3)

# Momentum
theta, v, alpha, beta = 0.0, 0.0, 0.3, 0.9
for step in range(2):
    g = dJ(theta)
    v = beta * v + (1 - beta) * g
    theta = theta - alpha * v
    print(f"momentum step {step+1}: theta={theta:.4f}")

# Adam
theta, m, v, alpha = 0.0, 0.0, 0.0, 0.3
beta1, beta2, eps = 0.9, 0.999, 1e-8
for t in range(1, 6):
    g = dJ(theta)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    theta = theta - alpha * m_hat / (np.sqrt(v_hat) + eps)
    print(f"adam step {t}: theta={theta:.4f}")
```

Expected output (momentum matches the hand computation; Adam converges
quickly toward the minimum $\theta=3$ due to adaptive step sizes):

```text
momentum step 1: theta=0.1800
momentum step 2: theta=0.5112
adam step 1: theta=0.3000
adam step 2: theta=0.5998
adam step 3: theta=0.8981
adam step 4: theta=1.1926
adam step 5: theta=1.4808
```

## Exercise

Using $J(\theta)=\theta^2$ ($J'(\theta)=2\theta$), start $\theta_0=5$.

1. Hand-compute 2 steps of momentum with $\alpha=0.1,\beta=0.9,v_0=0$.
2. Hand-compute 2 steps of vanilla gradient descent with the same $\alpha$
   and compare which moves faster initially — explain why.
3. Implement both in NumPy for 20 steps and plot/print $\theta$ over time.
4. Implement Adam for the same 20 steps with default hyperparameters and
   compare final $\theta$ values across all three methods.
