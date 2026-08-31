# 04 · Regularization Math (L1/L2)

Regularization adds a penalty term to the loss so the optimizer is pulled
toward simpler models, not just ones that fit the training data. The two
workhorses — L1 (lasso) and L2 (ridge/weight decay) — come from adding a
norm of the weights to the loss.

## The regularized objective

$$
J(w) = L(w) + \lambda R(w)
$$

where $L$ is the data loss (e.g. MSE or cross-entropy), $\lambda \ge 0$
controls penalty strength, and $R(w)$ is:

$$
R_{L2}(w) = \|w\|_2^2 = \sum_i w_i^2 \qquad
R_{L1}(w) = \|w\|_1 = \sum_i |w_i|
$$

## Gradients

**L2 (ridge):**

$$
\nabla_w R_{L2} = 2w \quad\Rightarrow\quad \nabla_w J = \nabla_w L + 2\lambda w
$$

The gradient step becomes $w \leftarrow w - \eta(\nabla_w L + 2\lambda w) =
(1-2\eta\lambda)w - \eta\nabla_w L$ — every step **shrinks** $w$ toward
zero multiplicatively before applying the data gradient. This is why L2 is
called "weight decay."

**L1 (lasso):**

$$
\nabla_w R_{L1} = \text{sign}(w) \quad\Rightarrow\quad \nabla_w J = \nabla_w L + \lambda\,\text{sign}(w)
$$

($|w|$ is non-differentiable at $w=0$; in practice use the subgradient
$\text{sign}(0)=0$.) The penalty subtracts a **constant** amount from each
weight regardless of magnitude, which is why L1 pushes many weights exactly
to zero (sparsity) while L2 shrinks everything smoothly but rarely to
exactly zero.

## Why L1 gives sparsity, geometrically

The L2 penalty contour is a circle (smooth, no corners), so the loss
contour typically touches it at a point where no coordinate is exactly
zero. The L1 penalty contour is a diamond with corners on the axes — the
loss contour is likely to first touch a corner, where one or more
coordinates are exactly zero.

## Worked numeric example

Take a single weight $w=3$ under MSE loss $L(w) = (w-1)^2$, so
$\nabla_w L = 2(w-1) = 4$, with $\lambda=0.5,\ \eta=0.1$.

**L2 step:**

$$
\nabla_w J = 4 + 2(0.5)(3) = 4+3 = 7 \quad\Rightarrow\quad w \leftarrow 3 - 0.1(7) = 2.3
$$

**L1 step:**

$$
\nabla_w J = 4 + 0.5\,\text{sign}(3) = 4.5 \quad\Rightarrow\quad w \leftarrow 3 - 0.1(4.5) = 2.55
$$

## Numeric verification

```python
import numpy as np

def loss(w):
    return (w - 1) ** 2

def grad_l2(w, lam):
    return 2 * (w - 1) + 2 * lam * w

def grad_l1(w, lam):
    return 2 * (w - 1) + lam * np.sign(w)

w, lam, eta = 3.0, 0.5, 0.1
w_l2 = w - eta * grad_l2(w, lam)
w_l1 = w - eta * grad_l1(w, lam)
print(f"L2 step: grad={grad_l2(w, lam):.4f} w_new={w_l2:.4f}")
print(f"L1 step: grad={grad_l1(w, lam):.4f} w_new={w_l1:.4f}")

# Finite-difference check of the L2 penalty gradient alone
h = 1e-6
r = lambda w: w ** 2
num_grad = (r(w + h) - r(w - h)) / (2 * h)
print(f"L2 penalty grad analytic={2*w:.4f} numeric={num_grad:.4f}")
```

```text
L2 step: grad=7.0000 w_new=2.3000
L1 step: grad=4.5000 w_new=2.5500
L2 penalty grad analytic=6.0000 numeric=6.0000
```

## Exercise

1. Run gradient descent (20 steps, $\eta=0.1$) on $L(w)=(w-1)^2$ starting
   at $w=3$ with L2 penalty $\lambda=0.5$, and separately with L1 penalty
   $\lambda=0.5$. Plot or print $w$ at each step.
2. Which one reaches exactly $w=0$ for some larger $\lambda$? Explain using
   the sign-vs-magnitude gradient behavior above.
3. Derive the gradient of "elastic net" $R = \alpha\|w\|_1 + (1-\alpha)\|w\|_2^2$
   and verify it numerically at $w=3,\alpha=0.3$.
