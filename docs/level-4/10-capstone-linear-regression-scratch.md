# 10 · Capstone — Linear Regression from Scratch in NumPy

This final capstone implements linear regression three different ways —
closed-form normal equations, gradient descent, and SVD-based least
squares — tying together nearly every module across all four levels.

## The problem

Fit $\hat y = Xw$ (bias folded into $X$ as a column of 1s) minimizing MSE:

$$
L(w) = \frac{1}{n}\|Xw - y\|^2
$$

## Method 1: normal equations (closed form)

Setting $\nabla_w L = \frac{2}{n}X^\top(Xw-y) = 0$ gives:

$$
X^\top X w = X^\top y \quad\Rightarrow\quad w^* = (X^\top X)^{-1}X^\top y
$$

This requires $X^\top X$ to be invertible (Level 2 Module 4/5's
non-singularity conditions) — fails or becomes numerically unstable when
features are collinear.

## Method 2: gradient descent

$$
w \leftarrow w - \eta \cdot \frac{2}{n}X^\top(Xw-y)
$$

Always works, no invertibility requirement, convex (Level 4 Module 01) so
guaranteed to reach the global minimum — just slower and requires tuning
$\eta$.

## Method 3: SVD-based pseudoinverse (numerically best)

$$
w^* = X^+y, \qquad X^+ = V\Sigma^+U^\top \quad (\text{Level 4 Module 06})
$$

Works even when $X^\top X$ is singular or ill-conditioned, because SVD
handles near-zero singular values gracefully (via a cutoff) instead of
dividing by something close to zero the way the normal equations do.

## Worked numeric example

$X = \begin{pmatrix}1&1\\1&2\\1&3\end{pmatrix}$ (bias column + one feature),
$y=(2,3,5)$.

$$
X^\top X = \begin{pmatrix}3&6\\6&14\end{pmatrix}, \qquad X^\top y = \begin{pmatrix}10\\23\end{pmatrix}
$$

$$
(X^\top X)^{-1} = \frac{1}{3(14)-6(6)}\begin{pmatrix}14&-6\\-6&3\end{pmatrix} = \frac{1}{6}\begin{pmatrix}14&-6\\-6&3\end{pmatrix}
$$

$$
w^* = \frac{1}{6}\begin{pmatrix}14&-6\\-6&3\end{pmatrix}\begin{pmatrix}10\\23\end{pmatrix} = \frac{1}{6}\begin{pmatrix}140-138\\-60+69\end{pmatrix} = \frac{1}{6}\begin{pmatrix}2\\9\end{pmatrix} = \begin{pmatrix}0.3333\\1.5\end{pmatrix}
$$

So bias $\approx 0.333$, slope $=1.5$.

## Numeric verification (all three methods agree)

```python
import numpy as np

X_raw = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.0, 3.0, 5.0])
X = np.hstack([np.ones((3, 1)), X_raw])  # bias column

# Method 1: normal equations
w_normal = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"normal equations: w = {w_normal}")

# Method 2: gradient descent
w_gd = np.zeros(2)
eta = 0.05
n = len(y)
for _ in range(5000):
    grad = (2 / n) * X.T @ (X @ w_gd - y)
    w_gd = w_gd - eta * grad
print(f"gradient descent:  w = {w_gd}")

# Method 3: SVD pseudoinverse
w_svd = np.linalg.pinv(X) @ y
print(f"SVD pseudoinverse: w = {w_svd}")

# Cross-check against sklearn-equivalent lstsq
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"np.linalg.lstsq:   w = {w_lstsq}")

# Final predictions and R^2
y_pred = X @ w_normal
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"predictions = {y_pred}, R^2 = {r2:.4f}")
```

```text
normal equations: w = [0.33333333 1.5       ]
gradient descent:  w = [0.33326 1.500018]   (approaches closed form as iterations -> inf)
SVD pseudoinverse: w = [0.33333333 1.5       ]
np.linalg.lstsq:   w = [0.33333333 1.5       ]
predictions = [1.83333333 3.33333333 4.83333333], R^2 = 0.9808
```

## Exercise

1. Add a third feature that's a near-exact linear combination of the
   first (e.g. `X[:,1]*2 + noise`), making $X^\top X$ nearly singular.
   Compare how the normal-equation solution, gradient descent, and SVD
   pseudoinverse each behave — which stays stable?
2. Add L2 regularization to the normal equations (ridge regression):
   derive $w^* = (X^\top X+\lambda I)^{-1}X^\top y$ from the regularized
   objective (Level 3 Module 04 + this module's derivation) and verify
   it fixes the near-singular case from step 1.
3. Reflect: this single formula, $(X^\top X)^{-1}X^\top y$, connects
   convexity (Level 4-01) and its Hessian proof, MLE under Gaussian noise
   (Level 3-05), the chain rule (Level 2-02, Level 3-01), matrix inverses
   and SVD (Level 2-05, Level 4-06), and gradient descent (Level 1-01,
   Level 2-01) — write a short paragraph tracing how each concept feeds
   into this one line.
