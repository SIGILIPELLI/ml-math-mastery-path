# 03 · Jacobians & Hessians

The gradient generalizes derivatives for a scalar-valued function of many
variables. Two more objects generalize further: the **Jacobian** (for
vector-valued outputs) and the **Hessian** (second derivatives of a scalar
function) — both used constantly in ML: Jacobians in backprop through
vector layers, Hessians in second-order optimizers and curvature analysis.

## The Jacobian

If $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ maps
$\mathbf{x} = (x_1,\dots,x_n)$ to $\mathbf{f}(\mathbf{x}) = (f_1,\dots,f_m)$,
the Jacobian is the $m \times n$ matrix of all first partial derivatives:

$$
J = \begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

Row $i$ is $\nabla f_i(\mathbf{x})^\top$ — the Jacobian is just all the
per-output gradients stacked.

### Worked example

$\mathbf{f}(x,y) = (x^2y,\ x+y^2)$, so $f_1 = x^2y$, $f_2 = x+y^2$.

$$
J = \begin{bmatrix} 2xy & x^2 \\ 1 & 2y \end{bmatrix}
$$

At $(x,y)=(1,2)$:

$$
J(1,2) = \begin{bmatrix} 4 & 1 \\ 1 & 4 \end{bmatrix}
$$

## The Hessian

For a scalar function $f(x,y)$, the Hessian is the matrix of **second**
partial derivatives:

$$
H = \begin{bmatrix}
\dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x \partial y} \\[6pt]
\dfrac{\partial^2 f}{\partial y \partial x} & \dfrac{\partial^2 f}{\partial y^2}
\end{bmatrix}
$$

For smooth functions the mixed partials are equal
($\partial^2 f/\partial x\partial y = \partial^2 f/\partial y\partial x$), so
$H$ is symmetric. The Hessian describes local curvature: its eigenvalues
(Module 4) tell you whether a critical point is a minimum (all positive),
maximum (all negative), or saddle (mixed signs).

### Worked example

$f(x,y) = x^2y + 3y^2 + 5x$ (same $f$ as Level 1 Module 8, where we found
$\nabla f = [2xy+5,\ x^2+6y]$).

$$
\frac{\partial^2 f}{\partial x^2} = 2y \qquad
\frac{\partial^2 f}{\partial y^2} = 6 \qquad
\frac{\partial^2 f}{\partial x \partial y} = 2x
$$

$$
H(x,y) = \begin{bmatrix} 2y & 2x \\ 2x & 6 \end{bmatrix}
$$

At $(2,1)$: $H(2,1) = \begin{bmatrix} 2 & 4 \\ 4 & 6 \end{bmatrix}$.

## Numeric verification

```python
import numpy as np

def f(x, y):
    return x**2 * y + 3 * y**2 + 5 * x

def hessian_numeric(x, y, h=1e-4):
    fxx = (f(x+h, y) - 2*f(x, y) + f(x-h, y)) / h**2
    fyy = (f(x, y+h) - 2*f(x, y) + f(x, y-h)) / h**2
    fxy = (f(x+h, y+h) - f(x+h, y-h) - f(x-h, y+h) + f(x-h, y-h)) / (4*h**2)
    return np.array([[fxx, fxy], [fxy, fyy]])

H_exact = np.array([[2*1, 2*2], [2*2, 6]])
H_numeric = hessian_numeric(2.0, 1.0)
print("exact Hessian:\n", H_exact)
print("numeric Hessian:\n", np.round(H_numeric, 4))
```

Expected output (numeric matches exact up to finite-difference error):

```text
exact Hessian:
 [[2 4]
 [4 6]]
numeric Hessian:
 [[2. 4.]
 [4. 6.]]
```

## Exercise

For $\mathbf{f}(x,y) = (xy,\ x^2 - y^2)$:

1. Compute the Jacobian symbolically and evaluate it at $(1,1)$.
2. For the scalar $g(x,y) = x^3 + xy^2$, compute the Hessian symbolically.
3. Evaluate the Hessian at $(1,2)$ and classify the point $(0,0)$ using the
   sign pattern of the Hessian at the origin (compute it there too).
4. Verify your Hessian at $(1,2)$ with the finite-difference pattern above.
