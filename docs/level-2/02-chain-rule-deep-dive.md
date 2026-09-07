# 02 · Chain Rule Deep Dive

The single-variable chain rule from Level 1 handles $f(g(x))$. Neural
networks are compositions of many multivariable functions, so we need the
**multivariable chain rule** — the rule that makes backpropagation (Level 3)
possible.

## Single-variable recap

$$
\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
$$

## Multivariable chain rule

Suppose $z = f(x, y)$ where both $x = x(t)$ and $y = y(t)$ depend on a single
variable $t$. Then

$$
\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}
$$

Each path from $t$ to $z$ (through $x$, and through $y$) contributes a term —
you sum contributions over every path.

## Worked example

Let $z = f(x,y) = x^2 y$, with $x(t) = t^2$ and $y(t) = 3t$.

**Partials of $f$:**

$$
\frac{\partial f}{\partial x} = 2xy \qquad \frac{\partial f}{\partial y} = x^2
$$

**Derivatives of $x(t), y(t)$:**

$$
\frac{dx}{dt} = 2t \qquad \frac{dy}{dt} = 3
$$

**Chain rule:**

$$
\frac{dz}{dt} = (2xy)(2t) + (x^2)(3) = 4txy + 3x^2
$$

Substitute $x = t^2, y = 3t$:

$$
\frac{dz}{dt} = 4t(t^2)(3t) + 3(t^2)^2 = 12t^4 + 3t^4 = 15t^4
$$

**Cross-check by direct substitution:** $z(t) = (t^2)^2(3t) = 3t^5$, so
$\frac{dz}{dt} = 15t^4$ — matches exactly.

## Chain rule for a scalar composed with a vector map

The form used constantly in ML: $z = f(\mathbf{y})$ where
$\mathbf{y} = \mathbf{g}(x)$ is vector-valued. Then

$$
\frac{dz}{dx} = \sum_i \frac{\partial f}{\partial y_i}\frac{dy_i}{dx} = \nabla f(\mathbf{y})^\top \frac{d\mathbf{y}}{dx}
$$

— a dot product between the gradient of the outer function and the vector of
derivatives of the inner map. This is exactly the pattern behind
backpropagation: each layer contributes a gradient term, and they combine via
dot products along the computational graph.

## Numeric verification

```python
import numpy as np

def z_of_t(t):
    x = t**2
    y = 3*t
    return x**2 * y

def dzdt_exact(t):
    return 15 * t**4

def dzdt_numeric(t, h=1e-6):
    return (z_of_t(t + h) - z_of_t(t - h)) / (2*h)

for t0 in [0.5, 1.0, 2.0]:
    print(f"t={t0}: exact={dzdt_exact(t0):.5f}, numeric={dzdt_numeric(t0):.5f}")
```

Expected output:

```text
t=0.5: exact=0.93750, numeric=0.93750
t=1.0: exact=15.00000, numeric=15.00000
t=2.0: exact=240.00000, numeric=240.00000
```

## How It Actually Works

The vector chain rule $\nabla_x (g\circ f)(x) = J_f(x)^T \nabla_g f(x)$ is
where the "computational graph" picture becomes unavoidable. In reverse-mode
autodiff, each elementary operation in your function is a node, and each
edge carries a **vector-Jacobian product (VJP)**, not a full Jacobian
matrix. Concretely, backpropagating through a layer $y = f(x)$ never
constructs the (potentially huge) Jacobian $J_f$ explicitly; instead, given
the upstream gradient $\bar{y} = \partial L/\partial y$, the framework calls
a function that computes $\bar{x} = J_f(x)^T \bar{y}$ directly — for a
matrix multiply $y = Wx$, this VJP is just another matrix multiply,
$\bar{x} = W^T\bar{y}$, computed with the same BLAS routines as the forward
pass, never forming $J_f$ (which for a layer with $m$ inputs and $n$
outputs would be an $n\times m$ matrix, potentially far larger than $W$
itself).

This is the actual reason deep learning frameworks scale to networks with
billions of parameters: every "chain rule step" in the graph is
implemented as a hand-written, matrix-shaped VJP function (`grad_fn` in
PyTorch) rather than as generic Jacobian-matrix multiplication, keeping
memory and compute proportional to the size of the tensors themselves,
not to the (much larger) size of their Jacobians.

## Exercise

Let $z = f(x,y) = \sin(x) y^2$, $x(t) = t$, $y(t) = t^2$.

1. Compute $\partial f/\partial x$ and $\partial f/\partial y$.
2. Compute $dx/dt$ and $dy/dt$.
3. Apply the chain rule to get $dz/dt$ as a function of $t$.
4. Verify with a central-difference numeric derivative in NumPy at
   $t = 1.0$ and $t = 2.0$.
