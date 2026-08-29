# 08 · Partial Derivatives & Gradients

Every derivative so far has been for a function of **one** variable. Real
ML cost functions depend on many parameters at once — a linear model with
10 features has 10 weights plus a bias, all being tuned simultaneously.
**Partial derivatives** and the **gradient** extend everything from Modules
6–7 to many variables at once.

## Partial derivatives

For a function of two variables $f(x, y)$, the **partial derivative with
respect to $x$**, written $\frac{\partial f}{\partial x}$, is the derivative
of $f$ treating $y$ as a fixed constant — apply every rule from Module 7
exactly as before, just holding the other variable still.

### Worked example

Let $f(x,y) = x^2y + 3y^2 + 5x$.

**With respect to $x$** (treat $y$ as constant):

$$
\frac{\partial f}{\partial x} = 2xy + 0 + 5 = 2xy + 5
$$

(The term $x^2y$ is $y\cdot x^2$ with $y$ constant, so power rule gives
$y\cdot 2x = 2xy$. The term $3y^2$ has no $x$ in it, so it's a constant with
respect to $x$, and its derivative is 0. The term $5x$ differentiates to
$5$.)

**With respect to $y$** (treat $x$ as constant):

$$
\frac{\partial f}{\partial y} = x^2 + 6y + 0 = x^2 + 6y
$$

($x^2y$ is $x^2\cdot y$ with $x^2$ constant, so its derivative w.r.t. $y$ is
just $x^2$. $3y^2$ differentiates to $6y$. $5x$ has no $y$, so it
contributes 0.)

## The gradient

Stack all partial derivatives of a scalar function into a vector — this is
the **gradient**, written $\nabla f$:

$$
\nabla f(x,y) = \begin{bmatrix} \dfrac{\partial f}{\partial x} \\[4pt] \dfrac{\partial f}{\partial y} \end{bmatrix}
$$

For $f(x,y) = x^2y+3y^2+5x$ above:

$$
\nabla f(x,y) = \begin{bmatrix} 2xy+5 \\ x^2+6y \end{bmatrix}
$$

## Why the gradient matters: direction of steepest ascent

The gradient $\nabla f$, evaluated at a point, is a vector that points in
the direction where $f$ **increases fastest** from that point. Its negative,
$-\nabla f$, points in the direction of **steepest decrease** — which is
exactly why gradient descent (previewed in Module 1, fully derived in Level
2) updates parameters by stepping in the direction $-\nabla J$: it's the
locally best direction to *reduce* the cost function.

## Worked numeric example

Evaluate $\nabla f$ at the point $(x,y) = (2,1)$ for
$f(x,y)=x^2y+3y^2+5x$:

$$
\nabla f(2,1) = \begin{bmatrix} 2(2)(1)+5 \\ 2^2+6(1) \end{bmatrix} = \begin{bmatrix} 4+5 \\ 4+6 \end{bmatrix} = \begin{bmatrix} 9 \\ 10 \end{bmatrix}
$$

We can verify each partial derivative numerically using the same
finite-difference idea from Module 6, applied one variable at a time
(holding the other fixed):

$$
\frac{\partial f}{\partial x}(2,1) \approx \frac{f(2+h,1)-f(2,1)}{h}
$$

```python
import numpy as np

def f(x, y):
    return x**2 * y + 3 * y**2 + 5 * x

def grad_exact(x, y):
    df_dx = 2 * x * y + 5
    df_dy = x**2 + 6 * y
    return np.array([df_dx, df_dy])

def grad_numeric(x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x, y)) / h
    df_dy = (f(x, y + h) - f(x, y)) / h
    return np.array([df_dx, df_dy])

x0, y0 = 2.0, 1.0
print("f(2,1)         :", f(x0, y0))
print("exact gradient  :", grad_exact(x0, y0))
print("numeric gradient:", grad_numeric(x0, y0))
```

Expected output (matches the hand computation $\nabla f(2,1) = [9, 10]$,
with the numeric version very close due to small `h`):

```text
f(2,1)         : 11.0
exact gradient  : [ 9. 10.]
numeric gradient: [8.99998 10.00003]
```

(f(2,1) = 4*1 + 3*1 + 10 = 4+3+10 = 11, confirming the function value too.)

## Exercise

Let $g(x,y) = 3x^2 + 2xy + y^2$.

1. Compute $\frac{\partial g}{\partial x}$ by hand, treating $y$ as
   constant.
2. Compute $\frac{\partial g}{\partial y}$ by hand, treating $x$ as
   constant.
3. Write $\nabla g(x,y)$ and evaluate it at the point $(1, 2)$ by hand.
4. Implement `grad_exact` and `grad_numeric` for $g$ in NumPy (following the
   pattern above) and confirm both match your hand-computed gradient at
   $(1,2)$.
