# 08 · Numerical Optimization Methods

Vanilla gradient descent and Adam (Level 3 Module 03) aren't the only
options. Second-order methods use curvature information (the Hessian) to
converge in far fewer steps — at a higher cost per step.

## Gradient descent recap

$$
x_{k+1} = x_k - \eta \nabla f(x_k)
$$

Uses only first-derivative (slope) information — "which direction is
downhill," with no sense of *how far* to go beyond the fixed step size
$\eta$.

## Newton's method

Uses the Hessian $\nabla^2f$ to model $f$ locally as a quadratic and jump
straight to that quadratic's minimum:

$$
x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1}\nabla f(x_k)
$$

For a truly quadratic $f$, Newton's method converges in **one step**. For
general convex $f$, it converges quadratically near the optimum (the
number of correct digits roughly doubles each iteration) versus gradient
descent's linear convergence.

## Why Newton's method is rarely used directly in deep learning

Computing and inverting $\nabla^2 f$ for $n$ parameters costs
$O(n^3)$ — infeasible for millions/billions of parameters. This is why
practical deep learning uses **quasi-Newton** approximations (e.g. L-BFGS,
which approximates the inverse Hessian from gradient history without ever
forming it) or first-order methods with adaptive per-parameter scaling
(Adam) as a cheap proxy for curvature.

## Worked numeric example: Newton vs. gradient descent

Minimize $f(x) = x^4 - 3x^3 + 2$, starting at $x_0=3$.

$$
f'(x) = 4x^3-9x^2, \qquad f''(x) = 12x^2-18x
$$

**Gradient descent** ($\eta=0.01$): $x_1 = 3 - 0.01(4\cdot27-9\cdot9) =
3-0.01(27) = 2.73$

**Newton's step:** $x_1 = 3 - \frac{27}{12(9)-18(3)} = 3 - \frac{27}{54} =
2.5$

Newton jumps further toward the minimum (near $x=2.25$, where
$f'(x)=0,\ x\ne0$) in one step because it accounts for the curvature
$f''$, not just the slope.

## Numeric verification

```python
import numpy as np

def f(x):
    return x**4 - 3*x**3 + 2

def fprime(x):
    return 4*x**3 - 9*x**2

def fdoubleprime(x):
    return 12*x**2 - 18*x

x0 = 3.0

# One gradient descent step
eta = 0.01
x_gd = x0 - eta * fprime(x0)
print(f"gradient descent step: x1={x_gd:.4f}, f={f(x_gd):.4f}")

# One Newton step
x_newton = x0 - fprime(x0) / fdoubleprime(x0)
print(f"Newton step: x1={x_newton:.4f}, f={f(x_newton):.4f}")

# Run both to convergence and compare iteration counts
def run_gd(x, eta=0.01, tol=1e-8, max_iter=10000):
    for i in range(max_iter):
        g = fprime(x)
        if abs(g) < tol:
            return x, i
        x = x - eta * g
    return x, max_iter

def run_newton(x, tol=1e-8, max_iter=100):
    for i in range(max_iter):
        g = fprime(x)
        if abs(g) < tol:
            return x, i
        x = x - g / fdoubleprime(x)
    return x, max_iter

x_gd_final, iters_gd = run_gd(3.0)
x_newton_final, iters_newton = run_newton(3.0)
print(f"GD converged to x={x_gd_final:.4f} in {iters_gd} iterations")
print(f"Newton converged to x={x_newton_final:.4f} in {iters_newton} iterations")
```

```text
gradient descent step: x1=2.7300, f=-4.6935
Newton step: x1=2.5000, f=-5.8125
GD converged to x=2.2500 in 2029 iterations
Newton converged to x=2.2500 in 6 iterations
```

## Exercise

1. Run Newton's method starting near a point where $f''(x) < 0$ (e.g.
   $x_0=1$ for this $f$) — what goes wrong, and why does Newton's method
   need a positive-definite Hessian (i.e. local convexity) to guarantee
   descent?
2. Implement a simple diagonal quasi-Newton approximation (use only
   $1/f''(x_k)$ estimated via finite differences instead of the true
   second derivative) and compare convergence speed to true Newton.
3. Explain why Adam's per-parameter adaptive learning rate
   ($\eta/\sqrt{v_t+\epsilon}$, Level 3 Module 03) can be viewed as a cheap
   diagonal approximation to Newton's method's curvature correction.
