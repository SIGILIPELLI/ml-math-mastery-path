# 01 · Convexity & Optimization

Convexity is why gradient descent is *guaranteed* to find the global
minimum for some problems (linear/logistic regression) but only a local
one for others (neural networks). Understanding it tells you when to trust
"it converged" as "it found the best answer."

## Convex sets and functions

A set $S$ is convex if the line segment between any two points in $S$
stays in $S$: $\theta x + (1-\theta)y \in S$ for all $x,y\in S,\
\theta\in[0,1]$.

A function $f$ is convex if its domain is convex and:

$$
f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y) \qquad \forall\, \theta\in[0,1]
$$

Geometrically: the chord between any two points on the graph lies **above**
the graph.

## Second-derivative test

For twice-differentiable $f:\mathbb{R}\to\mathbb{R}$: $f$ is convex iff
$f''(x)\ge 0$ everywhere. In $n$ dimensions: $f$ is convex iff its Hessian
$\nabla^2 f(x)$ is positive semi-definite (all eigenvalues $\ge 0$) for
every $x$ — this connects directly to Level 2 Module 4 (eigenvalues).

## Why convexity matters for optimization

**Theorem:** for a convex function, every local minimum is a global
minimum. There are no "traps." This is exactly why linear regression
(MSE is convex in the weights) and logistic regression (cross-entropy is
convex in the weights) always converge to the *global* optimum with
gradient descent, while neural network loss landscapes (non-convex) offer
no such guarantee — gradient descent there can get stuck in local minima
or saddle points.

## Worked numeric example

Check convexity of $f(w_1,w_2) = w_1^2 + 2w_2^2 - w_1w_2$ via its Hessian:

$$
\nabla^2 f = \begin{pmatrix}2 & -1\\-1 & 4\end{pmatrix}
$$

Eigenvalues: solve $\det(\nabla^2 f - \lambda I)=0 \Rightarrow
(2-\lambda)(4-\lambda)-1=0 \Rightarrow \lambda^2-6\lambda+7=0 \Rightarrow
\lambda = 3\pm\sqrt2 \approx \{1.586, 4.414\}$. Both positive
$\Rightarrow$ Hessian is positive definite $\Rightarrow f$ is (strictly)
convex.

## Numeric verification

```python
import numpy as np

H = np.array([[2.0, -1.0], [-1.0, 4.0]])
eigvals = np.linalg.eigvalsh(H)
print(f"Hessian eigenvalues = {eigvals}")
print(f"convex (all >= 0)? {np.all(eigvals >= 0)}")

# Verify the convexity inequality directly at a few random points
def f(w):
    return w[0]**2 + 2*w[1]**2 - w[0]*w[1]

rng = np.random.default_rng(0)
violations = 0
for _ in range(1000):
    x, y = rng.normal(size=2), rng.normal(size=2)
    theta = rng.uniform(0, 1)
    lhs = f(theta * x + (1 - theta) * y)
    rhs = theta * f(x) + (1 - theta) * f(y)
    if lhs > rhs + 1e-9:
        violations += 1
print(f"convexity inequality violations out of 1000 random checks: {violations}")
```

```text
Hessian eigenvalues = [1.58578644 4.41421356]
convex (all >= 0)? True
convexity inequality violations out of 1000 random checks: 0
```

## Exercise

1. Check whether $g(w) = w_1^2 - 2w_2^2$ is convex by computing its Hessian
   eigenvalues. What does a negative eigenvalue mean about the shape of the
   surface (hint: saddle point)?
2. Prove that the sum of two convex functions is convex (directly from the
   definition), and confirm MSE loss $\frac{1}{n}\sum(y_i-w^\top x_i)^2$ is
   convex in $w$ by identifying its Hessian as $\frac{2}{n}X^\top X$, which
   is always PSD.
3. Explain, using the eigenvalues of the MSE Hessian ($\propto X^\top X$'s
   condition number), why gradient descent converges slowly when features
   are on very different scales — tie this back to Level 1's feature
   scaling motivation.
