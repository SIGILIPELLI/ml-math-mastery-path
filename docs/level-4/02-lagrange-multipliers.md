# 02 · Lagrange Multipliers & Constrained Optimization

Many ML problems aren't "minimize $f$" but "minimize $f$ subject to
constraints" — SVM margins, probability simplex constraints, resource
budgets. Lagrange multipliers convert a constrained problem into an
unconstrained one.

## The setup

Minimize $f(x)$ subject to equality constraint $g(x)=0$. At the
constrained optimum, the gradient of $f$ must be **parallel** to the
gradient of $g$ (otherwise you could slide along the constraint surface
and decrease $f$ further). That parallelism is written:

$$
\nabla f(x) = \lambda \nabla g(x)
$$

$\lambda$ is the Lagrange multiplier. Package this into the **Lagrangian**:

$$
\mathcal{L}(x,\lambda) = f(x) - \lambda g(x)
$$

Setting $\nabla_x \mathcal{L}=0$ recovers the parallelism condition;
setting $\partial\mathcal{L}/\partial\lambda=0$ recovers the constraint
$g(x)=0$. Solving the stationary points of $\mathcal{L}$ jointly solves the
constrained problem.

## Worked example: closest point on a line to the origin

Minimize $f(x,y)=x^2+y^2$ subject to $g(x,y)=x+y-1=0$.

$$
\mathcal{L} = x^2+y^2-\lambda(x+y-1)
$$

$$
\frac{\partial\mathcal{L}}{\partial x}=2x-\lambda=0,\quad
\frac{\partial\mathcal{L}}{\partial y}=2y-\lambda=0,\quad
\frac{\partial\mathcal{L}}{\partial\lambda}=-(x+y-1)=0
$$

From the first two: $x=y=\lambda/2$. Substituting into the constraint:
$\lambda/2+\lambda/2=1\Rightarrow\lambda=1\Rightarrow x=y=0.5$.

Minimum value: $f(0.5,0.5)=0.5$.

## Inequality constraints: KKT conditions

For $\min f(x)$ s.t. $h(x)\le 0$, the Karush-Kuhn-Tucker (KKT) conditions
generalize Lagrange multipliers:

$$
\nabla f(x) + \mu \nabla h(x) = 0, \qquad \mu \ge 0, \qquad \mu\, h(x) = 0
$$

The last condition (**complementary slackness**) says either the
constraint is exactly active ($h(x)=0$) or its multiplier is zero — this
is precisely the mechanism behind SVM support vectors: only points on the
margin ($h(x)=0$) get nonzero $\mu$ (their "support" in support vector
machine).

## Numeric verification

```python
import numpy as np
from scipy.optimize import minimize

def f(v):
    x, y = v
    return x**2 + y**2

constraint = {'type': 'eq', 'fun': lambda v: v[0] + v[1] - 1}
result = minimize(f, x0=[0, 0], constraints=[constraint])
print(f"numeric optimum: x={result.x[0]:.4f}, y={result.x[1]:.4f}, f={result.fun:.4f}")
print(f"closed-form (Lagrange): x=0.5000, y=0.5000, f=0.5000")

# Verify the parallel-gradients condition at the optimum
def grad_f(v):
    return np.array([2*v[0], 2*v[1]])

def grad_g(v):
    return np.array([1.0, 1.0])

x_star = result.x
lam = grad_f(x_star)[0] / grad_g(x_star)[0]
print(f"grad f = {grad_f(x_star)}, lambda*grad g = {lam * grad_g(x_star)}")
```

```text
numeric optimum: x=0.5000, y=0.5000, f=0.5000
closed-form (Lagrange): x=0.5000, y=0.5000, f=0.5000
grad f = [1. 1.], lambda*grad g = [1. 1.]
```

## Exercise

1. Minimize $f(x,y)=xy$ subject to $x^2+y^2=1$ using Lagrange multipliers
   by hand, then verify with `scipy.optimize.minimize`.
2. Set up the Lagrangian for maximizing entropy $H(p)=-\sum_i p_i\log p_i$
   subject to $\sum_i p_i = 1$ (the probability simplex constraint), and
   show the unconstrained-optimum solution is the uniform distribution.
3. Explain, in your own words, why complementary slackness ($\mu\,h(x)=0$)
   is exactly why most points in an SVM's training set have zero influence
   on the decision boundary — only the "support vectors" do.
