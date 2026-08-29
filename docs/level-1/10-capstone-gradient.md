# 10 · Capstone — Derive & Verify a Gradient

This capstone combines every skill from Level 1: reading vector/matrix
notation, computing partial derivatives, assembling a gradient, and
cross-checking hand work against NumPy — both symbolically and numerically.

## The task

Consider the two-variable cost function

$$
J(w,b) = (w-3)^2 + (b+2)^2 + wb
$$

This isn't tied to a specific dataset — it's a hand-crafted "toy" cost
surface, deliberately built so we can derive its gradient exactly and then
verify every step. (Real cost functions, like Module 9's linear regression
$J$, have the same structure: a sum of terms, each differentiable by the
rules from Modules 6–8.)

## Step 1: Derive $\frac{\partial J}{\partial w}$ by hand

Differentiate term by term (sum rule), treating $b$ as constant:

- $\frac{\partial}{\partial w}(w-3)^2$: chain rule, outer $u^2$ with
  $u=w-3$, inner derivative $\frac{\partial u}{\partial w}=1$, so this term
  gives $2(w-3)\cdot 1 = 2(w-3)$.
- $\frac{\partial}{\partial w}(b+2)^2 = 0$ (no $w$ in this term, so it's a
  constant with respect to $w$).
- $\frac{\partial}{\partial w}(wb) = b$ (treating $b$ as a constant
  multiplier of $w$, power rule on $w^1$).

$$
\frac{\partial J}{\partial w} = 2(w-3) + b
$$

## Step 2: Derive $\frac{\partial J}{\partial b}$ by hand

Symmetric process, treating $w$ as constant:

- $\frac{\partial}{\partial w \to b}(w-3)^2 = 0$ (no $b$ in this term).
- $\frac{\partial}{\partial b}(b+2)^2 = 2(b+2)$ (chain rule, same as above).
- $\frac{\partial}{\partial b}(wb) = w$ (treating $w$ as a constant
  multiplier of $b$).

$$
\frac{\partial J}{\partial b} = 2(b+2) + w
$$

## Step 3: Assemble the gradient

$$
\nabla J(w,b) = \begin{bmatrix} 2(w-3)+b \\ 2(b+2)+w \end{bmatrix}
$$

## Step 4: Evaluate by hand at a specific point

Let's evaluate at $(w,b) = (1, 1)$:

$$
\nabla J(1,1) = \begin{bmatrix} 2(1-3)+1 \\ 2(1+2)+1 \end{bmatrix} = \begin{bmatrix} 2(-2)+1 \\ 2(3)+1 \end{bmatrix} = \begin{bmatrix} -4+1 \\ 6+1 \end{bmatrix} = \begin{bmatrix} -3 \\ 7 \end{bmatrix}
$$

Also compute $J(1,1)$ itself for the cross-check:

$$
J(1,1) = (1-3)^2 + (1+2)^2 + (1)(1) = 4 + 9 + 1 = 14
$$

## Step 5: Verify in NumPy — symbolic gradient formula

```python
import numpy as np

def J(w, b):
    return (w - 3) ** 2 + (b + 2) ** 2 + w * b

def grad_exact(w, b):
    dJ_dw = 2 * (w - 3) + b
    dJ_db = 2 * (b + 2) + w
    return np.array([dJ_dw, dJ_db])

w0, b0 = 1.0, 1.0
print("J(1,1)          :", J(w0, b0))
print("exact gradient   :", grad_exact(w0, b0))
```

Expected output (matches Step 4's hand computation):

```text
J(1,1)          : 14.0
exact gradient   : [-3.  7.]
```

## Step 6: Verify with numerical (finite-difference) gradient

As a completely independent check — using no symbolic derivative rules at
all, just the raw definition of a partial derivative from Module 8 — confirm
the same result:

```python
def grad_numeric(w, b, h=1e-5):
    dJ_dw = (J(w + h, b) - J(w, b)) / h
    dJ_db = (J(w, b + h) - J(w, b)) / h
    return np.array([dJ_dw, dJ_db])

print("numeric gradient :", grad_numeric(w0, b0))
```

Expected output (matches the exact gradient to within finite-difference
error):

```text
numeric gradient : [-2.99999 7.00001]
```

The symbolic derivation (Steps 1–4), the direct formula in code (Step 5),
and the independent finite-difference check (Step 6) all agree on
$\nabla J(1,1) \approx [-3, 7]$ — this triple-agreement is the standard way
to trust a hand-derived gradient before using it in any real training loop.

## Step 7: Use the gradient to take one gradient-descent step

As a preview of Level 2, use $-\nabla J$ (Module 8's "steepest decrease"
direction) to take one small step from $(1,1)$ with learning rate
$\alpha=0.1$:

$$
(w,b)_{\text{new}} = (1,1) - 0.1\cdot(-3, 7) = (1+0.3,\ 1-0.7) = (1.3,\ 0.3)
$$

```python
alpha = 0.1
w_new, b_new = np.array([w0, b0]) - alpha * grad_exact(w0, b0)
print("new (w,b):", (w_new, b_new))
print("J before :", J(w0, b0), " J after:", J(w_new, b_new))
```

Expected output (cost should decrease after the step, confirming we moved
in a helpful direction; by hand: $J(1.3,0.3)=(1.3-3)^2+(0.3+2)^2+1.3(0.3) =
2.89+5.29+0.39 = 8.57$):

```text
new (w,b): (1.3, 0.30000000000000004)
J before : 14.0  J after: 8.57
```

$J$ dropped from $14.0$ to $8.57$ after a single step in the
$-\nabla J$ direction — exactly the mechanism that trains every ML model
built on gradient descent, from linear regression to deep neural networks.

## Exercise (final Level 1 check)

Given $K(w,b) = (2w+1)^2 + 3b^2 - wb$:

1. Derive $\frac{\partial K}{\partial w}$ and $\frac{\partial K}{\partial b}$
   by hand, showing the chain rule step for the squared term.
2. Evaluate $\nabla K$ by hand at $(w,b) = (0, 1)$.
3. Implement `K`, `grad_exact`, and `grad_numeric` in NumPy (following this
   module's pattern) and confirm all three -- hand, exact formula, and
   finite difference -- agree at $(0,1)$.
4. Take one gradient-descent step from $(0,1)$ with $\alpha=0.1$ and confirm
   $K$ decreases.
