# 09 · Linear Regression as a Math Example

This module ties together every tool from Level 1 — vectors, dot products,
matrix multiplication, derivatives, partial derivatives, and gradients — in
one real algorithm: **linear regression**, both its cost function and its
closed-form solution.

## The model

For a single feature $x$, the model predicts

$$
\hat{y} = wx + b
$$

($w$ = weight/slope, $b$ = bias/intercept — the same line from Module 5,
now wearing ML terminology). With $n$ training examples $(x_i, y_i)$, we
want $w$ and $b$ that make $\hat{y}_i$ close to $y_i$ for every $i$.

## The cost function

We measure "closeness" with **mean squared error (MSE)**:

$$
J(w,b) = \frac{1}{n}\sum_{i=1}^n \left(wx_i+b-y_i\right)^2
$$

This is a function of two variables, $w$ and $b$ — exactly the setting
Module 8 built tools for. $J$ is a sum of squared terms, each of which is a
composite function (an "outer" square applied to an "inner" linear
expression) — exactly the chain-rule setting from Module 7.

## Deriving the gradient of $J$ by hand

Treat each summand $(wx_i+b-y_i)^2$ as $u_i^2$ where $u_i = wx_i+b-y_i$.
By the chain rule, $\frac{\partial}{\partial w}\left[u_i^2\right] = 2u_i \cdot
\frac{\partial u_i}{\partial w}$. Since $u_i = wx_i+b-y_i$, we have
$\frac{\partial u_i}{\partial w} = x_i$ (treating $b, y_i$ as constants w.r.t.
$w$) and $\frac{\partial u_i}{\partial b} = 1$.

$$
\frac{\partial J}{\partial w} = \frac{1}{n}\sum_{i=1}^n 2(wx_i+b-y_i)\cdot x_i = \frac{2}{n}\sum_{i=1}^n x_i(wx_i+b-y_i)
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^n 2(wx_i+b-y_i)\cdot 1 = \frac{2}{n}\sum_{i=1}^n (wx_i+b-y_i)
$$

So the gradient is

$$
\nabla J(w,b) = \begin{bmatrix} \dfrac{2}{n}\sum_i x_i(wx_i+b-y_i) \\[6pt] \dfrac{2}{n}\sum_i (wx_i+b-y_i) \end{bmatrix}
$$

This is precisely the "capstone formula" we will apply numerically in
Module 10, and derive again for the general multi-feature case in Level 2.

## The closed-form solution

Because $J$ is quadratic in $(w,b)$ (a bowl shape, per Module 5), it has a
single global minimum, found by setting both partial derivatives to zero and
solving. For a single feature, this gives the classic formulas (derivation
skipped here — it's straightforward algebra on the two equations above set
to zero):

$$
w^* = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sum_i (x_i-\bar{x})^2}
\qquad
b^* = \bar{y} - w^*\bar{x}
$$

where $\bar{x}, \bar{y}$ are the means of $x$ and $y$. This is the
"least-squares" formula — no iteration needed, unlike gradient descent
(Level 2), because the problem is simple enough to solve directly.

In matrix form, with $\mathbf{X}$ the design matrix (a column of features
plus a column of 1s for the bias) and $\mathbf{w}$ the stacked
$[w, b]^T$, the same solution is the **normal equation**:

$$
\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

— note the $\mathbf{X}^T\mathbf{X}$ term, exactly the transpose-then-multiply
pattern introduced in Module 4.

## Worked numeric example

Using our running example points $(1,2), (2,3), (3,5)$ from Module 1:

$\bar{x} = \frac{1+2+3}{3}=2$, $\bar{y}=\frac{2+3+5}{3}=\frac{10}{3}\approx3.333$.

$$
w^* = \frac{(1-2)(2-3.333)+(2-2)(3-3.333)+(3-2)(5-3.333)}{(1-2)^2+(2-2)^2+(3-2)^2}
= \frac{(-1)(-1.333)+0+(1)(1.667)}{1+0+1} = \frac{1.333+1.667}{2} = \frac{3.0}{2} = 1.5
$$

$$
b^* = 3.333 - 1.5(2) = 3.333 - 3 = 0.333
$$

So the best-fit line is $\hat{y} = 1.5x + 0.333$.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 5.0])

# closed-form via means (matches hand derivation)
x_bar, y_bar = x.mean(), y.mean()
w_star = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
b_star = y_bar - w_star * x_bar
print("w* =", round(w_star, 3), " b* =", round(b_star, 3))

# cross-check via the normal equation in matrix form
X = np.column_stack([x, np.ones_like(x)])  # design matrix [x, 1]
w_matrix = np.linalg.inv(X.T @ X) @ X.T @ y
print("normal-equation [w, b]:", w_matrix)

# cross-check the gradient at (w*, b*) should be ~[0, 0] -- the minimum
def grad(w, b):
    pred = w * x + b
    dJ_dw = np.mean(2 * x * (pred - y))
    dJ_db = np.mean(2 * (pred - y))
    return np.array([dJ_dw, dJ_db])

print("gradient at optimum:", grad(w_star, b_star))
```

Expected output (matches hand computation; gradient at the optimum should be
essentially zero, confirming $(w^*,b^*)$ really is where $J$ is flat):

```text
w* = 1.5  b* = 0.333
normal-equation [w, b]: [1.5        0.33333333]
gradient at optimum: [0. 0.]
```

## Exercise

Using the points $(1,1), (2,2), (3,2), (4,3)$:

1. Compute $\bar{x}$ and $\bar{y}$ by hand.
2. Use the closed-form formulas above to compute $w^*$ and $b^*$ by hand.
3. Confirm your answer with the NumPy `w_star`/`b_star` code above (swap in
   the new data).
4. Compute the gradient $\nabla J$ at your $(w^*, b^*)$ using the `grad`
   function and confirm it is approximately $[0, 0]$.
