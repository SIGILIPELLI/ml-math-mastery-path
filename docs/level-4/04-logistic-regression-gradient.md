# 04 · Logistic Regression Gradient from Scratch

Logistic regression is the cleanest place to see MLE, the chain rule, and
convexity all fit together into one working classifier — derived here
completely from scratch, gradient **and** Hessian.

## Model

$$
\hat y = \sigma(w^\top x + b) = \frac{1}{1+e^{-(w^\top x+b)}}
$$

## Loss: negative log-likelihood of a Bernoulli (Level 3 Module 05)

For $n$ examples:

$$
L(w,b) = -\frac{1}{n}\sum_{i=1}^n\left[y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)\right]
$$

## Gradient

Using the fused sigmoid+cross-entropy result from Level 3 Module 06,
$\partial L_i/\partial z_i = \hat y_i - y_i$ where $z_i=w^\top x_i+b$.
Chaining through $z_i = w^\top x_i + b$:

$$
\nabla_w L = \frac{1}{n}\sum_{i=1}^n (\hat y_i - y_i)x_i = \frac{1}{n}X^\top(\hat y - y)
$$

$$
\frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^n(\hat y_i-y_i)
$$

This has the same clean form as **linear regression's** gradient
($X^\top(\hat y - y)$) — only the definition of $\hat y$ (sigmoid vs.
identity) differs.

## Hessian: proof of convexity

Differentiating the gradient once more (chain rule, using
$\hat y_i(1-\hat y_i)$ from Level 3 Module 01):

$$
\nabla_w^2 L = \frac{1}{n}X^\top S X, \qquad S = \text{diag}(\hat y_i(1-\hat y_i))
$$

Since $\hat y_i(1-\hat y_i) > 0$ for all $\hat y_i \in (0,1)$, $S$ is
positive definite, so $\nabla_w^2 L = X^\top S X$ is positive semi-definite
for any $X$ — proving (Level 4 Module 01) that logistic regression's loss
is **convex**: gradient descent always finds the global optimum.

## Worked numeric example

$x=(1,2), y=1, w=(0.1,-0.2), b=0$.

$$
z = 0.1(1)-0.2(2)+0 = -0.3, \qquad \hat y = \sigma(-0.3) \approx 0.4256
$$

$$
\nabla_w L = (\hat y-y)x = (-0.5744)(1,2) = (-0.5744, -1.1488)
$$

$$
\frac{\partial L}{\partial b} = \hat y - y = -0.5744
$$

## Numeric verification

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.array([1.0, 2.0])
y = 1.0
w = np.array([0.1, -0.2])
b = 0.0

z = w @ x + b
y_hat = sigmoid(z)
grad_w = (y_hat - y) * x
grad_b = y_hat - y

print(f"z={z:.4f} y_hat={y_hat:.4f}")
print(f"analytic grad_w={grad_w} grad_b={grad_b:.4f}")

def loss(w, b):
    z = w @ x + b
    yh = sigmoid(z)
    return -(y*np.log(yh) + (1-y)*np.log(1-yh))

h = 1e-6
grad_w_num = np.array([
    (loss(w + np.array([h,0]), b) - loss(w - np.array([h,0]), b)) / (2*h),
    (loss(w + np.array([0,h]), b) - loss(w - np.array([0,h]), b)) / (2*h),
])
grad_b_num = (loss(w, b+h) - loss(w, b-h)) / (2*h)
print(f"numeric  grad_w={grad_w_num} grad_b={grad_b_num:.4f}")

# Hessian PSD check (single example, 2 features)
S = y_hat * (1 - y_hat)
X = x.reshape(1, -1)
H = S * (X.T @ X)
print(f"Hessian=\n{H}\neigenvalues={np.linalg.eigvalsh(H)}  (should be >= 0)")
```

```text
z=-0.3000 y_hat=0.4256
analytic grad_w=[-0.5744 -1.1488] grad_b=-0.5744
numeric  grad_w=[-0.5744 -1.1488] grad_b=-0.5744
Hessian=
[[0.2445 0.489 ]
 [0.489  0.978 ]]
eigenvalues=[0.     1.2225]  (should be >= 0)
```

## Exercise

1. Implement full-batch gradient descent for logistic regression on a
   small synthetic 2D dataset (e.g. `sklearn.datasets.make_classification`)
   and verify the decision boundary visually.
2. Add L2 regularization (Level 3 Module 04) to the gradient and Hessian;
   show the regularized Hessian $\frac{1}{n}X^\top SX + 2\lambda I$ is
   strictly positive definite even when $X^\top SX$ alone is only PSD.
3. Implement one step of Newton's method ($w \leftarrow w - H^{-1}\nabla_w
   L$) and compare its convergence speed to gradient descent on the same
   toy dataset.
