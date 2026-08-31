# 02 · Loss Functions & Their Derivatives

Every training loop needs $\partial L/\partial \hat y$ (or directly
$\partial L/\partial z$) to start backpropagation. This module derives the
gradients for the two loss functions used in almost every model: **Mean
Squared Error** (regression) and **Binary Cross-Entropy** (classification).

## Mean Squared Error (MSE)

For predictions $\hat y_i$ and targets $y_i$ over $n$ examples:

$$
L = \frac{1}{n}\sum_{i=1}^n (\hat y_i - y_i)^2
$$

**Gradient w.r.t. a single prediction:**

$$
\frac{\partial L}{\partial \hat y_i} = \frac{2}{n}(\hat y_i - y_i)
$$

This is a direct application of the power rule plus linearity of
differentiation under a sum (Level 1 Module 7).

## Binary Cross-Entropy (BCE)

For predictions $\hat y_i \in (0,1)$ (e.g. sigmoid outputs) and binary
labels $y_i \in \{0,1\}$:

$$
L = -\frac{1}{n}\sum_{i=1}^n\big[y_i\ln\hat y_i + (1-y_i)\ln(1-\hat y_i)\big]
$$

This is exactly the negative log-likelihood from Level 2 Module 10, applied
per-example and averaged.

**Gradient w.r.t. a single prediction:**

$$
\frac{\partial L}{\partial \hat y_i} = \frac{1}{n}\left(\frac{1-y_i}{1-\hat y_i} - \frac{y_i}{\hat y_i}\right) = \frac{1}{n}\cdot\frac{\hat y_i - y_i}{\hat y_i(1-\hat y_i)}
$$

**Key simplification — BCE composed with sigmoid.** If
$\hat y_i = \sigma(z_i)$, chaining through $\sigma'(z)=\hat y(1-\hat y)$
(Module 1) makes the $\hat y_i(1-\hat y_i)$ terms **cancel exactly**:

$$
\frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial \hat y_i}\cdot\hat y_i(1-\hat y_i) = \frac{1}{n}(\hat y_i - y_i)
$$

This is why frameworks combine sigmoid + BCE into one numerically stable op
— the clean $(\hat y_i - y_i)$ gradient avoids ever dividing by
$\hat y_i(1-\hat y_i)$, which can be near zero.

## Worked example

Two examples: $\hat y = [0.8, 0.3]$, $y=[1, 0]$.

**MSE:**

$$
L_{MSE} = \frac{1}{2}\big[(0.8-1)^2+(0.3-0)^2\big] = \frac{1}{2}(0.04+0.09)=0.065
$$

$$
\nabla_{\hat y} L_{MSE} = \left[\frac{2}{2}(0.8-1), \frac{2}{2}(0.3-0)\right] = [-0.2, 0.3]
$$

**BCE:**

$$
L_{BCE} = -\frac{1}{2}\big[\ln(0.8) + \ln(0.7)\big] \approx -\frac{1}{2}(-0.2231-0.3567) \approx 0.2899
$$

$$
\frac{\partial L}{\partial z} \text{ (if }\hat y=\sigma(z)\text{)} = \frac{1}{2}[(0.8-1), (0.3-0)] = [-0.1, 0.15]
$$

## Numeric verification

```python
import numpy as np

y_hat = np.array([0.8, 0.3])
y = np.array([1.0, 0.0])
n = len(y)

# MSE
L_mse = np.mean((y_hat - y) ** 2)
grad_mse = (2 / n) * (y_hat - y)
print("MSE:", L_mse, "grad:", grad_mse)

# BCE
L_bce = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
grad_bce_dyhat = (1 / n) * ((1 - y) / (1 - y_hat) - y / y_hat)
grad_bce_dz = (1 / n) * (y_hat - y)  # simplified sigmoid+BCE gradient
print("BCE:", L_bce, "grad wrt y_hat:", grad_bce_dyhat, "grad wrt z:", grad_bce_dz)

# Finite-difference check of grad wrt y_hat for BCE
def bce(y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

h = 1e-6
grad_num = np.array([
    (bce(y_hat + h * e) - bce(y_hat - h * e)) / (2 * h)
    for e in np.eye(2)
])
print("numeric grad wrt y_hat:", grad_num)
```

Expected output:

```text
MSE: 0.065 grad: [-0.2  0.3]
BCE: 0.28986... grad wrt y_hat: [-0.625  0.7142857] grad wrt z: [-0.1  0.15]
numeric grad wrt y_hat: [-0.625  0.7142857]
```

## Exercise

Given $\hat y = [0.6, 0.9]$, $y = [0, 1]$:

1. Compute $L_{MSE}$ and $\nabla_{\hat y}L_{MSE}$ by hand.
2. Compute $L_{BCE}$ by hand.
3. Compute the simplified gradient $\partial L_{BCE}/\partial z$ (assuming
   $\hat y = \sigma(z)$) by hand.
4. Verify all results in NumPy, including a finite-difference check of the
   BCE gradient with respect to $\hat y$ directly (not through $z$).
