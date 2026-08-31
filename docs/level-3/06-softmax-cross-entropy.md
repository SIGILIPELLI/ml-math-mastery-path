# 06 · Softmax & Cross-Entropy Derivatives

Softmax turns raw scores (logits) into a probability distribution;
cross-entropy measures how far that distribution is from the true label.
Together, their combined derivative is the cleanest gradient in all of deep
learning — and it's worth deriving by hand once.

## Softmax

For logits $z = (z_1,\dots,z_C)$:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}} = p_i
$$

## Cross-entropy loss

For one-hot true label $y$ (so $y_k=1$ for the true class $k$, else 0):

$$
L = -\sum_{i=1}^C y_i \log p_i = -\log p_k
$$

## The Jacobian of softmax

$$
\frac{\partial p_i}{\partial z_j} =
\begin{cases}
p_i(1-p_i) & i=j \\
-p_i p_j & i \ne j
\end{cases}
\;=\; p_i(\delta_{ij} - p_j)
$$

where $\delta_{ij}$ is the Kronecker delta.

## Combined gradient: the famous $p - y$

Chain rule through $L \to p \to z$:

$$
\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_j}
= \sum_i \left(-\frac{y_i}{p_i}\right) p_i(\delta_{ij}-p_j)
= -\sum_i y_i(\delta_{ij}-p_j)
$$

Since $\sum_i y_i = 1$:

$$
\frac{\partial L}{\partial z_j} = p_j - y_j
$$

The gradient of softmax + cross-entropy with respect to the **logits** is
simply predicted-minus-true. This is why frameworks fuse these two ops:
computing them separately (dividing by $p_i$, then multiplying by
$p_i(\delta_{ij}-p_j)$) is both slower and numerically worse than this
closed form.

## Worked numeric example

Logits $z=(2.0, 1.0, 0.1)$, true class $k=0$ (so $y=(1,0,0)$).

$$
e^z = (7.389, 2.718, 1.105), \qquad \sum = 11.212
$$

$$
p = (0.6590, 0.2424, 0.0986)
$$

$$
L = -\log(0.6590) \approx 0.4170
$$

$$
\nabla_z L = p - y = (-0.3410,\ 0.2424,\ 0.0986)
$$

## Numeric verification

```python
import numpy as np

def softmax(z):
    z = z - np.max(z)  # numerical stability, see Module 09
    e = np.exp(z)
    return e / e.sum()

z = np.array([2.0, 1.0, 0.1])
y = np.array([1.0, 0.0, 0.0])

p = softmax(z)
L = -np.sum(y * np.log(p))
grad_analytic = p - y

print(f"p = {p}")
print(f"L = {L:.4f}")
print(f"analytic grad = {grad_analytic}")

# Finite-difference check
def loss(z):
    p = softmax(z)
    return -np.sum(y * np.log(p))

h = 1e-6
grad_num = np.zeros_like(z)
for i in range(len(z)):
    z_plus, z_minus = z.copy(), z.copy()
    z_plus[i] += h
    z_minus[i] -= h
    grad_num[i] = (loss(z_plus) - loss(z_minus)) / (2 * h)

print(f"numeric grad = {grad_num}")
```

```text
p = [0.65900114 0.24243297 0.09856589]
L = 0.4170
analytic grad = [-0.34099886  0.24243297  0.09856589]
numeric grad = [-0.34099886  0.24243297  0.09856589]
```

## Exercise

1. Repeat the derivation and numeric check for true class $k=2$ instead of
   $k=0$.
2. Show that for binary classification, sigmoid + binary cross-entropy
   gives the same clean gradient form $\hat p - y$ (treat it as a
   two-class softmax with $z_2=0$).
3. Explain why subtracting $\max(z)$ before exponentiating (as in the code
   above) doesn't change $p$ but prevents overflow — tie this to Module 09.
