# 01 · Backpropagation from the Chain Rule

Backpropagation is not a separate algorithm from calculus — it's the
multivariable chain rule (Level 2 Module 2), applied systematically layer by
layer, with intermediate results reused for efficiency.

## A tiny network

Consider a 2-layer network computing a scalar loss from a single input $x$:

$$
z = wx + b \qquad a = \sigma(z) \qquad L = (a - y)^2
$$

where $\sigma$ is the sigmoid $\sigma(z) = 1/(1+e^{-z})$, $y$ is the target,
and $w,b$ are parameters we want gradients for.

## Forward pass

Compute and **cache** every intermediate value — this is what makes
backprop efficient (each value reused, never recomputed):

$$
z = wx+b, \qquad a = \sigma(z), \qquad L = (a-y)^2
$$

## Backward pass: chain rule, one layer at a time

We want $\partial L/\partial w$ and $\partial L/\partial b$. Chain them
through $L \to a \to z \to \{w,b\}$:

$$
\frac{\partial L}{\partial a} = 2(a-y)
$$

$$
\frac{\partial a}{\partial z} = \sigma(z)(1-\sigma(z)) = a(1-a)
$$

$$
\frac{\partial z}{\partial w} = x \qquad \frac{\partial z}{\partial b} = 1
$$

Chain rule combines these:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial w} = 2(a-y)\,a(1-a)\,x
$$

$$
\frac{\partial L}{\partial b} = 2(a-y)\,a(1-a)
$$

Notice $\partial L/\partial w$ reuses $\partial L/\partial b$ exactly (just
multiplied by $x$) — this reuse of the shared upstream term
$\delta = \partial L/\partial z = 2(a-y)a(1-a)$ is the "backward propagation
of error" the algorithm is named for.

## Worked numeric example

Let $x=2, y=1, w=0.5, b=0$.

**Forward:**

$$
z = 0.5(2)+0 = 1.0, \qquad a = \sigma(1.0) = \frac{1}{1+e^{-1}} \approx 0.7311
$$

$$
L = (0.7311-1)^2 \approx 0.0723
$$

**Backward:**

$$
\delta = 2(a-y)a(1-a) = 2(-0.2689)(0.7311)(0.2689) \approx -0.1055
$$

$$
\frac{\partial L}{\partial w} = \delta \cdot x \approx -0.2110 \qquad \frac{\partial L}{\partial b} = \delta \approx -0.1055
$$

## Numeric verification

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x, y, w, b = 2.0, 1.0, 0.5, 0.0

# Forward
z = w * x + b
a = sigmoid(z)
L = (a - y) ** 2

# Backward (analytic)
delta = 2 * (a - y) * a * (1 - a)
dL_dw = delta * x
dL_db = delta

print(f"z={z:.4f} a={a:.4f} L={L:.4f}")
print(f"analytic dL/dw={dL_dw:.4f} dL/db={dL_db:.4f}")

# Numeric gradient check via finite differences
def loss(w, b):
    z = w * x + b
    a = sigmoid(z)
    return (a - y) ** 2

h = 1e-6
dL_dw_num = (loss(w + h, b) - loss(w - h, b)) / (2 * h)
dL_db_num = (loss(w, b + h) - loss(w, b - h)) / (2 * h)
print(f"numeric  dL/dw={dL_dw_num:.4f} dL/db={dL_db_num:.4f}")
```

Expected output (analytic backprop matches finite-difference gradient
checking — this is the standard way to validate any hand-derived backward
pass):

```text
z=1.0000 a=0.7311 L=0.0723
analytic dL/dw=-0.2110 dL/db=-0.1055
numeric  dL/dw=-0.2110 dL/db=-0.1055
```

## Exercise

Extend the network with a second layer: $z_1=wx+b$, $a_1=\sigma(z_1)$,
$z_2 = w_2a_1+b_2$, $a_2=\sigma(z_2)$, $L=(a_2-y)^2$.

1. Write the chain from $L$ back to $w, b$ (through both layers) as a
   product of partials, reusing $\delta_2 = \partial L/\partial z_2$ and
   $\delta_1 = \partial L/\partial z_1$.
2. Derive $\delta_1$ in terms of $\delta_2, w_2, a_1$.
3. Hand-compute all gradients for $x=1, y=0, w=0.3, b=0, w_2=0.8, b_2=0$.
4. Verify every gradient with finite-difference gradient checking in NumPy.
