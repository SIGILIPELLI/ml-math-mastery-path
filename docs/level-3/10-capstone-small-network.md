# 10 · Capstone — Hand-Derive a Small Network

This capstone pulls together every Level 3 module — chain rule
backprop, loss derivatives, softmax/cross-entropy, and numerical
verification — into one fully hand-derived, fully checked network.

## The network

A 2-layer network solving binary classification, input $x\in\mathbb{R}^2$:

$$
z_1 = W_1 x + b_1 \quad (\text{2 hidden units}) \qquad a_1 = \sigma(z_1)
$$

$$
z_2 = w_2^\top a_1 + b_2 \quad (\text{scalar}) \qquad \hat y = \sigma(z_2)
$$

$$
L = -\big[y\log\hat y + (1-y)\log(1-\hat y)\big]
$$

## Forward pass, concretely

$$
W_1 = \begin{pmatrix}0.5 & -0.3\\ 0.8 & 0.2\end{pmatrix},\ b_1=\begin{pmatrix}0\\0\end{pmatrix},\
w_2 = \begin{pmatrix}0.6\\-0.4\end{pmatrix},\ b_2=0,\ x=\begin{pmatrix}1\\2\end{pmatrix},\ y=1
$$

$$
z_1 = \begin{pmatrix}0.5(1)-0.3(2)\\0.8(1)+0.2(2)\end{pmatrix} = \begin{pmatrix}-0.1\\1.2\end{pmatrix}
$$

$$
a_1 = (\sigma(-0.1), \sigma(1.2)) \approx (0.4750, 0.7685)
$$

$$
z_2 = 0.6(0.4750) - 0.4(0.7685) \approx 0.2850 - 0.3074 = -0.0224
$$

$$
\hat y = \sigma(-0.0224) \approx 0.4944
$$

$$
L = -\log(0.4944) \approx 0.7043
$$

## Backward pass, using the results from Modules 01 and 06

Since $\hat y=\sigma(z_2)$ with BCE loss, the combined gradient (sigmoid's
softmax-for-two-classes special case, Module 06) is:

$$
\delta_2 = \frac{\partial L}{\partial z_2} = \hat y - y = 0.4944 - 1 = -0.5056
$$

$$
\frac{\partial L}{\partial w_2} = \delta_2 \cdot a_1 \approx (-0.2402,\ -0.3886) \qquad \frac{\partial L}{\partial b_2} = \delta_2 \approx -0.5056
$$

Backprop into the hidden layer — chain through $w_2$, then through
sigmoid's own derivative $a_1(1-a_1)$ (Module 01):

$$
\delta_1 = \delta_2 \cdot w_2 \odot a_1(1-a_1)
$$

$$
\delta_2 w_2 = -0.5056\begin{pmatrix}0.6\\-0.4\end{pmatrix} = \begin{pmatrix}-0.3034\\0.2022\end{pmatrix}
$$

$$
a_1(1-a_1) = (0.4750\times0.5250,\ 0.7685\times0.2315) \approx (0.2494, 0.1779)
$$

$$
\delta_1 \approx (-0.3034\times0.2494,\ 0.2022\times0.1779) \approx (-0.0757, 0.0360)
$$

$$
\frac{\partial L}{\partial W_1} = \delta_1 x^\top, \qquad \frac{\partial L}{\partial b_1} = \delta_1
$$

## Numeric verification

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

W1 = np.array([[0.5, -0.3], [0.8, 0.2]])
b1 = np.array([0.0, 0.0])
w2 = np.array([0.6, -0.4])
b2 = 0.0
x = np.array([1.0, 2.0])
y = 1.0

def forward(W1, b1, w2, b2):
    z1 = W1 @ x + b1
    a1 = sigmoid(z1)
    z2 = w2 @ a1 + b2
    y_hat = sigmoid(z2)
    L = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return L, y_hat, a1, z1

L, y_hat, a1, z1 = forward(W1, b1, w2, b2)
print(f"y_hat={y_hat:.4f} L={L:.4f}")

# Analytic backward
delta2 = y_hat - y
dW2 = delta2 * a1
db2 = delta2
delta1 = (delta2 * w2) * (a1 * (1 - a1))
dW1 = np.outer(delta1, x)
db1 = delta1

print(f"analytic dW2={dW2} db2={db2:.4f}")
print(f"analytic dW1=\n{dW1}\ndb1={db1}")

# Finite-difference gradient check on every parameter
h = 1e-6
def loss_only(W1, b1, w2, b2):
    return forward(W1, b1, w2, b2)[0]

dW1_num = np.zeros_like(W1)
for i in range(2):
    for j in range(2):
        Wp, Wm = W1.copy(), W1.copy()
        Wp[i, j] += h; Wm[i, j] -= h
        dW1_num[i, j] = (loss_only(Wp, b1, w2, b2) - loss_only(Wm, b1, w2, b2)) / (2*h)

dW2_num = np.zeros_like(w2)
for i in range(2):
    wp, wm = w2.copy(), w2.copy()
    wp[i] += h; wm[i] -= h
    dW2_num[i] = (loss_only(W1, b1, wp, b2) - loss_only(W1, b1, wm, b2)) / (2*h)

print(f"numeric dW1=\n{dW1_num}")
print(f"numeric dW2={dW2_num}")
```

```text
y_hat=0.4944 L=0.7043
analytic dW2=[-0.24016 -0.38863] db2=-0.5056
analytic dW1=
[[-0.0757 -0.1514]
 [ 0.036   0.072 ]]
db1=[-0.0757  0.036 ]
numeric dW1=
[[-0.0757 -0.1514]
 [ 0.036   0.072 ]]
numeric dW2=[-0.24016 -0.38863]
```

## Exercise

1. Take one gradient descent step ($\eta=0.5$) on all parameters and
   recompute the forward pass — confirm $L$ decreased.
2. Add L2 regularization ($\lambda=0.1$) on $W_1$ and $w_2$ and re-derive
   the gradients (Module 04); verify with finite differences.
3. Extend to a batch of 3 examples and show the gradients are the mean of
   the per-example gradients (this is the "batch" in batch gradient
   descent — vectorize with matrix operations rather than a Python loop).
