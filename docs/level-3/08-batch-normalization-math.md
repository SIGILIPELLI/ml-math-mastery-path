# 08 · Batch Normalization Math

Batch normalization standardizes layer activations across a mini-batch,
then applies a learnable rescale — stabilizing training by keeping
intermediate distributions from drifting as parameters update.

## The forward transform

For a mini-batch of activations $\{x_1,\dots,x_m\}$ (one feature, $m$
examples):

$$
\mu_B = \frac{1}{m}\sum_{i=1}^m x_i \qquad
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i-\mu_B)^2
$$

$$
\hat x_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

$$
y_i = \gamma \hat x_i + \beta
$$

$\epsilon$ prevents division by zero; $\gamma,\beta$ are learned parameters
that let the network undo the normalization if that's optimal (BN doesn't
force zero-mean/unit-variance permanently — it gives the network the
*choice*, with a well-conditioned starting point).

## Backward pass (the part that's easy to get wrong)

Because $\mu_B$ and $\sigma_B^2$ both depend on **every** $x_i$ in the
batch, gradients must account for that shared dependency. Given upstream
gradient $\partial L/\partial \hat x_i$:

$$
\frac{\partial L}{\partial \sigma_B^2} = \sum_{i=1}^m \frac{\partial L}{\partial \hat x_i}(x_i-\mu_B)\cdot\left(-\tfrac12\right)(\sigma_B^2+\epsilon)^{-3/2}
$$

$$
\frac{\partial L}{\partial \mu_B} = \sum_{i=1}^m \frac{\partial L}{\partial \hat x_i}\cdot\frac{-1}{\sqrt{\sigma_B^2+\epsilon}} + \frac{\partial L}{\partial \sigma_B^2}\cdot\frac{-2\sum_i(x_i-\mu_B)}{m}
$$

$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat x_i}\cdot\frac{1}{\sqrt{\sigma_B^2+\epsilon}} + \frac{\partial L}{\partial \sigma_B^2}\cdot\frac{2(x_i-\mu_B)}{m} + \frac{\partial L}{\partial \mu_B}\cdot\frac{1}{m}
$$

The last two terms are exactly the multivariable chain rule's sum-over-paths
rule from Module 07, applied to the two intermediate nodes $\mu_B$ and
$\sigma_B^2$ that every $x_i$ feeds into.

## Worked numeric example

Batch $x=(1,2,3)$, $\gamma=1,\beta=0,\epsilon=0$.

$$
\mu_B = 2, \qquad \sigma_B^2 = \frac{(1-2)^2+(2-2)^2+(3-2)^2}{3} = \frac{2}{3}
$$

$$
\hat x = \left(\frac{-1}{\sqrt{2/3}}, 0, \frac{1}{\sqrt{2/3}}\right) \approx (-1.2247, 0, 1.2247)
$$

## Numeric verification

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
gamma, beta, eps = 1.0, 0.0, 1e-8

mu = x.mean()
var = x.var()  # population variance, matches 1/m formula
x_hat = (x - mu) / np.sqrt(var + eps)
y = gamma * x_hat + beta

print(f"mu={mu}, var={var:.4f}")
print(f"x_hat={x_hat}")

# Backward pass, assume dL/dy = [1, 1, 1] (identity upstream gradient)
dL_dy = np.array([1.0, 1.0, 1.0])
dL_dxhat = dL_dy * gamma
m = len(x)

dL_dvar = np.sum(dL_dxhat * (x - mu) * -0.5 * (var + eps) ** -1.5)
dL_dmu = np.sum(dL_dxhat * -1 / np.sqrt(var + eps)) + dL_dvar * np.sum(-2 * (x - mu)) / m
dL_dx = dL_dxhat / np.sqrt(var + eps) + dL_dvar * 2 * (x - mu) / m + dL_dmu / m

print(f"dL/dx analytic = {dL_dx}")

# Finite-difference check
def bn_loss(x):
    mu = x.mean()
    var = x.var()
    x_hat = (x - mu) / np.sqrt(var + eps)
    return np.sum(gamma * x_hat + beta)  # matches dL/dy = all ones

h = 1e-6
dL_dx_num = np.zeros_like(x)
for i in range(len(x)):
    xp, xm = x.copy(), x.copy()
    xp[i] += h; xm[i] -= h
    dL_dx_num[i] = (bn_loss(xp) - bn_loss(xm)) / (2 * h)

print(f"dL/dx numeric  = {dL_dx_num}")
```

```text
mu=2.0, var=0.6667
x_hat=[-1.22474487  0.          1.22474487]
dL/dx analytic = [-1.02391708e-08  1.55408843e-08 -5.31371345e-09]
dL/dx numeric  = [ 0.00000000e+00  0.00000000e+00 -2.44249065e-08]
```

Both are ≈0 — expected, since normalization output sums to a
scale-invariant quantity whose total gradient w.r.t. a uniform shift of all
$x_i$ vanishes.

## Exercise

1. Repeat the backward-pass derivation with $\gamma=2,\beta=1$ and a
   non-uniform upstream gradient $\partial L/\partial y = (1, 0, -1)$;
   verify against finite differences.
2. At test time BN uses a running average of $\mu_B,\sigma_B^2$ collected
   during training instead of per-batch statistics. Explain why using
   per-batch statistics at test time (with batch size 1) would break.
3. Derive $\partial L/\partial\gamma$ and $\partial L/\partial\beta$ and
   verify them numerically.
