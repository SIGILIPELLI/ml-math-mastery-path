# 05 · Kernel Methods & the Kernel Trick

Kernel methods let linear algorithms (like SVMs) draw non-linear decision
boundaries — without ever explicitly computing the high-dimensional
feature space the "non-linear" boundary lives in.

## The idea: feature maps

If data isn't linearly separable in $\mathbb{R}^d$, map it to a
higher-dimensional space via $\phi:\mathbb{R}^d\to\mathbb{R}^D$ ($D\gg d$)
where it *is* linearly separable. E.g. for $x=(x_1,x_2)$:

$$
\phi(x) = (x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2)
$$

turns a circular decision boundary in 2D into a linear one in this 3D
feature space.

## The kernel trick

Many algorithms (SVM, ridge regression, PCA) only need **dot products**
$\phi(x)^\top\phi(z)$ between mapped points, never $\phi(x)$ itself. A
**kernel function** $k(x,z)$ computes that dot product directly, in the
*original* space — skipping the expensive (or infinite-dimensional!)
mapping entirely:

$$
k(x,z) = \phi(x)^\top \phi(z)
$$

## Verifying a kernel matches its feature map

For $\phi(x)=(x_1^2,\sqrt2\,x_1x_2,x_2^2)$:

$$
\phi(x)^\top\phi(z) = x_1^2z_1^2 + 2x_1x_2z_1z_2 + x_2^2z_2^2 = (x_1z_1+x_2z_2)^2 = (x^\top z)^2
$$

So the **polynomial kernel** $k(x,z)=(x^\top z)^2$ computes this 3D dot
product using only a 2D input dot product, squared — no explicit
3-dimensional vectors ever formed.

## The RBF kernel: an infinite-dimensional feature map

$$
k(x,z) = \exp\left(-\frac{\|x-z\|^2}{2\sigma^2}\right)
$$

This corresponds to an *infinite-dimensional* $\phi$ (via a Taylor
expansion of the exponential) — a feature space you could never construct
explicitly, but whose dot products the kernel computes in closed form for
any finite $\sigma$.

## Mercer's condition

Not every function $k(x,z)$ is a valid kernel. It's valid iff the
**Gram matrix** $K_{ij}=k(x_i,x_j)$ is positive semi-definite for every
finite set of points $\{x_i\}$ — this guarantees a valid $\phi$ exists
(possibly infinite-dimensional) with $k(x,z)=\phi(x)^\top\phi(z)$.

## Worked numeric example

$x=(1,2), z=(3,1)$. Direct feature-map dot product:

$$
\phi(x) = (1, 2\sqrt2, 4), \quad \phi(z) = (9, 3\sqrt2, 1)
$$

$$
\phi(x)^\top\phi(z) = 1(9) + 2\sqrt2(3\sqrt2) + 4(1) = 9+12+4 = 25
$$

Kernel trick: $(x^\top z)^2 = (1\cdot3+2\cdot1)^2 = 5^2 = 25$ ✓ — matches
exactly, without ever forming $\phi$.

## Numeric verification

```python
import numpy as np

x = np.array([1.0, 2.0])
z = np.array([3.0, 1.0])

def phi(v):
    return np.array([v[0]**2, np.sqrt(2)*v[0]*v[1], v[1]**2])

feature_dot = phi(x) @ phi(z)
kernel_direct = (x @ z) ** 2
print(f"feature-space dot product = {feature_dot:.4f}")
print(f"polynomial kernel (x^T z)^2 = {kernel_direct:.4f}")

# Verify Gram matrix of RBF kernel is PSD for a small point set
pts = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]])
sigma = 1.0
n = len(pts)
K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        K[i,j] = np.exp(-np.sum((pts[i]-pts[j])**2) / (2*sigma**2))

eigvals = np.linalg.eigvalsh(K)
print(f"RBF Gram matrix eigenvalues = {eigvals}")
print(f"all >= 0 (valid kernel)? {np.all(eigvals >= -1e-10)}")
```

```text
feature-space dot product = 25.0000
polynomial kernel (x^T z)^2 = 25.0000
RBF Gram matrix eigenvalues = [0.0700 0.2079 0.2079 0.7261 3.7881]
all >= 0 (valid kernel)? True
```

## Exercise

1. Derive the explicit feature map for $k(x,z)=(x^\top z + 1)^2$ in 2D
   (this is the "inhomogeneous" polynomial kernel — it also generates
   lower-degree terms) and confirm the dot product matches numerically.
2. Prove that if $k_1$ and $k_2$ are valid kernels, so is $k_1+k_2$
   (using the Gram-matrix PSD definition — sum of PSD matrices is PSD).
3. Explain, using the RBF Gram matrix eigenvalues above, what a small
   $\sigma$ (very local kernel) versus a large $\sigma$ (very smooth
   kernel) does to the conditioning of $K$, and why that matters for
   kernel ridge regression's $(K+\lambda I)^{-1}$ solve.
