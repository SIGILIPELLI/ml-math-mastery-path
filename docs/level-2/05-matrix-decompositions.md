# 05 · Matrix Decompositions Overview

Building on eigenvalues, this module introduces the two decompositions that
show up everywhere in ML: **eigendecomposition** and the more general
**Singular Value Decomposition (SVD)** — the engine behind PCA,
recommender-system factorization, and pseudo-inverses for least squares.

## Eigendecomposition

If a square matrix $A$ ($n\times n$) has $n$ linearly independent
eigenvectors, it can be written as

$$
A = V \Lambda V^{-1}
$$

where $V$'s columns are the eigenvectors and $\Lambda$ is diagonal with the
eigenvalues on the diagonal. When $A$ is symmetric (e.g. a covariance
matrix), $V$ can be chosen orthogonal, so $V^{-1} = V^\top$:

$$
A = V \Lambda V^\top \qquad (A \text{ symmetric})
$$

## Singular Value Decomposition (SVD)

Eigendecomposition requires a square matrix. **Any** matrix $A$ ($m\times n$,
even non-square) can be factored as

$$
A = U \Sigma V^\top
$$

where $U$ ($m\times m$) and $V$ ($n\times n$) are orthogonal, and $\Sigma$
($m\times n$) is diagonal with non-negative **singular values**
$\sigma_1 \geq \sigma_2 \geq \dots \geq 0$ on the diagonal. Singular values
relate to eigenvalues of $A^\top A$: $\sigma_i = \sqrt{\lambda_i(A^\top A)}$.

## Worked example: eigendecomposition

Reuse $A = \begin{bmatrix}4&1\\2&3\end{bmatrix}$ from Module 4, with
$\lambda_1=5, \mathbf{v}_1=[1,1]$ and $\lambda_2=2, \mathbf{v}_2=[1,-2]$.

$$
V = \begin{bmatrix}1&1\\1&-2\end{bmatrix} \qquad \Lambda = \begin{bmatrix}5&0\\0&2\end{bmatrix}
$$

$$
V^{-1} = \frac{1}{\det V}\begin{bmatrix}-2&-1\\-1&1\end{bmatrix}, \quad \det V = 1(-2)-1(1) = -3
$$

$$
V^{-1} = \begin{bmatrix}2/3 & 1/3\\1/3 & -1/3\end{bmatrix}
$$

Reconstructing $A = V\Lambda V^{-1}$ should give back the original matrix
(verified numerically below).

## Worked example: SVD by hand (small case)

For $A = \begin{bmatrix}3&0\\0&-2\end{bmatrix}$ (already diagonal),
$A^\top A = \begin{bmatrix}9&0\\0&4\end{bmatrix}$, eigenvalues $9,4$, so
singular values $\sigma_1=3, \sigma_2=2$. Since $A$ is diagonal with signs,
$U = \begin{bmatrix}1&0\\0&-1\end{bmatrix}$, $\Sigma=\begin{bmatrix}3&0\\0&2\end{bmatrix}$,
$V = I$, and indeed $U\Sigma V^\top = \begin{bmatrix}3&0\\0&-2\end{bmatrix}=A$.

## Connection to PCA

Given a mean-centered data matrix $X$ ($n$ samples $\times$ $p$ features),
the covariance matrix $C = \frac{1}{n-1}X^\top X$ is symmetric. Its
eigendecomposition $C = V\Lambda V^\top$ gives the principal components in
$V$ (directions of max variance) and the variance explained by each in
$\Lambda$. Equivalently, the SVD of $X$ itself gives the same directions
directly, which is why most PCA implementations use SVD rather than
eigendecomposition of $C$ (better numerical stability).

## Numeric verification

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigvals, V = np.linalg.eig(A)
Lambda = np.diag(eigvals)
A_reconstructed = V @ Lambda @ np.linalg.inv(V)
print("A reconstructed from eigendecomposition:\n", np.round(A_reconstructed, 6))

B = np.array([[3, 0],
              [0, -2]], dtype=float)
U, S, Vt = np.linalg.svd(B)
print("singular values:", S)
print("B reconstructed from SVD:\n", np.round(U @ np.diag(S) @ Vt, 6))
```

Expected output:

```text
A reconstructed from eigendecomposition:
 [[4. 1.]
 [2. 3.]]
singular values: [3. 2.]
B reconstructed from SVD:
 [[ 3.  0.]
 [ 0. -2.]]
```

## Exercise

Let $D = \begin{bmatrix}2&0\\0&2\end{bmatrix}$ (a scaled identity) and
$E = \begin{bmatrix}0&1\\1&0\end{bmatrix}$ (a swap matrix).

1. Find the eigenvalues/eigenvectors of $D$ and $E$ by hand.
2. Confirm both are symmetric and check that their eigenvectors are
   orthogonal (dot product zero).
3. Compute the SVD of $F = \begin{bmatrix}1&1\\0&1\end{bmatrix}$ using
   `np.linalg.svd` and verify $U\Sigma V^\top$ reconstructs $F$.
4. Explain in one sentence why SVD works even for non-square matrices while
   eigendecomposition does not.
