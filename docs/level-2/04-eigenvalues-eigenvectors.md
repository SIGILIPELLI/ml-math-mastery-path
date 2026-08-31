# 04 · Eigenvalues & Eigenvectors Intuition

Eigenvalues and eigenvectors describe the directions a matrix "prefers" —
directions it only stretches or shrinks, without rotating. This idea
underlies PCA, spectral clustering, stability analysis of gradient descent,
and understanding the Hessian from Module 3.

## Definition

For a square matrix $A$, a nonzero vector $\mathbf{v}$ is an
**eigenvector** with **eigenvalue** $\lambda$ if

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

Applying $A$ to $\mathbf{v}$ doesn't change its direction — only scales it
by $\lambda$.

## Finding eigenvalues

Rearranging: $(A - \lambda I)\mathbf{v} = \mathbf{0}$. For a nonzero
$\mathbf{v}$ to exist, $A - \lambda I$ must be singular, i.e.

$$
\det(A - \lambda I) = 0
$$

This is the **characteristic equation**; its roots are the eigenvalues.

## Worked example

$$
A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}
$$

**Characteristic equation:**

$$
\det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix} = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = 0
$$

Factoring: $(\lambda-5)(\lambda-2)=0$, so $\lambda_1 = 5$, $\lambda_2 = 2$.

**Eigenvector for $\lambda_1=5$:** solve $(A-5I)\mathbf{v}=0$:

$$
\begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}\mathbf{v} = 0 \implies v_1 = v_2
$$

so $\mathbf{v}_1 = \begin{bmatrix}1\\1\end{bmatrix}$ (up to scaling).

**Eigenvector for $\lambda_2=2$:** solve $(A-2I)\mathbf{v}=0$:

$$
\begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}\mathbf{v} = 0 \implies 2v_1 = -v_2
$$

so $\mathbf{v}_2 = \begin{bmatrix}1\\-2\end{bmatrix}$.

**Check:** $A\mathbf{v}_1 = \begin{bmatrix}4+1\\2+3\end{bmatrix}=\begin{bmatrix}5\\5\end{bmatrix}=5\mathbf{v}_1$. ✓

## Why this matters for ML

* **PCA** (Level 2 Module 5 preview): the eigenvectors of a data covariance
  matrix are the principal directions of variance; eigenvalues give the
  variance explained by each.
* **Hessian eigenvalues** classify critical points: all positive →
  local min, all negative → local max, mixed signs → saddle point.
* **Symmetric matrices** (like covariance and Hessian matrices) always have
  real eigenvalues and orthogonal eigenvectors — a fact used throughout ML.

## Numeric verification

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigvals, eigvecs = np.linalg.eig(A)
print("eigenvalues:", eigvals)
print("eigenvectors (columns):\n", eigvecs)

# Verify A v = lambda v for each column
for i in range(2):
    v = eigvecs[:, i]
    lam = eigvals[i]
    print(f"A v = {A @ v}, lambda*v = {lam * v}")
```

Expected output (NumPy normalizes eigenvectors to unit length, so signs/scale
may differ from the hand solution but the direction ratio matches: the
$\lambda=5$ eigenvector is proportional to $[1,1]$, and $\lambda=2$ to
$[1,-2]$):

```text
eigenvalues: [5. 2.]
eigenvectors (columns):
 [[ 0.70710678 -0.4472136 ]
 [ 0.70710678  0.89442719]]
A v = [3.53553391 3.53553391], lambda*v = [3.53553391 3.53553391]
A v = [-0.89442719 -1.78885438], lambda*v = [-0.89442719 -1.78885438]
```

## Exercise

$$
B = \begin{bmatrix} 2 & 0 \\ 0 & 5 \end{bmatrix}
$$

1. Since $B$ is diagonal, state its eigenvalues and eigenvectors by
   inspection — explain why diagonal matrices always work this way.
2. For $C = \begin{bmatrix}1 & 2\\2 & 1\end{bmatrix}$, find both eigenvalues
   via the characteristic equation.
3. Find the corresponding eigenvectors by hand.
4. Verify both eigenpairs with `np.linalg.eig` and confirm $Av=\lambda v$
   numerically.
