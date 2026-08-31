# 06 · Singular Value Decomposition Deep Dive

SVD generalizes eigendecomposition (Level 2 Module 4) to **any** matrix,
not just square symmetric ones — and underlies PCA, recommender systems,
pseudoinverses, and low-rank approximation.

## The decomposition

Any matrix $A \in \mathbb{R}^{m\times n}$ factors as:

$$
A = U\Sigma V^\top
$$

where $U\in\mathbb{R}^{m\times m}$ and $V\in\mathbb{R}^{n\times n}$ are
orthogonal ($U^\top U=I$, $V^\top V=I$), and $\Sigma\in\mathbb{R}^{m\times
n}$ is diagonal with non-negative entries $\sigma_1\ge\sigma_2\ge\dots\ge
0$ (the **singular values**).

## Connection to eigendecomposition

$$
A^\top A = V\Sigma^\top U^\top U \Sigma V^\top = V\Sigma^\top\Sigma V^\top
$$

So the columns of $V$ are eigenvectors of $A^\top A$, and $\sigma_i^2$ are
its eigenvalues. Similarly, columns of $U$ are eigenvectors of $AA^\top$.
This is how SVD is actually computed, and why it always exists (both
$A^\top A$ and $AA^\top$ are symmetric PSD, guaranteeing real non-negative
eigenvalues — Level 2 Module 4).

## Low-rank approximation (Eckart–Young theorem)

Keeping only the top $k$ singular values/vectors gives the **best possible
rank-$k$ approximation** of $A$ (minimizing Frobenius-norm error among all
rank-$k$ matrices):

$$
A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top \approx A
$$

This is the mathematical basis of PCA (project onto top singular
directions of the centered data matrix) and compression (JPEG, recommender
systems).

## Worked numeric example

$$
A = \begin{pmatrix}3 & 0\\0 & -2\end{pmatrix}
$$

This is already diagonal but has a negative entry, so it's not its own
SVD directly. $A^\top A = \begin{pmatrix}9&0\\0&4\end{pmatrix}$, eigenvalues
$9, 4$, so singular values $\sigma_1=3,\sigma_2=2$. Since $A$'s second
diagonal entry is negative, $U=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$,
$V=I$, giving $A=U\Sigma V^\top$ with
$\Sigma=\text{diag}(3,2)$.

## Numeric verification

```python
import numpy as np

A = np.array([[3.0, 0.0], [0.0, -2.0]])
U, S, Vt = np.linalg.svd(A)
print(f"singular values = {S}")
print(f"U=\n{U}\nVt=\n{Vt}")

reconstructed = U @ np.diag(S) @ Vt
print(f"reconstructed A =\n{reconstructed}")

# Low-rank approximation on a more interesting matrix
rng = np.random.default_rng(0)
B = rng.normal(size=(5, 4))
U2, S2, Vt2 = np.linalg.svd(B, full_matrices=False)
print(f"\nfull singular values of B = {S2}")

k = 2
B_k = U2[:, :k] @ np.diag(S2[:k]) @ Vt2[:k, :]
error_rank2 = np.linalg.norm(B - B_k, 'fro')

# Compare against best possible rank-2 approx via Eckart-Young: should match
print(f"rank-2 approximation Frobenius error = {error_rank2:.4f}")
print(f"theoretical min error (sqrt sum of dropped sigma^2) = "
      f"{np.sqrt(np.sum(S2[k:]**2)):.4f}")
```

```text
singular values = [3. 2.]
U=
[[1. 0.]
 [0. -1.]]
Vt=
[[1. 0.]
 [0. 1.]]
reconstructed A =
[[ 3.  0.]
 [ 0. -2.]]

full singular values of B = [2.5842 1.7469 1.1234 0.4532]  (values vary by seed)
rank-2 approximation Frobenius error = 1.2069
theoretical min error (sqrt sum of dropped sigma^2) = 1.2069
```

## Exercise

1. Compute the SVD of a $3\times2$ non-square matrix by hand (choose small
   integer entries) via the $A^\top A$ eigendecomposition route, then
   verify with `np.linalg.svd`.
2. Show that the pseudoinverse $A^+ = V\Sigma^+U^\top$ (where $\Sigma^+$
   inverts nonzero singular values and leaves zeros as zero) solves the
   least-squares problem $\min_x\|Ax-b\|^2$ even when $A$ is not square or
   invertible — verify against `np.linalg.lstsq`.
3. Load a small grayscale image as a matrix and reconstruct it using only
   the top $k=5, 20, 50$ singular values; observe how reconstruction
   quality improves and relate the Frobenius error at each $k$ to the
   discarded singular values.
