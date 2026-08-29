# 04 · Matrix Operations

With vector operations settled, matrices need their own rules —
particularly matrix multiplication, which is the single most-executed
operation in all of machine learning.

## Matrix addition and scalar multiplication

Same-shape matrices add entrywise, just like vectors; a scalar multiplies
every entry. Nothing new here beyond Module 3's rules applied entry-by-entry.

## Transpose

The **transpose** $\mathbf{A}^T$ flips a matrix over its diagonal: row $i$,
column $j$ of $\mathbf{A}$ becomes row $j$, column $i$ of $\mathbf{A}^T$. If

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\quad (2\times 3)
$$

then

$$
\mathbf{A}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
\quad (3\times 2)
$$

Transpose is used constantly to make shapes compatible for multiplication,
and $\mathbf{A}^T\mathbf{A}$ appears in the closed-form solution to linear
regression (Module 9).

## Matrix multiplication

To multiply $\mathbf{A}$ ($m\times n$) by $\mathbf{B}$ ($n \times p$), the
**inner dimensions must match** ($n = n$), and the result is $m \times p$.
Entry $(i,j)$ of the product is the dot product of row $i$ of $\mathbf{A}$
with column $j$ of $\mathbf{B}$:

$$
(\mathbf{AB})_{ij} = \sum_{k=1}^n a_{ik}b_{kj}
$$

**This is why the dot product from Module 3 matters so much**: matrix
multiplication is literally "a grid of dot products."

### Worked example

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\quad
\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

Both are $2\times2$, so $\mathbf{AB}$ is $2\times2$:

- $(\mathbf{AB})_{11} = (1)(5)+(2)(7) = 5+14 = 19$
- $(\mathbf{AB})_{12} = (1)(6)+(2)(8) = 6+16 = 22$
- $(\mathbf{AB})_{21} = (3)(5)+(4)(7) = 15+28 = 43$
- $(\mathbf{AB})_{22} = (3)(6)+(4)(8) = 18+32 = 50$

$$
\mathbf{AB} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

**Important:** matrix multiplication is *not commutative* in general —
$\mathbf{AB} \neq \mathbf{BA}$ (check it yourself in the exercise).

### Matrix-vector multiplication: the linear model connection

A very common special case is $\mathbf{A}$ ($m\times n$) times a vector
$\mathbf{x}$ ($n\times 1$), giving an $m\times1$ vector. For a linear model
over $d$ features with $n$ training examples stacked as rows of
$\mathbf{X}$ ($n\times d$) and a weight vector $\mathbf{w}$ ($d\times1$),

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}
$$

computes **all $n$ predictions at once** — row $i$ of the result is exactly
$\mathbf{x}^{(i)}\cdot\mathbf{w}$, the dot-product prediction from Module 3,
for every example simultaneously. This is why ML libraries are fast: one
matrix multiplication replaces a loop over every training example.

## The identity matrix

$\mathbf{I}$ has 1s on the diagonal and 0s elsewhere. For any compatible
matrix $\mathbf{A}$: $\mathbf{IA} = \mathbf{A}$ and $\mathbf{AI} = \mathbf{A}$
— it behaves like the number 1 for matrices. It appears in the closed-form
linear regression solution and in regularization terms (Level 3).

## Worked numeric example (full, with NumPy cross-check)

Using $\mathbf{A}$ and $\mathbf{B}$ from above:

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

AB = A @ B          # matrix multiplication ('@' or np.matmul, NOT '*')
BA = B @ A
AT = A.T
I  = np.eye(2)

print("A @ B =\n", AB)
print("B @ A =\n", BA)
print("A.T   =\n", AT)
print("A @ I =\n", A @ I)
```

Expected output (matches the hand-computed $\mathbf{AB}$ above, and shows
$\mathbf{AB}\ne\mathbf{BA}$):

```text
A @ B =
 [[19 22]
 [43 50]]
B @ A =
 [[23 34]
 [31 46]]
A.T   =
 [[1 3]
 [2 4]]
A @ I =
 [[1. 2.]
 [3. 4.]]
```

Note the important pitfall: `A * B` in NumPy does **entrywise**
multiplication (also called the Hadamard product), not matrix
multiplication — always use `@` or `np.matmul` for the linear-algebra
product.

## Exercise

Given

$$
\mathbf{C} = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}
\quad
\mathbf{D} = \begin{bmatrix} 1 & 4 \\ 2 & 1 \end{bmatrix}
$$

1. Compute $\mathbf{CD}$ by hand using the row-dot-column rule.
2. Compute $\mathbf{DC}$ by hand and confirm it differs from $\mathbf{CD}$.
3. Compute $\mathbf{C}^T$ by hand.
4. Verify all three with NumPy (`C @ D`, `D @ C`, `C.T`), and also print
   `C * D` (entrywise) to see how it differs from `C @ D`.
