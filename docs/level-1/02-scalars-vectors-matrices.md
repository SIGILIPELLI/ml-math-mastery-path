# 02 · Scalars, Vectors & Matrices

Every object in ML math is one of a small number of shapes. Getting the
vocabulary and notation solid now saves confusion in every later module.

## Scalars

A **scalar** is a single number: $x = 5$, $\alpha = 0.01$, $\pi$. Lowercase
italic letters (often Greek for special constants) denote scalars.

## Vectors

A **vector** is an ordered list of numbers, written as a column by
convention:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}
$$

Bold lowercase letters denote vectors. The number of entries is the vector's
**dimension**; $\mathbf{v} \in \mathbb{R}^3$ means $\mathbf{v}$ is a
3-dimensional real vector. In ML, a single data example (say, a house with 3
features: square footage, bedrooms, age) is naturally a vector:

$$
\mathbf{x} = \begin{bmatrix} 1500 \\ 3 \\ 12 \end{bmatrix}
$$

## Matrices

A **matrix** is a rectangular grid of numbers, bold uppercase:

$$
\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}
$$

This is a $3 \times 2$ matrix (3 rows, 2 columns) — **rows first, columns
second**, always. Entry $a_{ij}$ is the value in row $i$, column $j$. In ML,
an entire dataset of $n$ examples with $d$ features each is naturally an
$n \times d$ matrix: each **row** is one example (a vector), and each
**column** is one feature across all examples.

## Tensors, briefly

A **tensor** generalizes this idea to more axes — a stack of matrices (e.g.
a batch of images, each a matrix of pixels, stacked into a 3D or 4D block).
Level 1 sticks to scalars/vectors/matrices; tensors reappear once we discuss
neural networks in later levels.

## Special vectors and matrices worth knowing now

- **Zero vector** $\mathbf{0}$: every entry is 0.
- **Zero matrix**: every entry is 0.
- **Identity matrix** $\mathbf{I}$: 1s on the diagonal, 0 elsewhere — the
  matrix version of the number 1 (multiplying by it changes nothing). We
  derive this properly in Module 4.
- **Transpose** $\mathbf{A}^T$: flip rows and columns, so a $3\times2$
  matrix becomes $2\times3$. Also covered fully in Module 4.

## Worked numeric example

Take the small dataset of 3 houses, each with 2 features (square footage in
thousands, and number of bedrooms):

$$
\mathbf{X} = \begin{bmatrix} 1.5 & 3 \\ 2.0 & 4 \\ 1.2 & 2 \end{bmatrix}
$$

This is a $3 \times 2$ matrix: 3 rows (examples), 2 columns (features). By
hand:

- Entry $x_{21}$ (row 2, column 1) is $2.0$ — the square footage of house 2.
- Entry $x_{32}$ (row 3, column 2) is $2$ — the bedroom count of house 3.
- Row 1, as a vector, is $\mathbf{x}^{(1)} = \begin{bmatrix}1.5 \\ 3\end{bmatrix}$.
- Column 1, as a vector, is $\begin{bmatrix}1.5 \\ 2.0 \\ 1.2\end{bmatrix}$ — the square-footage feature across all houses.

```python
import numpy as np

X = np.array([
    [1.5, 3],
    [2.0, 4],
    [1.2, 2],
])

print("shape:", X.shape)          # (3, 2) -- 3 rows, 2 columns
print("x_21 (row2,col1):", X[1, 0])   # 0-indexed: row index 1, col index 0
print("x_32 (row3,col2):", X[2, 1])   # row index 2, col index 1
print("row 1 (house 1):", X[0, :])
print("column 1 (sqft feature):", X[:, 0])
```

Expected output (matches the hand-computed values above; NumPy is 0-indexed,
so "row 2" in math notation is `X[1, :]` in code):

```text
shape: (3, 2)
x_21 (row2,col1): 2.0
x_32 (row3,col2): 2.0
row 1 (house 1): [1.5 3. ]
column 1 (sqft feature): [1.5 2.  1.2]
```

## Exercise

Given the matrix

$$
\mathbf{B} = \begin{bmatrix} 4 & 0 & 1 \\ 2 & 5 & 3 \end{bmatrix}
$$

1. State its shape (rows $\times$ columns) using the row-first convention.
2. Write out row 2 as a vector, using math notation.
3. Write out column 3 as a vector, using math notation.
4. Confirm all three answers with NumPy by creating `B = np.array(...)` and
   indexing `B.shape`, `B[1, :]`, and `B[:, 2]`.
