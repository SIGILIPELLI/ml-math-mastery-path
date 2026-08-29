# 03 · Vector Operations

Now that vectors have a shape, they need operations. Three matter most for
ML: addition/scaling, the **dot product**, and the **norm** (length).

## Addition and scalar multiplication

Vectors of the same dimension add entrywise:

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}
$$

A scalar multiplies every entry:

$$
c\,\mathbf{u} = \begin{bmatrix} c\,u_1 \\ c\,u_2 \end{bmatrix}
$$

This is exactly what happens in a gradient descent update
$\theta \leftarrow \theta - \alpha \nabla J$: $\alpha \nabla J$ is a scalar
times a vector, then subtracted (vector addition with a negated vector).

## The dot product

The **dot product** of two vectors of the same dimension multiplies
corresponding entries and sums the results, producing a single **scalar**:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i
$$

For $\mathbf{u} = \begin{bmatrix}1\\2\\3\end{bmatrix}$ and
$\mathbf{v} = \begin{bmatrix}4\\5\\6\end{bmatrix}$:

$$
\mathbf{u}\cdot\mathbf{v} = 1(4) + 2(5) + 3(6) = 4 + 10 + 18 = 32
$$

**Where this shows up in ML:** a linear model's prediction
$\hat{y} = w_1x_1 + w_2x_2 + \dots + w_dx_d$ is *exactly* the dot product
$\mathbf{w}\cdot\mathbf{x}$. Every linear layer of a neural network is a
stack of dot products.

## Geometric meaning of the dot product

The dot product also equals

$$
\mathbf{u}\cdot\mathbf{v} = \lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert \cos\theta
$$

where $\theta$ is the angle between the two vectors. This tells us:

- If $\mathbf{u}\cdot\mathbf{v} > 0$, the vectors point in a broadly similar
  direction ($\theta < 90°$).
- If $\mathbf{u}\cdot\mathbf{v} = 0$, the vectors are **orthogonal**
  (perpendicular, $\theta = 90°$) — an idea that reappears constantly (e.g.
  orthogonal weight initialization, PCA's orthogonal components).
- If $\mathbf{u}\cdot\mathbf{v} < 0$, they point in broadly opposite
  directions.

This is also the basis of **cosine similarity**, used to compare embedding
vectors in NLP/recommendation systems:

$$
\cos\theta = \frac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert}
$$

## The norm (length)

The **Euclidean norm** (length) of a vector is

$$
\lVert \mathbf{v} \rVert = \sqrt{\mathbf{v}\cdot\mathbf{v}} = \sqrt{\sum_{i=1}^n v_i^2}
$$

For $\mathbf{v} = \begin{bmatrix}3\\4\end{bmatrix}$:
$\lVert\mathbf{v}\rVert = \sqrt{3^2+4^2} = \sqrt{9+16} = \sqrt{25} = 5$ — the
familiar 3-4-5 right triangle. Norms show up as the "size" of an error
vector, the "size" of a weight vector (used in L2 regularization, Level 3),
and the denominator in cosine similarity above.

## Worked numeric example

Let $\mathbf{u} = \begin{bmatrix}1\\2\\3\end{bmatrix}$ and
$\mathbf{v} = \begin{bmatrix}4\\5\\6\end{bmatrix}$ (from above).

- $\mathbf{u}\cdot\mathbf{v} = 32$ (computed above).
- $\lVert\mathbf{u}\rVert = \sqrt{1^2+2^2+3^2} = \sqrt{14} \approx 3.742$.
- $\lVert\mathbf{v}\rVert = \sqrt{4^2+5^2+6^2} = \sqrt{77} \approx 8.775$.
- $\cos\theta = \dfrac{32}{\sqrt{14}\sqrt{77}} = \dfrac{32}{\sqrt{1078}} \approx \dfrac{32}{32.833} \approx 0.9746$.

So the angle between $\mathbf{u}$ and $\mathbf{v}$ is
$\theta = \arccos(0.9746) \approx 12.9°$ — a small angle, meaning the two
vectors point in a very similar direction, which makes sense since both have
steadily increasing, proportionally similar entries.

```python
import numpy as np

u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])

dot = np.dot(u, v)
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)
cos_theta = dot / (norm_u * norm_v)
theta_deg = np.degrees(np.arccos(cos_theta))

print("u . v      =", dot)
print("||u||      =", round(norm_u, 3))
print("||v||      =", round(norm_v, 3))
print("cos(theta) =", round(cos_theta, 4))
print("theta (deg)=", round(theta_deg, 1))
```

Expected output (matches the hand computation above):

```text
u . v      = 32.0
||u||      = 3.742
||v||      = 8.775
cos(theta) = 0.9746
theta (deg)= 12.9
```

## Exercise

Let $\mathbf{a} = \begin{bmatrix}2\\0\end{bmatrix}$ and
$\mathbf{b} = \begin{bmatrix}0\\3\end{bmatrix}$.

1. Compute $\mathbf{a}\cdot\mathbf{b}$ by hand. What does the result tell you
   about the angle between them, geometrically?
2. Compute $\lVert\mathbf{a}\rVert$ and $\lVert\mathbf{b}\rVert$ by hand.
3. Use the dot-product formula to compute $\cos\theta$ and confirm
   $\theta = 90°$.
4. Verify all three answers with `np.dot`, `np.linalg.norm`, and
   `np.arccos`.
