# 09 · Math Behind Transformer Attention

Transformer attention is a weighted average — the "weighting" comes from
scaled dot products passed through softmax (Level 3 Module 06), and the
"scaling" is a numerical-stability fix (Level 3 Module 09) that's easy to
forget but essential.

## Scaled dot-product attention

Given queries $Q\in\mathbb{R}^{n\times d_k}$, keys $K\in\mathbb{R}^{m\times
d_k}$, and values $V\in\mathbb{R}^{m\times d_v}$:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

## Reading it piece by piece

$QK^\top$ is an $n\times m$ matrix of raw similarity scores: row $i$,
column $j$ is $q_i \cdot k_j$ — how much query $i$ "attends to" key $j$.
Softmax (row-wise) turns each row into a probability distribution over
the $m$ keys. Multiplying by $V$ produces, for each query, a weighted
average of the value vectors — weighted by how relevant each key was.

## Why divide by $\sqrt{d_k}$

If $q,k$ have independent components with mean 0, variance 1, then
$q\cdot k = \sum_{i=1}^{d_k} q_ik_i$ has variance $d_k$ (sum of $d_k$
independent unit-variance terms — Level 2 Module 7's variance-of-a-sum
rule). For large $d_k$ (e.g. 64 or 128), raw dot products can be huge,
pushing softmax into a saturated regime where gradients vanish (extremely
peaked distributions, Level 3 Module 09's numerical stability territory).
Dividing by $\sqrt{d_k}$ rescales the variance back to $\approx 1$
regardless of dimension, keeping softmax's input in a well-behaved range.

## Worked numeric example

$d_k=4$. One query $q=(1,0,1,0)$, two keys $k_1=(1,1,0,0),\
k_2=(0,0,1,1)$, values $v_1=(1,0),\ v_2=(0,1)$.

$$
q\cdot k_1 = 1, \qquad q\cdot k_2 = 1
$$

Scaled: $1/\sqrt4 = 0.5$ for both. Softmax of $(0.5, 0.5)$ is $(0.5, 0.5)$
(a tie, since both scores are equal). Output: $0.5v_1 + 0.5v_2 = (0.5,
0.5)$ — an even blend, correctly reflecting that $q$ is equally similar to
both keys.

## Numeric verification

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

Q = np.array([[1.0, 0.0, 1.0, 0.0]])          # 1 query, d_k=4
K = np.array([[1.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 1.0]])           # 2 keys
V = np.array([[1.0, 0.0],
              [0.0, 1.0]])                     # 2 values, d_v=2

d_k = Q.shape[-1]
scores = Q @ K.T / np.sqrt(d_k)
weights = softmax(scores)
output = weights @ V

print(f"raw scores (unscaled) = {Q @ K.T}")
print(f"scaled scores = {scores}")
print(f"attention weights = {weights}")
print(f"output = {output}")

# Demonstrate the variance-blowup problem the scaling fixes
rng = np.random.default_rng(0)
for d_k_test in [4, 64, 512]:
    q = rng.normal(size=d_k_test)
    k = rng.normal(size=d_k_test)
    raw_dot = q @ k
    scaled_dot = raw_dot / np.sqrt(d_k_test)
    print(f"d_k={d_k_test}: raw dot~std({np.sqrt(d_k_test):.1f}) -> "
          f"{raw_dot:.2f}, scaled -> {scaled_dot:.2f}")
```

```text
raw scores (unscaled) = [[1. 1.]]
scaled scores = [[0.5 0.5]]
attention weights = [[0.5 0.5]]
output = [[0.5 0.5]]
d_k=4: raw dot~std(2.0) -> 1.68, scaled -> 0.84
d_k=64: raw dot~std(8.0) -> -0.24, scaled -> -0.03
d_k=512: raw dot~std(22.6) -> -30.85, scaled -> -1.36
```

## Exercise

1. Add a third key $k_3=(1,0,1,0)$ (identical to $q$) with value
   $v_3=(1,1)$ — recompute attention weights and output, and confirm the
   identical key dominates the softmax distribution.
2. Implement multi-head attention conceptually: split $Q,K,V$ into $h$
   chunks along the feature dimension, run scaled dot-product attention on
   each chunk independently, and concatenate the outputs. Explain why each
   head effectively uses a smaller $d_k$, changing the right scaling factor.
3. Derive $\partial(\text{Attention output})/\partial Q$ conceptually by
   naming which upstream modules it decomposes into (softmax Jacobian from
   Module 06 of Level 3, matrix multiplication gradients) — you don't need
   to write the full tensor expression, just identify the chain.
