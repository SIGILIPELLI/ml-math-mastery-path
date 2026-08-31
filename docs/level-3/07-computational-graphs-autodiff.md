# 07 · Computational Graphs & Autodiff

Frameworks like PyTorch and TensorFlow don't use symbolic differentiation
or numerical finite differences — they use **automatic differentiation
(autodiff)**, which computes exact derivatives by tracking a computational
graph of primitive operations and applying the chain rule mechanically.

## The computational graph

Any expression decomposes into a DAG (directed acyclic graph) of primitive
operations. For $f(x,y) = (x+y) \cdot (x \cdot y)$, introduce intermediate
nodes:

$$
a = x + y \qquad b = x \cdot y \qquad f = a \cdot b
$$

## Forward mode vs. reverse mode

**Forward mode** propagates derivatives *with* the computation, tracking
$\dot v = \partial v/\partial x$ for a chosen input $x$ alongside each node
value $v$. Cheap when there are few inputs, many outputs.

**Reverse mode** (what backprop is) does a forward pass to compute all node
values, then a backward pass computing $\bar v = \partial L/\partial v$
starting from the output and working back via the chain rule. Cheap when
there are many inputs (e.g. millions of weights) and one scalar output
(the loss) — exactly the neural network case, computed in one backward pass
regardless of parameter count.

## Reverse-mode by hand

For $a=x+y,\ b=xy,\ f=ab$, with $x=2, y=3$:

**Forward:** $a = 5,\ b = 6,\ f = 30$

**Backward**, seeding $\bar f = \partial f/\partial f = 1$:

$$
\bar a = \bar f \cdot \frac{\partial f}{\partial a} = 1\cdot b = 6 \qquad
\bar b = \bar f \cdot \frac{\partial f}{\partial b} = 1\cdot a = 5
$$

Each intermediate's gradient is now propagated to *its* inputs:

$$
\bar x \mathrel{+}= \bar a \cdot \frac{\partial a}{\partial x} = 6\cdot 1 = 6
\qquad
\bar x \mathrel{+}= \bar b \cdot \frac{\partial b}{\partial x} = 5\cdot y = 15
$$

$$
\bar y \mathrel{+}= \bar a \cdot 1 = 6 \qquad \bar y \mathrel{+}= \bar b \cdot x = 10
$$

So $\partial f/\partial x = 6+15=21$, $\partial f/\partial y = 6+10=16$.
Note $x$ contributes through **two paths** ($a$ and $b$) — autodiff sums
gradients at every node with multiple consumers, the multivariable chain
rule's sum rule.

## Verify symbolically

$f = (x+y)(xy) = x^2y + xy^2$, so $\partial f/\partial x = 2xy+y^2 =
2(2)(3)+9=21$ ✓, $\partial f/\partial y = x^2+2xy = 4+12=16$ ✓.

## Numeric verification

```python
import numpy as np

x, y = 2.0, 3.0

# Forward pass
a = x + y
b = x * y
f = a * b

# Reverse pass (manual autodiff)
f_bar = 1.0
a_bar = f_bar * b
b_bar = f_bar * a

x_bar = a_bar * 1 + b_bar * y   # two paths into x
y_bar = a_bar * 1 + b_bar * x   # two paths into y

print(f"f = {f}")
print(f"df/dx (autodiff) = {x_bar}, df/dy (autodiff) = {y_bar}")

# Finite-difference check
h = 1e-6
def F(x, y): return (x + y) * (x * y)
dfdx_num = (F(x+h, y) - F(x-h, y)) / (2*h)
dfdy_num = (F(x, y+h) - F(x, y-h)) / (2*h)
print(f"df/dx (numeric) = {dfdx_num:.4f}, df/dy (numeric) = {dfdy_num:.4f}")
```

```text
f = 30.0
df/dx (autodiff) = 21.0, df/dy (autodiff) = 16.0
df/dx (numeric) = 21.0000, df/dy (numeric) = 16.0000
```

## Exercise

1. Build the computational graph for $g(x) = \sin(x^2) + x$ and compute
   $g'(x)$ at $x=1$ via manual reverse-mode autodiff (define nodes
   $u=x^2,\ v=\sin(u),\ g=v+x$).
2. Verify your answer against the symbolic derivative
   $g'(x) = 2x\cos(x^2)+1$ and a finite-difference check.
3. Explain why reverse mode needs to store all intermediate node values
   from the forward pass (this is the "memory cost of backprop" that
   motivates techniques like gradient checkpointing).
