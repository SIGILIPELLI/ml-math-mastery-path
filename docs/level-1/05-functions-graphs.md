# 05 · Functions & Graphs Refresher

Before derivatives, a quick refresher on functions and their graphs — the
vocabulary of "slope," "curve," and "rate of change" all rest on this.

## What a function is

A function $f$ takes an input and produces exactly one output:
$f: x \mapsto f(x)$. In ML, the function of central interest is the **cost
function** $J(\theta)$, which takes model parameters $\theta$ as input and
outputs a single number measuring how wrong the model currently is. Training
a model is nothing more than *finding the input $\theta$ that minimizes the
output of $J$*.

## Linear functions

$f(x) = mx + c$ is a straight line: $m$ is the **slope** (rise over run —
how much $f$ changes per unit change in $x$), and $c$ is the
**y-intercept** (value of $f$ at $x=0$). Slope is constant everywhere on a
line — this is the simplest possible "rate of change," and it's exactly what
a derivative generalizes to curves.

$$
\text{slope} = \frac{f(x_2) - f(x_1)}{x_2 - x_1}
$$

## Quadratic functions

$f(x) = ax^2 + bx + c$ is a parabola. If $a > 0$ it opens upward and has a
single minimum; if $a < 0$ it opens downward with a single maximum. Cost
functions for linear regression are quadratic in the parameters (a
"bowl" shape) — which is exactly why they have one unique minimum that
gradient descent can reliably find (more in Level 4's convexity module).

## Composite functions

A **composite function** plugs one function into another: $h(x) = f(g(x))$.
For example, if $g(x) = x^2$ and $f(u) = \sqrt{u}$, then
$h(x) = f(g(x)) = \sqrt{x^2}$. Composite functions are everywhere in ML: a
neural network is a long chain of compositions (linear transform, then
activation, then another linear transform, ...), and the **chain rule**
(Module 7, and deep-dived in Level 2) is precisely the tool for
differentiating them.

## Rate of change: average vs. instantaneous

The **average rate of change** of $f$ between $x_1$ and $x_2$ is the slope
of the straight line connecting the two points on the graph (a "secant
line"):

$$
\text{avg. rate of change} = \frac{f(x_2)-f(x_1)}{x_2-x_1}
$$

For a curve, this average depends on how far apart $x_1$ and $x_2$ are. The
**instantaneous rate of change** at a single point is what you get by
shrinking that gap to zero — which is exactly the definition of a derivative
we build in Module 6.

## Worked numeric example

Let $f(x) = x^2$. Compute the average rate of change between $x_1 = 1$ and
$x_2 = 3$:

$$
\frac{f(3)-f(1)}{3-1} = \frac{9-1}{2} = 4
$$

Now shrink the interval to $x_1=1, x_2=1.1$:

$$
\frac{f(1.1)-f(1)}{1.1-1} = \frac{1.21-1}{0.1} = \frac{0.21}{0.1} = 2.1
$$

And even tighter, $x_1=1, x_2=1.01$:

$$
\frac{f(1.01)-f(1)}{1.01-1} = \frac{1.0201-1}{0.01} = \frac{0.0201}{0.01} = 2.01
$$

Notice the average rate of change is homing in on $2$ as the interval
shrinks — a preview of the exact result Module 6 proves: the derivative of
$x^2$ at $x=1$ is exactly $2$.

```python
import numpy as np

def f(x):
    return x ** 2

x1 = 1.0
for x2 in [3.0, 1.1, 1.01, 1.001]:
    avg_rate = (f(x2) - f(x1)) / (x2 - x1)
    print(f"secant slope from x=1 to x={x2}: {avg_rate:.4f}")
```

Expected output (matches the hand computation above, converging toward 2):

```text
secant slope from x=1 to x=3.0: 4.0000
secant slope from x=1 to x=1.1: 2.1000
secant slope from x=1 to x=1.01: 2.0100
secant slope from x=1 to x=1.001: 2.0010
```

## Exercise

Let $f(x) = x^2 + 1$.

1. Compute the average rate of change between $x=2$ and $x=4$ by hand.
2. Compute the average rate of change between $x=2$ and $x=2.1$ by hand.
3. Compute it again between $x=2$ and $x=2.001$, and note what value it
   seems to be approaching.
4. Write a small Python loop (like the one above) that computes the secant
   slope for a shrinking sequence of gaps and confirm your hand-computed
   trend.
