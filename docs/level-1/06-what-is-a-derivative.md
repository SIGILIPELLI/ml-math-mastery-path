# 06 · What Is a Derivative?

Module 5 ended with the average rate of change of $f(x)=x^2$ homing in on
$2$ as the interval around $x=1$ shrank to zero. That limiting value **is**
the derivative — the instantaneous rate of change, or equivalently, the
slope of the line tangent to the curve at that exact point.

## The first-principles definition

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

This is precisely the secant-slope formula from Module 5
($\frac{f(x_2)-f(x_1)}{x_2-x_1}$) with $x_2 = x+h$, $x_1 = x$, and $h$
shrinking toward (but never reaching) zero. $f'(x)$ (also written
$\frac{df}{dx}$) is itself a function of $x$: it tells you the slope of $f$
at every point.

## Deriving the derivative of $x^2$ from scratch

Let's prove, algebraically, that the derivative of $f(x)=x^2$ is $f'(x)=2x$
— the value Module 5's numeric experiment was converging toward.

$$
f'(x) = \lim_{h\to0} \frac{(x+h)^2 - x^2}{h}
$$

Expand $(x+h)^2 = x^2 + 2xh + h^2$:

$$
= \lim_{h\to0} \frac{x^2 + 2xh + h^2 - x^2}{h} = \lim_{h\to0} \frac{2xh + h^2}{h}
$$

Factor $h$ out of the numerator and cancel (valid since $h\neq 0$ throughout
the limit, only approaching zero):

$$
= \lim_{h\to0} (2x + h) = 2x
$$

So $f'(x) = 2x$. At $x=1$: $f'(1) = 2(1) = 2$ — exactly matching the numeric
trend from Module 5 ($4, 2.1, 2.01, 2.001, \dots \to 2$).

## Geometric meaning

$f'(x_0)$ is the slope of the **tangent line** to the curve $y=f(x)$ at the
point $(x_0, f(x_0))$ — the line that just grazes the curve there, matching
its direction exactly at that single point. Where $f'(x) > 0$ the function
is increasing; where $f'(x) < 0$ it's decreasing; where $f'(x) = 0$ the
tangent is flat — a candidate minimum, maximum, or saddle point. This last
fact is *the entire reason derivatives matter for ML*: the minimum of a cost
function occurs where its derivative is zero (or, for gradient descent,
where the gradient is very small).

## Worked numeric example: verifying $f'(x)=2x$ with finite differences

We can approximate $f'(x)$ numerically without symbolic calculus, using a
small but nonzero $h$ directly in the definition:

$$
f'(x) \approx \frac{f(x+h) - f(x)}{h} \quad \text{for small } h
$$

At $x = 3$: exact answer is $f'(3) = 2(3) = 6$.

Approximate with $h = 0.001$:

$$
\frac{f(3.001)-f(3)}{0.001} = \frac{9.006001 - 9}{0.001} = \frac{0.006001}{0.001} = 6.001
$$

Very close to the exact value $6$ — the tiny discrepancy ($0.001$) is
because $h$ is small but not literally zero.

```python
import numpy as np

def f(x):
    return x ** 2

def f_prime_exact(x):
    return 2 * x

def f_prime_numeric(x, h=1e-3):
    return (f(x + h) - f(x)) / h

x0 = 3.0
print("exact f'(3)   :", f_prime_exact(x0))
print("numeric f'(3) :", f_prime_numeric(x0))
print("numeric, smaller h:", f_prime_numeric(x0, h=1e-6))
```

Expected output (matches the hand computation; smaller `h` gets closer to
exact):

```text
exact f'(3)   : 6.0
numeric f'(3) : 6.000999999999479
numeric, smaller h: 6.000000999417476
```

This "finite difference" trick — approximating a derivative numerically by
plugging a tiny $h$ into the definition — is a standard way to **sanity
check** a hand-derived (symbolic) derivative or a backpropagation
implementation, and we'll reuse it in the capstone and again in Level 3.

## Exercise

1. Using the first-principles definition, prove that the derivative of
   $f(x) = 3x$ is $f'(x) = 3$ (hint: it should not depend on $x$ at all,
   since a line's slope is constant).
2. Using the first-principles definition, derive $f'(x)$ for
   $f(x) = x^2 + 5$ (hint: the constant $5$ should vanish in the process —
   think about why).
3. Write a `f_prime_numeric` function for $f(x) = x^2 + 5$ and confirm it
   matches your symbolic answer at $x=2$ within a small tolerance.
