# 07 · Basic Derivative Rules

Deriving every function from first principles (Module 6's limit definition)
would make ML math unbearably slow. In practice we use a small toolkit of
**rules**, each provable from first principles once and reused forever.

## The power rule

$$
\frac{d}{dx}\left[x^n\right] = n\,x^{n-1}
$$

This generalizes what we proved by hand in Module 6: for $n=2$,
$\frac{d}{dx}[x^2] = 2x^{2-1} = 2x$ — exactly matching our first-principles
derivation. A few more examples: $\frac{d}{dx}[x^3] = 3x^2$,
$\frac{d}{dx}[x] = 1\cdot x^0 = 1$ (slope of the line $y=x$ is 1, as
expected), and $\frac{d}{dx}[1] = \frac{d}{dx}[x^0] = 0$ — the derivative of
any constant is zero (a constant doesn't change, so its rate of change is
zero).

## The constant multiple rule

$$
\frac{d}{dx}\left[c\cdot f(x)\right] = c\cdot f'(x)
$$

Constants "pass through" differentiation. Example:
$\frac{d}{dx}[5x^2] = 5\cdot 2x = 10x$.

## The sum rule

$$
\frac{d}{dx}\left[f(x) + g(x)\right] = f'(x) + g'(x)
$$

You can differentiate term by term. Example: for
$f(x) = 3x^2 + 4x + 7$,

$$
f'(x) = \frac{d}{dx}[3x^2] + \frac{d}{dx}[4x] + \frac{d}{dx}[7] = 6x + 4 + 0 = 6x+4
$$

This single rule is why the derivative of a sum-of-squared-errors cost
function can be computed term by term — differentiate the error contributed
by each data point separately, then sum (used constantly from Module 9
onward).

## The chain rule (introduction)

For a **composite** function $h(x) = f(g(x))$ (Module 5's terminology), the
chain rule says:

$$
h'(x) = f'(g(x))\cdot g'(x)
$$

In words: differentiate the "outer" function (treating the inner function as
a single blob), then multiply by the derivative of the "inner" function.
This is the single most important rule for ML — it is *literally* the
mathematical mechanism behind backpropagation (Level 3), since a neural
network is one long composition of functions.

### Worked chain rule example

Let $h(x) = (3x+1)^2$. This is a composition: outer function
$f(u) = u^2$, inner function $g(x) = 3x+1$, with $u = g(x)$.

- $f'(u) = 2u$ (power rule), so $f'(g(x)) = 2(3x+1)$.
- $g'(x) = 3$ (sum rule + constant multiple rule: derivative of $3x$ is 3,
  derivative of constant $1$ is 0).

$$
h'(x) = f'(g(x))\cdot g'(x) = 2(3x+1)\cdot 3 = 6(3x+1) = 18x+6
$$

**Check by expanding first.** $(3x+1)^2 = 9x^2 + 6x + 1$. By the power/sum
rules directly: $\frac{d}{dx}[9x^2+6x+1] = 18x + 6$ — matches exactly.

## Worked numeric example

Let's verify $h'(x) = 18x+6$ for $h(x)=(3x+1)^2$ at $x=2$, both
symbolically and via finite differences (Module 6's technique).

Symbolic: $h'(2) = 18(2)+6 = 42$.

Finite difference with $h=0.001$: $h(2.001) = (3(2.001)+1)^2 = (7.003)^2 = 49.042009$,
and $h(2) = 7^2 = 49$, so
$\frac{h(2.001)-h(2)}{0.001} = \frac{0.042009}{0.001} = 42.009$ — very close
to the exact $42$.

```python
import numpy as np

def h(x):
    return (3 * x + 1) ** 2

def h_prime_exact(x):
    return 18 * x + 6

def h_prime_numeric(x, eps=1e-3):
    return (h(x + eps) - h(x)) / eps

x0 = 2.0
print("exact  h'(2):", h_prime_exact(x0))
print("numeric h'(2):", h_prime_numeric(x0))
```

Expected output (matches hand computation):

```text
exact  h'(2): 42
numeric h'(2): 42.00899999999978
```

## Exercise

1. Using the power, constant-multiple, and sum rules, differentiate
   $f(x) = 4x^3 - 2x^2 + 7x - 5$ by hand.
2. Using the chain rule, differentiate $g(x) = (2x - 5)^3$ by hand (outer
   function $u^3$, inner function $2x-5$).
3. Expand $(2x-5)^3$ fully and differentiate term-by-term with the power
   rule; confirm it matches your chain-rule answer from step 2.
4. Write `g_prime_numeric` using the finite-difference trick and confirm it
   matches your symbolic answer at $x=1$.
