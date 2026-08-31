# Level 3 · Advanced <span class="level-badge">In Progress</span>

Goal: assemble Level 1–2 tools into the math that actually trains neural
networks — backpropagation derived from the multivariable chain rule, the
derivatives behind every common loss function, momentum/Adam as
refinements of vanilla gradient descent, regularization, maximum likelihood
estimation as the principle behind most loss functions, softmax/cross-entropy,
computational graphs and autodiff, batch normalization, and the numerical
stability tricks production ML code relies on.

## Modules

1. [Backpropagation from the Chain Rule](01-backpropagation-chain-rule.md)
2. [Loss Functions & Their Derivatives](02-loss-functions-derivatives.md)
3. [Optimization Beyond Vanilla GD](03-momentum-adam.md)
4. [Regularization Math (L1/L2)](04-regularization-math.md)
5. [Maximum Likelihood Estimation](05-maximum-likelihood-estimation.md)
6. [Softmax & Cross-Entropy Derivatives](06-softmax-cross-entropy.md)
7. [Computational Graphs & Autodiff](07-computational-graphs-autodiff.md)
8. [Batch Normalization Math](08-batch-normalization-math.md)
9. [Numerical Stability in ML Math](09-numerical-stability.md)
10. [Capstone — Hand-Derive a Small Network](10-capstone-small-network.md)

By the end of this level you'll be able to hand-derive backpropagation
through a small network, derive gradients for MSE/cross-entropy/softmax,
explain why Adam converges faster than vanilla gradient descent, and reason
about the numerical pitfalls (overflow, underflow, vanishing/exploding
gradients) that separate textbook math from working code — with every
derivation checked against NumPy/PyTorch-style autodiff.
