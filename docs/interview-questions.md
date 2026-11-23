---
title: "ML Math Foundations Interview Questions and Answers (180 Questions)"
description: "180 ML Math Foundations interview questions with answers and explanations, from beginner to advanced, organised by level. Free, from the ML Math Foundations Mastery Path."
---

# ML Math Foundations Interview Questions and Answers

**180 questions** with the correct answer and a short explanation, organised from beginner (Level 1) to advanced (Level 4). Every question comes from a lesson in the free ML Math Foundations Mastery Path, so you can open the lesson to go deeper.

!!! tip "Practise, don't just read"
    Each lesson also ends with an interactive quiz, and the [final quiz](quiz.md) tests all four levels. Try to answer each question below before reading the answer.

## Level 1 · Entry

### [01 · Why Math Matters for ML](level-1/01-why-math-matters.md)

**Q1.** What are the three mathematical pillars of ML in this lesson?

**Answer:** Linear algebra, calculus and probability &amp; statistics

Data and parameters are vectors and matrices; training needs derivatives; loss functions come from probability.

**Q2.** What do the symbols in θ_new = θ_old − α∇J(θ) mean?

**Answer:** θ is a parameter vector, ∇J the gradient of the cost, α a scalar learning rate

It's the gradient descent update.

**Q3.** Why is calculus needed for training?

**Answer:** The derivative of the error tells you which direction to adjust each parameter

It is the mechanism of learning.

**Q4.** Why does the lesson say math matters even if .fit() runs without it?

**Answer:** When training diverges, loss becomes NaN or a model won't learn, the math is the only way through

It also helps you read papers' notation.

### [02 · Scalars, Vectors &amp; Matrices](level-1/02-scalars-vectors-matrices.md)

**Q5.** A dataset has n examples with d features each. What is its matrix shape?

**Answer:** n × d, one row per example

Each row is an example vector and each column a feature.

**Q6.** What does 'rows first, columns second' mean for a 3 × 2 matrix?

**Answer:** 3 rows and 2 columns

Entry a_ij is row i, column j.

**Q7.** What is a tensor?

**Answer:** A generalization with more axes, like a batch of images stacked into 3D/4D

Level 1 sticks to scalars, vectors and matrices.

**Q8.** What does the identity matrix do when you multiply by it?

**Answer:** Changes nothing, like the number 1

It has 1s on the diagonal and 0s elsewhere.

### [03 · Vector Operations](level-1/03-vector-operations.md)

**Q9.** What is the dot product of [1,2,3] and [4,5,6]?

**Answer:** 32

1·4 + 2·5 + 3·6 = 4 + 10 + 18.

**Q10.** What does a dot product of 0 tell you about two vectors?

**Answer:** They are orthogonal (perpendicular)

u·v = ‖u‖‖v‖cos θ, which is 0 at 90°.

**Q11.** What is the Euclidean norm of [3, 4]?

**Answer:** 5

√(9 + 16) = √25.

**Q12.** What does a linear model's prediction ŷ = w·x correspond to?

**Answer:** A dot product of weights and features

Every linear layer in a neural network is a stack of dot products; cosine similarity compares embeddings.

### [04 · Matrix Operations](level-1/04-matrix-operations.md)

**Q13.** For AB to be defined with A (m×n) and B (p×q), what must hold?

**Answer:** n = p, and the result is m × q

The inner dimensions must match.

**Q14.** For A = [[1,2],[3,4]] and B = [[5,6],[7,8]], what is (AB)₁₁?

**Answer:** 19

1·5 + 2·7 = 19.

**Q15.** Is matrix multiplication commutative?

**Answer:** No, AB ≠ BA in general

Check it yourself in the exercise.

**Q16.** What does ŷ = Xw compute?

**Answer:** All n predictions at once

X is n × d and w is d × 1.

### [05 · Functions &amp; Graphs Refresher](level-1/05-functions-graphs.md)

**Q17.** What is training a model, in terms of the cost function J(θ)?

**Answer:** Finding the input θ that minimizes J

J takes parameters as input and outputs how wrong the model is.

**Q18.** Why can gradient descent reliably find the minimum of a linear regression cost?

**Answer:** It is a quadratic 'bowl' with one unique minimum

A parabola with a &gt; 0 opens upward with a single minimum.

**Q19.** What is a composite function?

**Answer:** One function plugged into another, h(x) = f(g(x))

A neural network is a long chain of compositions, differentiated by the chain rule.

**Q20.** What is the average rate of change between x₁ and x₂?

**Answer:** The slope of the secant line: (f(x₂) − f(x₁)) / (x₂ − x₁)

Shrinking the gap to zero gives the instantaneous rate of change.

### [06 · What Is a Derivative?](level-1/06-what-is-a-derivative.md)

**Q21.** What is the derivative geometrically?

**Answer:** The slope of the tangent line at a point

It's the instantaneous rate of change.

**Q22.** What is f'(x) for f(x) = x² derived from first principles?

**Answer:** 2x

Expanding (x+h)² and cancelling h gives lim (2x + h) = 2x.

**Q23.** Using h = 0.001 at x = 3 for f(x) = x², the finite-difference estimate is?

**Answer:** 6.001

(9.006001 − 9)/0.001 = 6.001, close to the exact 6.

**Q24.** Where f'(x) = 0, what does the tangent look like and why does it matter for ML?

**Answer:** It is flat, marking a candidate minimum, maximum or saddle; minima of the cost occur there

That's why derivatives matter for training.

### [07 · Basic Derivative Rules](level-1/07-derivative-rules.md)

**Q25.** What is d/dx[x³] by the power rule?

**Answer:** 3x²

n·x^(n−1) with n = 3.

**Q26.** What is the derivative of f(x) = 3x² + 4x + 7?

**Answer:** 6x + 4

The sum rule works term by term; the constant's derivative is 0.

**Q27.** For h(x) = (3x + 1)², what is h'(x) by the chain rule?

**Answer:** 18x + 6

Outer derivative 2(3x+1) times inner derivative 3, which equals 18x + 6.

**Q28.** Why is the chain rule the most important rule for ML?

**Answer:** It is the mathematical mechanism behind backpropagation, since a network is a composition of functions

Backpropagation is covered in Level 3.

### [08 · Partial Derivatives &amp; Gradients](level-1/08-partial-derivatives-gradients.md)

**Q29.** What is ∂f/∂x for f(x,y) = x²y + 3y² + 5x?

**Answer:** 2xy + 5

Treat y as a constant: x²y gives 2xy, 3y² gives 0, and 5x gives 5.

**Q30.** What is the gradient ∇f?

**Answer:** A vector of all partial derivatives

For the example, ∇f = [2xy + 5, x² + 6y].

**Q31.** What does ∇f(2, 1) equal for that function?

**Answer:** [9, 10]

2(2)(1) + 5 = 9 and 2² + 6(1) = 10.

**Q32.** Which direction does −∇f point?

**Answer:** The direction of steepest decrease, which is why gradient descent steps along it

It's the locally best direction to reduce the cost.

### [09 · Linear Regression as a Math Example](level-1/09-linear-regression-math.md)

**Q33.** What cost function does the lesson use for linear regression?

**Answer:** Mean squared error J(w,b) = (1/n) Σ (wx_i + b − y_i)²

A function of two variables, w and b.

**Q34.** What is ∂J/∂b for that cost?

**Answer:** (2/n) Σ (wx_i + b − y_i)

By the chain rule ∂u_i/∂b = 1; for w it's x_i.

**Q35.** Why does the cost have a single global minimum?

**Answer:** It is quadratic in (w, b), a bowl shape

Set both partial derivatives to zero to solve.

**Q36.** What does the closed-form b* equal?

**Answer:** ȳ − w*x̄

With w* = Σ(x−x̄)(y−ȳ)/Σ(x−x̄)².

### [10 · Capstone — Derive &amp; Verify a Gradient](level-1/10-capstone-gradient.md)

**Q37.** For J(w,b) = (w−3)² + (b+2)² + wb, what is ∂J/∂w?

**Answer:** 2(w − 3) + b

The (b+2)² term has no w, and wb differentiates to b.

**Q38.** What is ∇J(1, 1)?

**Answer:** [−3, 7]

2(1−3)+1 = −3 and 2(1+2)+1 = 7.

**Q39.** What is J(1, 1)?

**Answer:** 14

4 + 9 + 1.

**Q40.** Why verify a hand-derived gradient numerically in NumPy?

**Answer:** To cross-check the hand work against finite differences

Both a symbolic formula and a numerical check are compared.

## Level 2 · Intermediate

### [01 · Gradient Descent Step-by-Step](level-2/01-gradient-descent.md)

**Q41.** What is the gradient descent update?

**Answer:** θ_{t+1} = θ_t − α∇J(θ_t)

α is the learning rate.

**Q42.** For J(θ) = (θ−3)² + 2, starting at 0 with α = 0.3, what is θ₁?

**Answer:** 1.8

J'(0) = −6, so θ₁ = 0 − 0.3(−6) = 1.8.

**Q43.** By what factor does the distance to the minimum shrink each step in that example?

**Answer:** 0.4 (1 − 2α)

The contraction factor is |1 − 2α|.

**Q44.** What happens for that quadratic when α &gt; 1?

**Answer:** It diverges, since |1 − 2α| exceeds 1

Too small a rate is slow; α near 0.5 converges in one step here.

### [02 · Chain Rule Deep Dive](level-2/02-chain-rule-deep-dive.md)

**Q45.** What is the multivariable chain rule for z = f(x,y) with x(t), y(t)?

**Answer:** dz/dt = ∂f/∂x · dx/dt + ∂f/∂y · dy/dt

Each path from t to z contributes a term.

**Q46.** For z = x²y, x = t², y = 3t, what is dz/dt?

**Answer:** 15t⁴

z = 3t⁵ by substitution, so dz/dt = 15t⁴.

**Q47.** Why does this rule matter for ML?

**Answer:** It makes backpropagation possible: each layer contributes a gradient term combined via dot products along the graph

Neural networks compose many multivariable functions.

**Q48.** What does the numeric check at t = 2 expect?

**Answer:** exact and numeric both 240

15·2⁴ = 240.

### [03 · Jacobians &amp; Hessians](level-2/03-jacobians-hessians.md)

**Q49.** What shape is the Jacobian of f: ℝⁿ → ℝᵐ?

**Answer:** m × n

Row i is ∇f_i transposed: all the per-output gradients stacked.

**Q50.** What is the Jacobian of f(x,y) = (x²y, x + y²) at (1, 2)?

**Answer:** [[4, 1], [1, 4]]

J = [[2xy, x²], [1, 2y]].

**Q51.** What does the Hessian contain?

**Answer:** Second partial derivatives of a scalar function, describing local curvature

For smooth functions it is symmetric.

**Q52.** All Hessian eigenvalues are positive at a critical point. What is it?

**Answer:** A minimum

All negative means maximum; mixed signs means saddle.

### [04 · Eigenvalues &amp; Eigenvectors Intuition](level-2/04-eigenvalues-eigenvectors.md)

**Q53.** What is an eigenvector of a square matrix A?

**Answer:** A nonzero v with Av = λv, which A only scales without changing its direction

λ is its eigenvalue.

**Q54.** How are eigenvalues found?

**Answer:** By solving the characteristic equation det(A − λI) = 0

A − λI must be singular for a nonzero v to exist.

**Q55.** What are the eigenvalues of A = [[4,1],[2,3]]?

**Answer:** 5 and 2

λ² − 7λ + 10 = 0 gives (λ − 5)(λ − 2) = 0.

**Q56.** Which property do symmetric matrices like covariance and Hessians have?

**Answer:** Real eigenvalues and orthogonal eigenvectors

For PCA, eigenvectors of the covariance matrix are the principal directions.

### [05 · Matrix Decompositions Overview](level-2/05-matrix-decompositions.md)

**Q57.** Which decomposition applies to any matrix, even non-square?

**Answer:** SVD: A = UΣVᵀ

Eigendecomposition needs a square matrix.

**Q58.** How do singular values relate to eigenvalues?

**Answer:** σ_i = √λ_i(AᵀA)

They're non-negative and sorted descending.

**Q59.** For a symmetric A, eigendecomposition becomes?

**Answer:** A = VΛVᵀ with orthogonal V

Because V⁻¹ = Vᵀ when V is orthogonal.

**Q60.** For A = diag(3, −2), what are the singular values?

**Answer:** 3 and 2

AᵀA has eigenvalues 9 and 4, so σ = 3 and 2.

### [06 · Probability Basics](level-2/06-probability-basics.md)

**Q61.** What is P(A ∪ B) in general?

**Answer:** P(A) + P(B) − P(A ∩ B)

Inclusion–exclusion.

**Q62.** For a fair die, A = even, B = ≥ 4. What is P(A | B)?

**Answer:** 2/3

P(A∩B) = 2/6 and P(B) = 3/6.

**Q63.** Are A and B independent there?

**Answer:** No: P(A)P(B) = 0.25 but P(A∩B) = 1/3

Independence needs P(A∩B) = P(A)P(B).

**Q64.** What does 'naive' in naive Bayes mean?

**Answer:** The assumption that features are independent, often false but useful

It applies conditional probability and independence rules directly.

### [07 · Expectation &amp; Variance](level-2/07-expectation-variance.md)

**Q65.** What is E[X] for a fair die roll?

**Answer:** 3.5

(1+2+3+4+5+6)/6 = 21/6.

**Q66.** What is Var(X) for a fair die?

**Answer:** About 2.9167

E[X²] − (E[X])² = 91/6 − 12.25.

**Q67.** What does Var(aX + b) equal?

**Answer:** a²Var(X)

Shifting doesn't change spread; scaling by a scales variance by a².

**Q68.** Is E[X + Y] = E[X] + E[Y] true for dependent variables?

**Answer:** Yes, expectation is linear

Linearity holds regardless of dependence.

### [08 · Bayes' Theorem](level-2/08-bayes-theorem.md)

**Q69.** What is Bayes' theorem?

**Answer:** P(A|B) = P(B|A)P(A) / P(B)

Posterior = likelihood × prior / evidence.

**Q70.** A disease affects 1%; a test has a 99% true positive rate and 5% false positive rate. What is P(D | +)?

**Answer:** About 16.7%

0.0099 / 0.0594 ≈ 0.1667.

**Q71.** Why is the positive result only about 16.7% reliable?

**Answer:** The disease is rare, so false positives from the large healthy population dominate (base-rate fallacy)

Bayes' theorem corrects for the base rate.

**Q72.** What does naive Bayes assume?

**Answer:** Features are conditionally independent given the class

So P(x|c) = ∏ P(x_i|c).

### [09 · Distributions Used in ML](level-2/09-distributions-in-ml.md)

**Q73.** What are the mean and variance of a Bernoulli(p)?

**Answer:** p and p(1−p)

It models a single binary trial.

**Q74.** What is P(X = 6) for 10 fair coin flips?

**Answer:** 210/1024 ≈ 0.2051

C(10,6)·0.5¹⁰ = 210/1024.

**Q75.** What are the Binomial mean and variance for n = 10, p = 0.5?

**Answer:** 5 and 2.5

np and np(1−p).

**Q76.** Why is the Gaussian so common in ML?

**Answer:** The Central Limit Theorem: sums of many independent effects tend toward it; Gaussian errors lead to MSE loss

Noise and weight initializations often assume it.

### [10 · Capstone — Probability &amp; Optimization Mini-Project](level-2/10-capstone-probability-optimization.md)

**Q77.** What does the capstone estimate with gradient descent?

**Answer:** The heads probability p, by minimizing the negative log-likelihood of Bernoulli flips

The same strategy trains logistic regression.

**Q78.** Why take logs of the likelihood?

**Answer:** It turns the product into a sum, which is much easier to differentiate

ℓ(p) = Σ[y ln p + (1−y) ln(1−p)].

**Q79.** With 7 heads in 10 flips, what is the closed-form optimum p*?

**Answer:** 0.7

n₁/n, the sample mean.

**Q80.** For n₁ = 7, n₀ = 3 and α = 0.02, what is p₁ starting from p = 0.5?

**Answer:** 0.66

dJ/dp = −14 + 6 = −8, so p₁ = 0.5 + 0.16.

## Level 3 · Advanced

### [01 · Backpropagation from the Chain Rule](level-3/01-backpropagation-chain-rule.md)

**Q81.** What is backpropagation, per the lesson?

**Answer:** The multivariable chain rule applied layer by layer, reusing intermediate results

The forward pass caches every intermediate value so none is recomputed.

**Q82.** For a = σ(z), what is ∂a/∂z?

**Answer:** a(1 − a)

The sigmoid derivative.

**Q83.** For x = 2, y = 1, w = 0.5, b = 0, what are z and a?

**Answer:** z = 1.0 and a ≈ 0.7311

z = 0.5·2 + 0 = 1 and σ(1) ≈ 0.7311.

**Q84.** What is the shared upstream term δ?

**Answer:** ∂L/∂z = 2(a − y)a(1 − a), reused for both ∂L/∂w (× x) and ∂L/∂b

That reuse is the 'backward propagation of error'.

### [02 · Loss Functions &amp; Their Derivatives](level-3/02-loss-functions-derivatives.md)

**Q85.** What is the gradient of MSE with respect to a single prediction?

**Answer:** (2/n)(ŷ_i − y_i)

Power rule plus linearity under the sum.

**Q86.** What does composing BCE with a sigmoid give for ∂L/∂z_i?

**Answer:** (1/n)(ŷ_i − y_i), because the ŷ(1−ŷ) terms cancel

It's why frameworks fuse sigmoid + BCE into one stable operation.

**Q87.** For ŷ = [0.8, 0.3] and y = [1, 0], what is the MSE?

**Answer:** 0.065

½(0.04 + 0.09).

**Q88.** What is BCE the per-example, averaged form of?

**Answer:** The negative log-likelihood from the Bernoulli model

It's the same strategy as Level 2's capstone.

### [03 · Optimization Beyond Vanilla GD](level-3/03-momentum-adam.md)

**Q89.** What does momentum use instead of the raw gradient?

**Answer:** An exponentially decaying running average of past gradients

Like a ball rolling downhill, it smooths oscillations.

**Q90.** For J(θ) = (θ−3)² + 2, θ₀ = 0, α = 0.3, β = 0.9, what is θ₁ with momentum?

**Answer:** 0.18

v₁ = 0.1(−6) = −0.6, so θ₁ = 0 + 0.3·0.6.

**Q91.** What two running averages does Adam track?

**Answer:** The gradient mean (first moment) and the squared-gradient average (second moment)

It uses both to adapt step size per parameter.

**Q92.** Why does Adam bias-correct m_t and v_t?

**Answer:** Because they start at zero, early estimates are biased toward zero

m̂ = m/(1−β₁ᵗ) and v̂ = v/(1−β₂ᵗ).

### [04 · Regularization Math (L1/L2)](level-3/04-regularization-math.md)

**Q93.** What is the regularized objective?

**Answer:** J(w) = L(w) + λR(w)

λ controls the penalty strength.

**Q94.** Why is L2 called weight decay?

**Answer:** Each step shrinks w multiplicatively by (1 − 2ηλ) before the data gradient

The gradient is ∇L + 2λw.

**Q95.** Why does L1 give sparsity?

**Answer:** It subtracts a constant amount from each weight regardless of magnitude, and its diamond contour has corners on the axes

L2 shrinks smoothly but rarely to exactly zero.

**Q96.** For w = 3, L = (w−1)², λ = 0.5, η = 0.1, what is the L2 step result?

**Answer:** 2.3

∇ = 4 + 2(0.5)(3) = 7, so w = 3 − 0.7.

### [05 · Maximum Likelihood Estimation](level-3/05-maximum-likelihood-estimation.md)

**Q97.** What does MLE choose?

**Answer:** Parameters that make the observed data most probable under an assumed distribution

Maximizing the likelihood is equivalent to minimizing the negative log-likelihood.

**Q98.** Why maximize the log-likelihood instead of the likelihood?

**Answer:** Products underflow numerically; log is monotonic so the maximizer is unchanged

It also turns products into sums.

**Q99.** MSE is the MLE under which noise model?

**Answer:** Gaussian noise

Dropping constants, minimizing −ℓ is minimizing Σ(y − f)².

**Q100.** Binary cross-entropy is the negative log-likelihood of which model?

**Answer:** Bernoulli

p(y|x) = p̂^y(1−p̂)^(1−y).

### [06 · Softmax &amp; Cross-Entropy Derivatives](level-3/06-softmax-cross-entropy.md)

**Q101.** What does softmax do?

**Answer:** Turns raw logits into a probability distribution

softmax(z)_i = e^{z_i} / Σ e^{z_j}.

**Q102.** What is the combined gradient of softmax + cross-entropy with respect to the logits?

**Answer:** p − y

The cleanest gradient in deep learning.

**Q103.** For logits (2.0, 1.0, 0.1) and true class 0, which is the approximate loss?

**Answer:** 0.4170

−log(0.6590).

**Q104.** Why subtract max(z) in a softmax implementation?

**Answer:** Numerical stability, avoiding overflow in exp

It doesn't change the result.

### [07 · Computational Graphs &amp; Autodiff](level-3/07-computational-graphs-autodiff.md)

**Q105.** What does reverse-mode autodiff do?

**Answer:** A forward pass computes node values, then a backward pass propagates ∂L/∂v from the output using the chain rule

It is what backprop is.

**Q106.** When is reverse mode cheaper than forward mode?

**Answer:** With many inputs and one scalar output, like millions of weights and a loss

One backward pass regardless of parameter count.

**Q107.** For f = (x+y)(xy) at x = 2, y = 3, what is ∂f/∂x?

**Answer:** 21

2xy + y² = 12 + 9.

**Q108.** Why are gradients summed at nodes with several consumers?

**Answer:** x contributes through multiple paths; the multivariable chain rule sums over paths

x feeds both a and b.

### [08 · Batch Normalization Math](level-3/08-batch-normalization-math.md)

**Q109.** What does batch normalization do?

**Answer:** Standardizes activations over a mini-batch, then applies a learnable rescale y = γx̂ + β

It stabilizes training.

**Q110.** Why do γ and β exist?

**Answer:** They let the network undo the normalization if that's optimal

ε prevents division by zero.

**Q111.** For batch (1, 2, 3), what are μ_B and σ_B²?

**Answer:** 2 and 2/3

The population variance is (1 + 0 + 1)/3.

**Q112.** Why is the backward pass easy to get wrong?

**Answer:** μ_B and σ_B² depend on every x_i in the batch, so gradients must account for that shared dependency

It's the sum-over-paths rule applied to the two shared nodes.

### [09 · Numerical Stability in ML Math](level-3/09-numerical-stability.md)

**Q113.** What is the log-sum-exp trick for softmax?

**Answer:** Subtract max(z) before exponentiating: mathematically identical but safe from overflow

Every exponent becomes ≤ 0, so each term is in (0, 1].

**Q114.** Why not compute log(softmax(z)) in two steps?

**Answer:** A probability can underflow to exactly 0, giving log(0); fuse as z_k − logsumexp(z)

The fused form never materializes a zero probability.

**Q115.** What is catastrophic cancellation?

**Answer:** Subtracting two nearly equal large floats destroys precision, e.g. the one-pass variance going negative

The two-pass formula Σ(x − x̄)²/n is more stable.

**Q116.** For logits (1000, 1001, 999), what does the stable softmax give approximately?

**Answer:** (0.2447, 0.6652, 0.0900)

With c = 1001 the exponents are (−1, 0, −2); the naive version overflows to NaN.

### [10 · Capstone — Hand-Derive a Small Network](level-3/10-capstone-small-network.md)

**Q117.** What is the combined gradient ∂L/∂z₂ for sigmoid output with BCE loss?

**Answer:** ŷ − y

For y = 1 and ŷ ≈ 0.4944, δ₂ = −0.5056.

**Q118.** How does the gradient flow into the hidden layer?

**Answer:** δ₁ = δ₂ w₂ ⊙ a₁(1 − a₁): through w₂, then through the sigmoid derivative

Then ∂L/∂W₁ = δ₁ xᵀ and ∂L/∂b₁ = δ₁.

**Q119.** What are z₁ for W₁ = [[0.5, −0.3], [0.8, 0.2]] and x = (1, 2)?

**Answer:** (−0.1, 1.2)

0.5 − 0.6 = −0.1 and 0.8 + 0.4 = 1.2.

**Q120.** How is the hand derivation verified?

**Answer:** Numerically against finite-difference gradients

The capstone combines backprop, loss derivatives and numeric checks.

## Level 4 · Master

### [01 · Convexity &amp; Optimization](level-4/01-convexity-optimization.md)

**Q121.** What is true of every local minimum of a convex function?

**Answer:** It is a global minimum

There are no traps.

**Q122.** How do you test a twice-differentiable function in n dimensions for convexity?

**Answer:** Its Hessian is positive semi-definite everywhere (all eigenvalues ≥ 0)

This connects to the eigenvalue module.

**Q123.** Why does gradient descent find the global optimum for linear and logistic regression but not neural networks?

**Answer:** Their losses are convex in the weights; network loss landscapes are non-convex

Networks can get stuck in local minima or saddle points.

**Q124.** f(w₁,w₂) = w₁² + 2w₂² − w₁w₂ has Hessian eigenvalues of about 1.586 and 4.414. What does that imply?

**Answer:** It is strictly convex

Both are positive, so the Hessian is positive definite.

### [02 · Lagrange Multipliers &amp; Constrained Optimization](level-4/02-lagrange-multipliers.md)

**Q125.** What condition holds at a constrained optimum with g(x) = 0?

**Answer:** ∇f is parallel to ∇g: ∇f = λ∇g

Otherwise you could slide along the constraint and decrease f.

**Q126.** Minimizing x² + y² subject to x + y = 1 gives what point and value?

**Answer:** (0.5, 0.5) with value 0.5

x = y = λ/2 with λ = 1.

**Q127.** What does complementary slackness say about an inequality constraint?

**Answer:** Either the constraint is active (h = 0) or its multiplier μ is zero

The KKT condition μ·h(x) = 0.

**Q128.** How does this relate to SVM support vectors?

**Answer:** Only points on the margin (active constraints) get nonzero μ

That's their 'support'.

### [03 · Information Theory Basics](level-4/03-information-theory-basics.md)

**Q129.** What does entropy H(p) measure?

**Answer:** The average surprise of a distribution, maximal for the uniform distribution and 0 when all mass is on one outcome

−log p_i is the surprise of outcome i.

**Q130.** What is D_KL(p‖q) in terms of cross-entropy?

**Answer:** H(p,q) − H(p)

It is always ≥ 0, with equality iff p = q.

**Q131.** Why does minimizing cross-entropy equal minimizing KL divergence?

**Answer:** H(p) doesn't depend on the model

Training a classifier minimizes KL from the true label distribution to the prediction.

**Q132.** For p = (1,0,0) and q = (0.7,0.2,0.1), what is H(p,q)?

**Answer:** About 0.3567

−log(0.7), with H(p) = 0 for a one-hot label.

### [04 · Logistic Regression Gradient from Scratch](level-4/04-logistic-regression-gradient.md)

**Q133.** What is the gradient of the logistic loss with respect to w?

**Answer:** (1/n) Xᵀ(ŷ − y)

The same form as linear regression, with ŷ = σ(z) instead of identity.

**Q134.** Why is the logistic loss convex?

**Answer:** Its Hessian XᵀSX with S = diag(ŷ(1−ŷ)) is positive semi-definite

ŷ(1 − ŷ) &gt; 0 for all ŷ in (0, 1).

**Q135.** For x = (1,2), y = 1, w = (0.1, −0.2), b = 0, what is ŷ?

**Answer:** About 0.4256

z = −0.3 and σ(−0.3) ≈ 0.4256.

**Q136.** What does convexity guarantee here?

**Answer:** Gradient descent finds the global optimum

The gradient there is (ŷ − y)x = (−0.5744, −1.1488).

### [05 · Kernel Methods &amp; the Kernel Trick](level-4/05-kernel-methods.md)

**Q137.** What does the kernel trick avoid?

**Answer:** Explicitly computing the high-dimensional feature map φ(x)

k(x,z) = φ(x)ᵀφ(z) computed in the original space.

**Q138.** For x = (1,2), z = (3,1), what is the polynomial kernel (xᵀz)²?

**Answer:** 25

(3 + 2)² = 25, matching the explicit feature-map dot product.

**Q139.** What does the RBF kernel correspond to?

**Answer:** An infinite-dimensional feature map

k = exp(−‖x−z‖² / 2σ²).

**Q140.** What is Mercer's condition?

**Answer:** A valid kernel's Gram matrix must be positive semi-definite for every finite point set

It guarantees a valid φ exists.

### [06 · Singular Value Decomposition Deep Dive](level-4/06-svd-deep-dive.md)

**Q141.** What does SVD A = UΣVᵀ guarantee?

**Answer:** It exists for any m × n matrix, with orthogonal U and V and non-negative singular values

Σ is diagonal with σ₁ ≥ σ₂ ≥ … ≥ 0.

**Q142.** How are singular values related to AᵀA?

**Answer:** σ_i² are the eigenvalues of AᵀA, and the columns of V are its eigenvectors

AᵀA and AAᵀ are symmetric PSD, so SVD always exists.

**Q143.** What does the Eckart–Young theorem say?

**Answer:** Keeping the top k singular values gives the best rank-k approximation in Frobenius error

It underlies PCA and compression.

**Q144.** What are the singular values of diag(3, −2)?

**Answer:** 3 and 2

AᵀA has eigenvalues 9 and 4.

### [07 · Bayesian Inference Foundations](level-4/07-bayesian-inference.md)

**Q145.** What are the prior, likelihood and posterior?

**Answer:** Belief before data, the data's probability given θ, and the updated belief after data

The evidence p(D) is a normalizing constant.

**Q146.** How does MAP differ from MLE?

**Answer:** MAP maximizes likelihood × prior (prior-informed); MLE ignores the prior

A Gaussian prior on weights adds an L2 penalty λ‖w‖².

**Q147.** A Beta(1,1) prior and 7 heads in 10 flips give which posterior?

**Answer:** Beta(8, 4), with mean about 0.667

Posterior = Beta(α + k, β + n − k).

**Q148.** What does 'conjugate' mean here?

**Answer:** The posterior is in the same family as the prior, with a simple update

Beta is conjugate to Bernoulli.

### [08 · Numerical Optimization Methods](level-4/08-numerical-optimization-methods.md)

**Q149.** What does Newton's method use that gradient descent doesn't?

**Answer:** The Hessian, modeling f locally as a quadratic and jumping to its minimum

x_{k+1} = x_k − H⁻¹∇f.

**Q150.** How many steps does Newton's method need for a truly quadratic f?

**Answer:** One

Near the optimum it converges quadratically for general convex f.

**Q151.** Why is Newton's method rarely used directly in deep learning?

**Answer:** Computing and inverting the Hessian costs O(n³), infeasible for millions or billions of parameters

Practitioners use quasi-Newton methods like L-BFGS or Adam.

**Q152.** For f(x) = x⁴ − 3x³ + 2 from x₀ = 3, what is the Newton step's x₁?

**Answer:** 2.5

3 − 27/54.

### [09 · Math Behind Transformer Attention](level-4/09-attention-math.md)

**Q153.** What is scaled dot-product attention?

**Answer:** softmax(QKᵀ/√d_k)V

It's a weighted average of the values.

**Q154.** Why divide by √d_k?

**Answer:** Dot products have variance d_k; without scaling softmax saturates and gradients vanish

Dividing restores variance to about 1 regardless of dimension.

**Q155.** What does row i, column j of QKᵀ represent?

**Answer:** qᵢ · kⱼ, how much query i attends to key j

Softmax is applied row-wise.

**Q156.** In the worked example both keys score 1 with d_k = 4. What is the output?

**Answer:** (0.5, 0.5)

A tie gives an even blend of v₁ and v₂.

### [10 · Capstone — Linear Regression from Scratch in NumPy](level-4/10-capstone-linear-regression-scratch.md)

**Q157.** What do the normal equations give?

**Answer:** w* = (XᵀX)⁻¹Xᵀy

It requires XᵀX to be invertible.

**Q158.** When do the normal equations fail or become unstable?

**Answer:** When features are collinear (XᵀX singular or ill-conditioned)

SVD handles near-zero singular values gracefully.

**Q159.** For X = [[1,1],[1,2],[1,3]] and y = (2,3,5), what are the fitted bias and slope?

**Answer:** About 0.333 and 1.5

w* = (1/6)(2, 9).

**Q160.** What are the three methods the capstone compares?

**Answer:** Normal equations, gradient descent and the SVD pseudoinverse

All three agree numerically.

## Mixed questions across all levels

**Q161.** A model's loss becomes NaN during training with large logits. Which fix applies?

**Answer:** Use the log-sum-exp / subtract-max trick in softmax

Naive softmax overflows exp; subtracting the max is mathematically identical but safe.

**Q162.** You need all n predictions of a linear model at once. Which expression?

**Answer:** ŷ = Xw with X (n × d) and w (d × 1)

Matrix-vector multiplication computes every prediction in one step.

**Q163.** The gradient of a cost points in which direction, and which way does gradient descent step?

**Answer:** ∇J points steepest up; descent steps along −∇J

θ ← θ − α∇J.

**Q164.** You want the derivative of (3x + 1)². Which rule and result?

**Answer:** Chain rule: 2(3x+1)·3 = 18x + 6

Outer derivative times the inner derivative.

**Q165.** Why is a quick check of a hand-derived gradient with finite differences useful?

**Answer:** It cross-checks the analytic gradient numerically

The capstones verify every hand derivation numerically.

**Q166.** A positive-definite Hessian at a critical point means what?

**Answer:** A local minimum

All eigenvalues positive means a minimum.

**Q167.** Which matrix factorization applies to a non-square matrix?

**Answer:** SVD

SVD exists for any matrix.

**Q168.** For a fair die, what are E[X] and Var(X)?

**Answer:** 3.5 and about 2.92

E[X²] = 91/6 and Var = 91/6 − 12.25.

**Q169.** A disease affects 1% of people; a test has 99% sensitivity and 5% false positives. A positive result means roughly?

**Answer:** A 16.7% chance of disease (base-rate fallacy)

Bayes' theorem corrects for the rare base rate.

**Q170.** Which loss is the MLE under Gaussian noise, and which under a Bernoulli model?

**Answer:** MSE and binary cross-entropy

Negative log-likelihood under each model yields those losses.

**Q171.** Why does a softmax + cross-entropy gradient simplify to p − y?

**Answer:** The p_i(δ_ij − p_j) terms cancel against −y_i/p_i

That is why frameworks fuse the two operations.

**Q172.** Adam divides each step by √v̂. What does that provide?

**Answer:** Adaptive per-parameter learning rates

Large consistent gradients get smaller steps; sparse small gradients get larger ones.

**Q173.** What does L1 regularization tend to do that L2 does not?

**Answer:** Push many weights exactly to zero (sparsity)

Its diamond contour has corners on the axes.

**Q174.** In reverse-mode autodiff, why are gradients summed at a node with two consumers?

**Answer:** The value contributes through multiple paths (multivariable chain rule)

For f = (x+y)(xy), x contributes via both a and b.

**Q175.** Why can gradient descent be trusted to find the global optimum of logistic regression but not of a neural network?

**Answer:** Its loss is convex in the weights; network losses are non-convex

Convexity means every local minimum is global.

**Q176.** You must minimize f subject to g(x) = 0. What do you set up?

**Answer:** A Lagrangian L = f − λg and solve its stationary points

∇f = λ∇g at the constrained optimum.

**Q177.** What does the RBF kernel let an SVM do?

**Answer:** Use an infinite-dimensional feature map without ever constructing it

Only dot products (kernel evaluations) are needed.

**Q178.** What does KL divergence minimization correspond to in classifier training?

**Answer:** Minimizing cross-entropy, since H(p) is constant

D_KL = H(p,q) − H(p).

**Q179.** In attention, why divide QKᵀ by √d_k?

**Answer:** To keep dot-product variance near 1 so softmax doesn't saturate

Raw dot products have variance d_k.

**Q180.** Which approach to least squares is numerically best when features are nearly collinear?

**Answer:** The SVD pseudoinverse

SVD handles near-zero singular values with a cutoff.

---

More free practice: the [ML Math Foundations Mastery Path](index.md) curriculum, the [Interview Prep Pack](https://learnmasterypath.com/interview-pack/) and the [free resume ATS check](https://learnmasterypath.com/resume/).
