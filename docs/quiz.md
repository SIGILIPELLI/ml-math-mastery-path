---
title: "Final Quiz"
description: "Test what you have learned across all four levels of ML Math Foundations Mastery Path with a 20-question final quiz."
---

# Final Quiz

20 questions drawn from all four levels of this course. You need **70% (14 out of 20)** to pass —
you can retake it as many times as you like. Your best result is saved in this browser.

<div id="mp-quiz-root"></div>

<script type="application/json" id="mp-quiz-data">
[
  {
    "question": "A model's loss becomes NaN during training with large logits. Which fix applies?",
    "options": [
      "Add more layers",
      "Use the log-sum-exp / subtract-max trick in softmax",
      "Remove the loss",
      "Increase the learning rate"
    ],
    "correctIndex": 1,
    "explanation": "Naive softmax overflows exp; subtracting the max is mathematically identical but safe."
  },
  {
    "question": "You need all n predictions of a linear model at once. Which expression?",
    "options": [
      "ŷ = Xᵀ",
      "ŷ = X + w",
      "ŷ = wX with w (d × 1)",
      "ŷ = Xw with X (n × d) and w (d × 1)"
    ],
    "correctIndex": 3,
    "explanation": "Matrix-vector multiplication computes every prediction in one step."
  },
  {
    "question": "The gradient of a cost points in which direction, and which way does gradient descent step?",
    "options": [
      "They are unrelated",
      "∇J is perpendicular to the surface",
      "∇J points steepest down; descent steps along ∇J",
      "∇J points steepest up; descent steps along −∇J"
    ],
    "correctIndex": 3,
    "explanation": "θ ← θ − α∇J."
  },
  {
    "question": "You want the derivative of (3x + 1)². Which rule and result?",
    "options": [
      "Product rule: 9x",
      "Sum rule: 6x",
      "Power rule only: 2x",
      "Chain rule: 2(3x+1)·3 = 18x + 6"
    ],
    "correctIndex": 3,
    "explanation": "Outer derivative times the inner derivative."
  },
  {
    "question": "Why is a quick check of a hand-derived gradient with finite differences useful?",
    "options": [
      "It sets the learning rate",
      "It picks features",
      "It cross-checks the analytic gradient numerically",
      "It trains the model"
    ],
    "correctIndex": 2,
    "explanation": "The capstones verify every hand derivation numerically."
  },
  {
    "question": "A positive-definite Hessian at a critical point means what?",
    "options": [
      "A local minimum",
      "Nothing",
      "A saddle",
      "A maximum"
    ],
    "correctIndex": 0,
    "explanation": "All eigenvalues positive means a minimum."
  },
  {
    "question": "Which matrix factorization applies to a non-square matrix?",
    "options": [
      "Cholesky",
      "SVD",
      "Eigendecomposition",
      "LU with pivoting only"
    ],
    "correctIndex": 1,
    "explanation": "SVD exists for any matrix."
  },
  {
    "question": "For a fair die, what are E[X] and Var(X)?",
    "options": [
      "3.5 and about 2.92",
      "3 and 3",
      "6 and 1",
      "3.5 and 3.5"
    ],
    "correctIndex": 0,
    "explanation": "E[X²] = 91/6 and Var = 91/6 − 12.25."
  },
  {
    "question": "A disease affects 1% of people; a test has 99% sensitivity and 5% false positives. A positive result means roughly?",
    "options": [
      "50%",
      "99%",
      "95%",
      "A 16.7% chance of disease (base-rate fallacy)"
    ],
    "correctIndex": 3,
    "explanation": "Bayes' theorem corrects for the rare base rate."
  },
  {
    "question": "Which loss is the MLE under Gaussian noise, and which under a Bernoulli model?",
    "options": [
      "L1 and L2",
      "Cross-entropy and MSE",
      "MSE and binary cross-entropy",
      "Hinge and MSE"
    ],
    "correctIndex": 2,
    "explanation": "Negative log-likelihood under each model yields those losses."
  },
  {
    "question": "Why does a softmax + cross-entropy gradient simplify to p − y?",
    "options": [
      "Softmax is linear",
      "y is zero",
      "Logits are equal",
      "The p_i(δ_ij − p_j) terms cancel against −y_i/p_i"
    ],
    "correctIndex": 3,
    "explanation": "That is why frameworks fuse the two operations."
  },
  {
    "question": "Adam divides each step by √v̂. What does that provide?",
    "options": [
      "Adaptive per-parameter learning rates",
      "A larger Hessian",
      "Momentum only",
      "Random noise"
    ],
    "correctIndex": 0,
    "explanation": "Large consistent gradients get smaller steps; sparse small gradients get larger ones."
  },
  {
    "question": "What does L1 regularization tend to do that L2 does not?",
    "options": [
      "Push many weights exactly to zero (sparsity)",
      "Remove the loss",
      "Increase variance",
      "Shrink all weights equally"
    ],
    "correctIndex": 0,
    "explanation": "Its diamond contour has corners on the axes."
  },
  {
    "question": "In reverse-mode autodiff, why are gradients summed at a node with two consumers?",
    "options": [
      "To save memory",
      "It is a bug",
      "To normalize",
      "The value contributes through multiple paths (multivariable chain rule)"
    ],
    "correctIndex": 3,
    "explanation": "For f = (x+y)(xy), x contributes via both a and b."
  },
  {
    "question": "Why can gradient descent be trusted to find the global optimum of logistic regression but not of a neural network?",
    "options": [
      "It uses momentum",
      "Its loss is convex in the weights; network losses are non-convex",
      "It uses the sigmoid",
      "It has fewer features"
    ],
    "correctIndex": 1,
    "explanation": "Convexity means every local minimum is global."
  },
  {
    "question": "You must minimize f subject to g(x) = 0. What do you set up?",
    "options": [
      "A Hessian",
      "A Lagrangian L = f − λg and solve its stationary points",
      "A norm",
      "A Jacobian"
    ],
    "correctIndex": 1,
    "explanation": "∇f = λ∇g at the constrained optimum."
  },
  {
    "question": "What does the RBF kernel let an SVM do?",
    "options": [
      "Avoid labels",
      "Reduce classes",
      "Skip the data",
      "Use an infinite-dimensional feature map without ever constructing it"
    ],
    "correctIndex": 3,
    "explanation": "Only dot products (kernel evaluations) are needed."
  },
  {
    "question": "What does KL divergence minimization correspond to in classifier training?",
    "options": [
      "Maximizing entropy",
      "Maximizing loss",
      "Minimizing cross-entropy, since H(p) is constant",
      "Minimizing variance"
    ],
    "correctIndex": 2,
    "explanation": "D_KL = H(p,q) − H(p)."
  },
  {
    "question": "In attention, why divide QKᵀ by √d_k?",
    "options": [
      "To keep dot-product variance near 1 so softmax doesn't saturate",
      "To speed up exp",
      "To shrink V",
      "To remove keys"
    ],
    "correctIndex": 0,
    "explanation": "Raw dot products have variance d_k."
  },
  {
    "question": "Which approach to least squares is numerically best when features are nearly collinear?",
    "options": [
      "Gradient descent with a huge rate",
      "The SVD pseudoinverse",
      "Guessing",
      "The normal equations with an explicit inverse"
    ],
    "correctIndex": 1,
    "explanation": "SVD handles near-zero singular values with a cutoff."
  }
]
</script>
