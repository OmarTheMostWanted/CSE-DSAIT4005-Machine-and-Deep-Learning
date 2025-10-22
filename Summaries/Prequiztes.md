## Linear combination
A linear combination is an expression formed by multiplying each element of a given set by a scalar (coefficient) and adding the results together. This applies to numbers, vectors, functions, matrices, or any objects in a vector space.

### Formal definition
Let \(v_1,\dots,v_n\) be elements of a vector space and \(a_1,\dots,a_n\) be scalars. A linear combination of the \(v_i\) is
\[
w = a_1 v_1 + a_2 v_2 + \dots + a_n v_n.
\]
The \(a_i\) are called the coefficients of the linear combination.

---

## Sigmoid function
A sigmoid function is a smooth S-shaped function that maps any real number to a value between 0 and 1. The most common sigmoid used in machine learning and logistic regression is the logistic function \(\sigma(x)\) defined by  
\[
\displaystyle \sigma(x)=\frac{1}{1+e^{-x}}.
\]  
This function is widely used because it is continuous, differentiable, monotonic, and saturates toward 0 and 1 for large negative and large positive inputs respectively.

### Key properties
- **Range**: \(\sigma(x)\in(0,1)\) for all real \(x\).  
- **Symmetry**: \(\sigma(-x)=1-\sigma(x)\).  
- **Derivative**: \(\displaystyle \sigma'(x)=\sigma(x)\bigl(1-\sigma(x)\bigr)\).  
- **Behaviour**: as \(x\to+\infty\) the value approaches 1, and as \(x\to-\infty\) it approaches 0.

## Sigmoid probability
A sigmoid probability is the value \(\sigma(z)\) when the input \(z\) is interpreted as a log-odds (logit) score. In logistic regression the model computes a linear combination \(z=\beta_0+\beta_1 x_1+\dots+\beta_p x_p\) and converts it to a probability of the positive class by \(P(y=1\mid x)=\sigma(z)\). The output can therefore be interpreted as the model’s estimated probability for the positive outcome because it lies between 0 and 1 and varies smoothly with the input score.

---

## Log-likelihood
The log-likelihood is the sum of the logarithms of the probabilities the model assigns to the observed outcomes for all training examples.  
For binary logistic regression with N examples \( (x^{(i)}, y^{(i)}) \) where `x` is the feature vector, \( y^{(i)}\in\{0,1\} \) the label and model probability \( p^{(i)}=\sigma(z^{(i)}) \) where \( z^{(i)}=\beta_0+\beta_1x_1+\dots+\beta_px_p \), the log-likelihood is  
\[
\ell(\beta)=\sum_{i=1}^N \bigl[y^{(i)}\log p^{(i)} + (1-y^{(i)})\log(1-p^{(i)})\bigr].
\]  

where \(\beta\) is the vector of model parameters (coefficients).

Maximizing this quantity finds the parameter vector β that makes the observed labels most probable under the model.


## Loss (negative log-likelihood / cross-entropy)
The loss used in logistic regression is the negative log-likelihood, also called log loss or binary cross-entropy.  
The loss for the dataset is  
\[
L(\beta) = -\ell(\beta) = -\sum_{i=1}^N \bigl[y^{(i)}\log p^{(i)} + (1-y^{(i)})\log(1-p^{(i)})\bigr].
\]  
Minimizing this loss is equivalent to maximizing the log-likelihood and is the standard objective in logistic regression and related classifiers.

### Why use log-likelihood and this loss
- The log transforms product-of-probabilities into a sum, improving numerical stability and analytic convenience.  
- The negative log-likelihood is convex for logistic regression, giving a single global minimum for the parameters in the usual (non-regularized) case.  
- The loss penalizes confident but wrong predictions heavily and rewards confident correct predictions, which aligns training with probabilistic accuracy.

### Optimization and gradients
- The gradient of the loss with respect to parameters is computed from residuals \( (y^{(i)} − p^{(i)}) \).
    - Where \( p^{(i)} \) is the model’s predicted probability for example \( i \) and \( y^{(i)} \) is the true label (0 or 1).
    - This difference measures how much the model’s predicted probability deviates from the true label for each example.
- For the intercept (bias) and one feature x:  
  \[
  \frac{\partial L}{\partial \beta_j} = -\sum_{i=1}^N (y^{(i)}-p^{(i)}) x_j^{(i)}
  \]  


Where:
    - N: number of training examples (how many rows of data you have).  
    - \( x^{(i)} \): the feature vector for example number i (all input values for that example).  
    - \( x_j^{(i)} \): the j-th feature value of example i (one number inside x^{(i)}).  
    - \( y^{(i)} \): the true label for example i; either \( 0 \) or \( 1 \) (the real answer we want the model to predict).  
    - \( \beta \) (beta): the parameter vector the model learns. \( \beta_j \) is the parameter (weight) associated with feature j. \( \beta_0 \) is the intercept (bias).  
    - \( z^{(i)} \): the linear score for example i, computed as the weighted sum of features:  
    \[
    z^{(i)}=\beta_0+\beta_1 x_1^{(i)}+\beta_2 x_2^{(i)}+\dots
    \]
    Think of z as the raw number the model uses before turning it into a probability.  
    - σ(·) or \( p^{(i)} \): the sigmoid of \( z^{(i)} \), i.e. the model’s predicted probability that \( y^{(i)}=1 \):  
    \[
    p^{(i)}=\sigma\bigl(z^{(i)}\bigr)=\frac{1}{1+e^{-z^{(i)}}}.
    \]
    - L or ℓ: the loss (negative log-likelihood) — a single number that measures how bad the model’s predictions are on the whole dataset. Smaller is better.  
    - \( ∂L/∂β_j \): the gradient (partial derivative) of the loss with respect to parameter \( β_j \) which is the direction and amount we should change \( β_j \) to reduce the loss.

- Iterative optimization methods (gradient descent, Newton-Raphson, quasi-Newton) use these gradients (and optionally second-derivative information) to find the β that minimizes the loss and thus maximizes likelihood.


#### Intuition: how the parts fit together

1. Start with features x and parameters β. Multiply and add them to get a score z for each example. z is a single real number that summarizes how strongly the model thinks the example should be class 1.  
2. Pass z through the sigmoid to get p between 0 and 1. p is the model’s belief (probability) that the label is 1.  
3. Compare p to the true label y. If y=1 and p is small, the model is wrong and the loss is large for that example. If y=0 and p is large, the model is also wrong. The loss adds up these penalties across examples.  
4. To improve the model we compute how the loss changes if we change each \( \beta_j \). That change is the gradient \( ∂L/∂β_j \). We then move \( β_j \) a little in the direction that reduces the loss (gradient descent).

#### Why the residual (y − p) appears and what it means

- **Residual = \(y^{(i)} - p^{(i)}\)** is the signed error for example i.  
  - If y=1 and p=0.2, residual = 0.8 (model underestimates probability — push p up).  
  - If y=0 and p=0.9, residual = -0.9 (model overestimates probability — push p down).  
- The gradient contribution from example i to parameter \( \beta_j \) is proportional to \( (y^{(i)} − p^{(i)}) × x_j^{(i)} \).  
  - Multiplying by \( x_j^{(i)} \) means features with larger values for that example should cause larger parameter changes for \( \beta_j \).  
  - Summing over i accumulates evidence across all examples.

Intuitive summary: the gradient collects all per-example errors, weights them by the feature value, and tells each \( \beta_j \) how much it contributed to the total error and in which direction to change.

#### Intercept (\( \beta_0 \)) special case

- For \( \beta_0 \) we set \( x_0^{(i)} = 1 \), so its gradient is just the sum of residuals:  
  \[
  \frac{\partial L}{\partial \beta_0} = -\sum_{i=1}^N (y^{(i)}-p^{(i)}).
  \]
- Intuition: the intercept shifts all predictions up or down uniformly; its update depends only on whether, overall, the model is predicting too large or too small probabilities.


### Final intuition in one line

The model predicts probabilities p from scores z; residuals (y−p) tell you the direction and size of error per example; multiplying by feature values assigns that error to each parameter; the gradient sums those contributions and tells you how to change each \( \beta \) to reduce overall error.

---

## Gradient Descent

Gradient descent is an iterative method for finding a set of parameters that makes a chosen objective (loss) as small as possible. Imagine the loss as a landscape of hills and valleys; gradient descent moves the parameters downhill, step by step, until it reaches a valley (ideally the lowest one).


### Symbols and what they mean (plain language)
- \( f(\beta) \) or \( L(\beta) \): the loss function you want to minimize; it maps parameters \( \beta \) to a single number (higher = worse).  
- \( \beta \): the vector of parameters (weights) you are optimizing; \( \beta_j \) is the j-th parameter.  
- \( ∂L/∂β_j \) or \( ∇L(β) \): derivative (gradient) of the loss with respect to \( β \); it points in the direction of steepest increase of the loss.  
- η (eta) or α (alpha): learning rate, a positive scalar that controls how big a step you take each iteration.  
- t or k: iteration index (which step of the algorithm you are on).  
- \( \beta(t) \): parameter vector at iteration t.  
- \( \beta(t+1) \): updated parameters after taking a step at iteration t.


### The update rule and its intuition
Basic (batch) gradient descent update for parameter \( \beta_j \) at iteration \( t \) is:
\[
\beta_j^{(t+1)} = \beta_j^{(t)} - \eta \cdot \frac{\partial L}{\partial \beta_j}\bigg|_{\beta^{(t)}}.
\]

Intuition:
- The gradient \( ∂L/∂β_j \) tells how the loss changes if you nudge \( β_j \) a tiny bit; a positive gradient means increasing \( β_j \) would increase loss, a negative gradient means increasing \( β_j \) would decrease loss.  
- We move in the opposite direction of the gradient (hence the minus sign) because we want to decrease the loss.  
- The learning rate \( η \) scales the step: small \( η \) = small cautious steps; large \( η \) = big leaps that may overshoot.


### Why gradients look like (y − p) in logistic regression (intuitive recap)
- For probabilistic models like logistic regression, the gradient simplifies to sums of per-example residuals times features: contribution  \( ≈ −(y − p) x_j. \)
- (y − p) is the signed error: positive when true label is 1 but predicted probability p is too small, negative when p is too large.  
- Multiplying by \( x_j \) assigns that error to each parameter in proportion to the feature value.


### Types of gradient descent and why you would pick one
1. Batch gradient descent
   - Uses all N training examples to compute the gradient each step.  
   - Stable, smooth steps toward the minimum; expensive per update when N is large.

2. Stochastic gradient descent (SGD)
   - Uses one randomly chosen example per update.  
   - Noisy updates that can escape shallow local minima and converge faster in wall-clock time for large datasets; needs more steps and careful learning-rate scheduling.

3. Mini-batch gradient descent
   - Uses a small batch (e.g., 32, 128 examples) per update.  
   - Tradeoff between stability and speed; common default in deep learning.

---

## Logistic Regression
Logistic regression is a model that turns a weighted sum of input features into a probability for a binary outcome and fits the weights so those probabilities match observed labels. It combines a linear score, a sigmoid mapping to [0,1], a probabilistic loss (negative log-likelihood), and an optimization method (gradient descent) to learn parameters that make predicted probabilities match true labels.


### Model components
- **Linear combination (score)**  
  The model computes a score z for each example using a weighted sum:  
  \[
  z=\beta_0+\beta_1 x_1+\beta_2 x_2+\dots+\beta_p x_p.
  \]  
  The score z is a single real number that represents how strongly the model favors the positive class before converting to a probability.

- **Sigmoid mapping to probability**  
  The score z is converted to a probability p that the label is 1 using the logistic sigmoid:  
  \[
  p=\sigma(z)=\frac{1}{1+e^{-z}}.
  \]  
  The sigmoid makes large positive z correspond to probabilities near 1 and large negative z correspond to probabilities near 0.

- **Decision rule**  
  To predict a class, compare p to a threshold (commonly 0.5). Predict 1 if p≥0.5, otherwise predict 0. The threshold can be moved to trade precision and recall.


### Loss and log-likelihood intuition
- **Per-example probability and fit**  
  If the true label is y∈{0,1}, the model assigns probability p to y=1 and 1−p to y=0. Good fits give high probability to the observed label.

- **Log-likelihood**  
  The log-likelihood for the dataset sums log probabilities the model assigns to the observed labels:  
  \[
  \ell(\beta)=\sum_{i=1}^N \bigl[y^{(i)}\log p^{(i)}+(1-y^{(i)})\log(1-p^{(i)})\bigr].
  \]  
  Maximizing ℓ means choosing β that make observed labels as probable as possible under the model.

- **Loss (negative log-likelihood / cross-entropy)**  
  Minimizing the negative log-likelihood is equivalent to maximizing the log-likelihood:  
  \[
  L(\beta)=-\ell(\beta)=-\sum_{i=1}^N \bigl[y^{(i)}\log p^{(i)}+(1-y^{(i)})\log(1-p^{(i)})\bigr].
  \]  
  The loss punishes confident wrong predictions heavily and rewards confident correct ones.


### Gradient, residuals, and why (y − p) appears
- **Residual as error signal**  
  For example i the residual \( r^{(i)}=y^{(i)}−p^{(i)} \) is the signed error: positive when the model underestimates the probability of y=1, negative when it overestimates.

- **Gradient formula**  
  The gradient of the loss with respect to parameter \( \beta_j \) is the sum of per-example contributions, (the derivative of L with respect to \( \beta_j \)):
  \[
  \frac{\partial L}{\partial \beta_j}=-\sum_{i=1}^N (y^{(i)}-p^{(i)})\,x_j^{(i)}.
  \]  
  Each example contributes \( (y−p) \) scaled by the feature \( x_j \). The intercept uses \( x_0^{(i)}=1 \) so its gradient is the negative sum of residuals.

- **Why the sigmoid cancels**  
  The derivative of the sigmoid is \( p(1−p) \). When differentiating the log-loss, algebra cancels the \( p \) and \( 1−p \) factors, leaving the simple residual \( y−p \). This is why the error signal is intuitive and direct.

- **Intuition of the gradient**  
  If an example has \( y=1 \) but \( p \) small, \( (y−p) \) is large positive and it increases weights of features active in that example. If \( y=0 \) but \( p \) large, \( (y−p) \) is negative and it decreases weights of features active in that example. Summing over all examples aggregates evidence for each parameter.

### Gradient descent updates and interpretation
- **Update rule**  
  Gradient descent changes parameters in small steps opposite to the gradient:  
  \[
  \beta_j \leftarrow \beta_j - \eta\frac{\partial L}{\partial \beta_j}.
  \]  
  Substituting the logistic gradient gives a practical update: the parameter change is proportional to the sum over examples of \( (y−p)×x_j \) scaled by the learning rate \( η \).

- **Intuition of a single update**  
  The update is a weighted average of per-example corrections. Examples where the model is wrong (large |y−p|) and where the feature \( x_j \) is large have the biggest influence on that parameter. The update nudges the model so predicted probabilities move toward true labels.

- **Learning rate role**  
  The learning rate controls how large a correction each step applies. Small learning rates give stable slow progress; large rates can overshoot and diverge.

### Complete numeric walk-through tying everything together
Data with one feature and intercept, three examples:
- x = [0, 1, 2], y = [0, 1, 1].

Initialize \( \beta_0=0 \), \( \beta_1=0 \) so \( z=0 \) for all examples and \( p=0.5 \) for all examples.

Compute the probability predictions, residuals, gradients, and parameter updates step-by-step:

Step 0 Initial predictions
- For all examples, \( z = 0 + 0·x = 0 \rightarrow p = \sigma(0) = 0.5 \).

Step 1 Compute residuals
- Example 1: \( r_1 = y_1 − p_1 = 0 − 0.5 = −0.5 \).
- Example 2: \( r_2 = 1 − 0.5 = 0.5 \).
- Example 3: \( r_3 = 1 − 0.5 = 0.5 \).

Step 2 Compute gradients
- Gradient for intercept \( \beta_0 \):  
  \[
  \frac{\partial L}{\partial \beta_0}=-\sum r_i = -(-0.5+0.5+0.5) = -0.5.
  \]
- Gradient for slope \( \beta_1 \):  
  \[
  \frac{\partial L}{\partial \beta_1}=-\sum r_i x_i = -(-0.5\cdot0 + 0.5\cdot1 + 0.5\cdot2) = -(0 + 0.5 + 1)= -1.5.
  \]

Step 3 Update parameters with learning rate \( \eta=0.1 \)
Since the gradients are negative, we add a positive amount:
- \( \beta_0 \leftarrow 0 − 0.1·(−0.5) = 0.05 \).
- \( \beta_1 \leftarrow 0 − 0.1·(−1.5) = 0.15 \).

Step 4 Recompute z and p to see the effect
- New z values: \( z = [0.05, 0.05+0.15=0.20, 0.05+0.30=0.35] \).
- New p = σ(z) ≈ [0.5125, 0.5498, 0.5866].
- Residuals shrink where the model needed to increase probability and change sign where it previously overestimated.

Step 5 Repeat updates until residuals and loss are small.

This sequence shows how linear combination \(\rightarrow\) sigmoid \(\rightarrow\) residual \(\rightarrow\) gradient \(\rightarrow\) parameter update forms a closed loop that steadily improves predicted probabilities.

---

### Interpretation of learned parameters
- \( \beta_0 \) (intercept) sets the baseline log-odds when all features are zero. Increasing \( \beta_0 \) shifts all predicted probabilities up.
- \( \beta_j \) (feature weight) indicates how much feature \( x_j \) influences the log-odds of the positive class. Specifically,
  \( \beta_j \) is the change in log-odds per unit increase in \( x_j \). Increasing \( \beta_j \) increases \( z \) for examples with large \( x_j \) and thus increases their predicted probability \( p \).

- **Odds interpretation**
  If \( z \) increases by 1, odds = \( p/(1−p) \) are multiplied by \( e^{1} \). If \( \beta_j \) increases by \( \Delta \), the multiplicative change in odds per unit \( x_j \) is \( e^{\Delta} \).


### Practical intuition and checks
- If many residuals are positive, the intercept will increase to raise all probabilities; if residuals are mixed, individual \( \beta_j \) values adjust according to where features signal the class.
- Use mini-batches for large datasets to get stable but efficient updates.
- Monitor loss and predicted probabilities rather than only binary accuracy during training because probabilities provide richer feedback.
- Regularization (adding a penalty to L) shrinks large \( \beta \) values to prevent overfitting and adds simple extra terms to the gradient.

### One-sentence summary
Logistic regression builds a score from features, turns that score into a probability with a sigmoid, measures mismatch with a log-loss that penalizes confident mistakes, and uses gradient descent where the simple residual (y−p) drives parameter updates that make probabilities closer to the true labels.

---

## Linear Regression
Linear regression predicts a **continuous** numeric target as a weighted sum of input features. It chooses weights so predicted values are as close as possible to observed targets according to a chosen loss, most often mean squared error. Linear regression is simple, interpretable, and has clean analytic formulas that make its behaviour easy to understand.

### Model and prediction
- Model form (one example):  
  \[
  \hat{y}^{(i)} = \beta_0 + \beta_1 x_1^{(i)} + \dots + \beta_p x_p^{(i)} = \beta^\top x^{(i)}.
  \]
  - \(\beta\) is the parameter vector including the intercept \(\beta_0\).  
  - \(x^{(i)}\) is the feature vector for example i with a leading 1 if you include the intercept in \(\beta\).  
  - \(\hat{y}^{(i)}\) is the model’s predicted numeric value for example i.
- Intuition: each coefficient \(\beta_j\) is the amount the prediction changes when feature \(x_j\) increases by one unit, holding other features fixed.

### Loss function and its meaning
- Mean Squared Error (MSE) for the dataset:
  \[
  L(\beta) = \frac{1}{2N}\sum_{i=1}^N\bigl(\hat{y}^{(i)}-y^{(i)}\bigr)^2 = \frac{1}{2N}\|X\beta - y\|_2^2.
  \]
  - \(\mathbf{y}^{(i)}\) is the true numeric target for example i.  
  - \(\mathbf{X}\) is the design matrix (N by (p+1)) whose i-th row is \(x^{(i)}\).  
  - Factor 1/2 is conventional so derivatives have a simple 2 that cancels.
- Intuition: the loss averages squared vertical distances between predictions and targets; squaring emphasizes large errors.

- Comparison to logistic regression loss:
  - Linear regression uses squared error for continuous targets, while logistic regression uses negative log-likelihood for binary targets.  
  - Both losses measure how well predictions match observed values but in different contexts (continuous vs. categorical).
  - Why they differ: linear regression assumes Gaussian noise around true values, while logistic regression models probabilities of discrete classes.

### Gradient of the loss and intuition behind each term
- Gradient with respect to \(\beta\) (vector form):
  \[
  \nabla_\beta L(\beta) = \frac{1}{N} X^\top (X\beta - y).
  \]
  - The vector \(X\beta - y\) is the per-example residuals \((\hat{y}^{(i)} - y^{(i)})\).  
  - Multiplying by \(X^\top\) accumulates how each feature correlates with the residuals so each coefficient’s derivative reflects how much its feature contributed to prediction errors.
- Elementwise form for parameter \(\beta_j\):
  \[
  \frac{\partial L}{\partial \beta_j} = \frac{1}{N}\sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)})\,x_j^{(i)}.
  \]
  - Intuition: each example contributes its signed error times the feature value; large positive errors with large \(x_j\) push \(\beta_j\) down, large negative errors push \(\beta_j\) up.

### Optimization by gradient descent
- Update rule (batch gradient descent):
  \[
  \beta \leftarrow \beta - \eta\,\nabla_\beta L(\beta) = \beta - \frac{\eta}{N}X^\top(X\beta-y).
  \]
  - \(\eta\) is the learning rate controlling step size.  
  - Intuition: move \(\beta\) opposite the direction that increases loss; contributions are averages of (prediction error × feature).
- Practical variants:
  - Stochastic gradient descent uses single examples per update for noisy but fast steps.  
  - Mini-batch uses small groups to trade stability and speed.

### Closed form solution normal equations and intuition
- Normal equations (ordinary least squares):
  \[
  \hat{\beta} = (X^\top X)^{-1}X^\top y,
  \]
  valid when \(X^\top X\) is invertible.
- Intuition:
  - \(X^\top X\) captures feature covariance and curvature of the quadratic loss.  
  - \(X^\top y\) captures alignment between features and targets.  
  - Solving balances these to find \(\beta\) that makes the weighted sum of residual-feature products zero, which is the first‑order optimality condition \(\nabla_\beta L(\hat\beta)=0\).  
  - Because the loss is quadratic, this solution is the global minimizer; no iterative method is required if you can compute the inverse reliably.

### Regularization and its formulas
- Ridge (L2) regularized loss:
  \[
  L_{ridge}(\beta)=\frac{1}{2N}\|X\beta-y\|_2^2+\frac{\lambda}{2}\|\beta\|_2^2.
  \]
  - Gradient: \(\nabla L_{ridge}= \frac{1}{N}X^\top(X\beta-y)+\lambda\beta.\)  
  - Closed form: \(\hat\beta=(X^\top X + \lambda N I)^{-1}X^\top y.\)
- Intuition: regularization penalizes large coefficients to reduce overfitting and stabilizes inversion when features are correlated.

### Final intuition summary
- Linear regression fits a straight relationship between features and a numeric target by minimizing squared errors.  
- Formulas split into three interpretable parts: model (linear combination), loss (squared residuals measuring fit), and optimization (gradient or closed-form solve that sets averaged residual-feature correlations to zero).  
- Each matrix or vector in the formulas has a clear statistical meaning: \(X\) encodes inputs, \(y\) encodes outputs, \(X^\top(X\beta−y)\) accumulates how features correlate with errors, and invertible \(X^\top X\) captures the geometry needed to solve for \(\beta\) exactly.