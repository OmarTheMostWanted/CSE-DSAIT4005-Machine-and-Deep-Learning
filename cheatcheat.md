# Normal Distribution Probability Density Function

$ P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} $

# Multivariate Normal Distribution Probability Density Function

$ P(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)} $

# Conditional Probability

$ P(A|B) = \frac{P(A \cap B)}{P(B)} $

# Bayes' Theorem

$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $

# Law of Total Probability

$ P(B) = \sum_{i=1}^{n} P(B|A_i)P(A_i) $

# ML terminology

## Posterior Probability

$ P(\theta|X) $

## Prior Probability

$ P(\theta) $

## Class Conditional Probability

$ P(X|\theta) $

# Sum of Posterior Probability

$ \sum_{i=1}^{n} P(\theta_i|X) = 1 $