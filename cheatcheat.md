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


# Linear Discriminant Analysis (LDA)

## Intuition
Linear Discriminant Analysis (LDA) is a technique used in statistics and machine learning to find a linear combination of features that best separates two or more classes of objects or events. The main goal of LDA is to project the data onto a lower-dimensional space with good class-separability to avoid overfitting and reduce computational costs.

## Simple Example
Imagine you have a dataset with two features (e.g., height and weight) and two classes (e.g., male and female). LDA will find a line (or hyperplane in higher dimensions) that best separates the two classes. When a new data point comes in, you can project it onto this line and classify it based on which side of the line it falls.

## Formulas and Terms

1. **Within-class scatter matrix $(S_W)$**:
   $$ S_W = \sum_{i=1}^{c} S_i $$
   where $ S_i $ is the scatter matrix for class $ i $.

    - **Scatter matrix (S_i)**: Measures the spread (variance) of the data points within each class. For class $ i $, it is calculated as:
      $$ S_i = \sum_{x \in D_i} (x - \mu_i)(x - \mu_i)^T $$
      where $ x $ is a data point in class $ i $, $ \mu_i $ is the mean vector of class $ i $, and $ D_i $ is the set of data points in class $ i $.

2. **Between-class scatter matrix (S_B)**:
   $$ S_B = \sum_{i=1}^{c} N_i (\mu_i - \mu)(\mu_i - \mu)^T $$
   where $ N_i $ is the number of samples in class $ i $, $ \mu_i $ is the mean vector of class $ i $, and $ \mu $ is the overall mean vector.

    - **Between-class scatter matrix (S_B)**: Measures the spread (variance) of the mean vectors of the classes. It captures how much the class means differ from the overall mean. The larger the between-class scatter, the better the classes are separated.

3. **LDA projection**:
   $$ w = S_W^{-1} (\mu_1 - \mu_2) $$
   where $ w $ is the vector that defines the linear discriminant.

## Parameters
- **Classes (c)**: The number of distinct classes in the dataset.
- **Features**: The number of features in the dataset.
- **Mean vectors (\mu_i)**: The mean of each feature for each class.
- **Scatter matrices (S_W and S_B)**: Matrices that measure the spread of the data within and between classes.

## Steps in LDA
1. Compute the mean vectors for each class.
2. Compute the within-class scatter matrix $ S_W $ and the between-class scatter matrix $ S_B $.
3. Compute the eigenvalues and eigenvectors for the matrix $ S_W^{-1} S_B $.
4. Select the eigenvectors corresponding to the largest eigenvalues to form a matrix $ W $.
5. Project the data onto the new subspace using $ Y = XW $.

## Conclusion
LDA is a powerful technique for dimensionality reduction and classification. It works by finding the linear combinations of features that best separate the classes, making it easier to classify new data points.

