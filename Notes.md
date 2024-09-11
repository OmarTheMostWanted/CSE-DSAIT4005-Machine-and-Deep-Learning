
Sure! Here's a cheat sheet for some of the most common mathematical symbols:

### Arithmetic Symbols
- **Addition**: \( + \)
- **Subtraction**: \( - \)
- **Multiplication**: \( \times \) or \( \cdot \)
- **Division**: \( \div \) or \( / \)
- **Equals**: \( = \)
- **Not equal**: \( \neq \)

### Algebraic Symbols
- **Variable**: \( x, y, z \)
- **Exponentiation**: \( x^n \)
- **Square root**: \( \sqrt{x} \)
- **nth root**: \( \sqrt[n]{x} \)
- **Absolute value**: \( |x| \)

### Calculus Symbols
- **Derivative**: \( \frac{dy}{dx} \)
- **Integral**: \( \int \)
- **Partial derivative**: \( \frac{\partial y}{\partial x} \)
- **Limit**: \( \lim_{x \to a} \)

### Set Theory Symbols
- **Element of**: \( \in \)
- **Not an element of**: \( \notin \)
- **Subset**: \( \subset \)
- **Union**: \( \cup \)
- **Intersection**: \( \cap \)

### Logic Symbols
- **And**: \( \land \)
- **Or**: \( \lor \)
- **Not**: \( \neg \)
- **Implies**: \( \implies \)
- **If and only if**: \( \iff \)

### Common Constants
- **Pi**: \( \pi \)
- **Euler's number**: \( e \)
- **Imaginary unit**: \( i \)

### Miscellaneous Symbols
- **Infinity**: \( \infty \)
- **Summation**: \( \sum \)
- **Product**: \( \prod \)
- **Approximately equal**: \( \approx \)
- **Proportional to**: \( \propto \)
- **Degree**: \( ^\circ \)
- **Subscript**: \( x_1 \)


# Lecture 1: Introduction

## Objects in feature space
We can interpret the measurements as a vector in a vector space: 
$ x = (x_1, x_2, ..., x_n) $
This originates, in principle, from a probability density over  the whole feature space
$ p(x, Y) $

## Classification
- Given labeled data: X
- Assign to each object a class label
- In effect splits the feature space in separate regions

## Class Posterior Probabilities
- The probability of an object x belonging to class y: $ P(y|x) $
- The sum of the posterior probabilities of all classes is 1: $ \sum_{i=1}^{C} P(y_i|x) = 1 $ where C is the number of classes   

## Conditional Probability
- The probability of an event given that another event has occurred
- $ P(y|x) = \frac{P(x \land y)}{P(x)} $

## Bayes theorem
- $ P(y|x) = \frac{P(x|y)P(y)}{P(x)} $
- Class (conditional) distribution $ P(x|y) $ is the likelihood of the data given the class
- $ P(y) $ is the class prior probability
- unconditioned distribution $ P(x) $ is the data distribution

## Law of total probability
- The probability of an object x is the sum of the probabilities of x given all classes times the prior probabilities of the classes
- $ P(x) = \sum_{i=1}^{C} P(x|y_i)P(y_i) $
- For two classes: $ P(x) = P(x|y_1)P(y_1) + P(x|y_2)P(y_2) $

## Classification
- Calculate (estimate) the class posterior probabilities
- Multiply by the prior probabilities
- Choose the class with the highest probability
- $ y = argmax_{y_i} P(y_i|x) $ or $ y = argmax_{y_i} P(x|y_i)P(y_i) $

## Decision boundaries
- The boundaries that separate the regions of different classes
- The decision boundaries are defined by the points where the posterior probabilities are equal
- For example in a 2 class problem, the decision boundary is the set of points where $ P(y_1|x) = P(y_2|x) $

## Classification error
- The probability of misclassification
- $ P(error) = \sum_{i=1}^{C} P(error|y_i)P(y_i) $ (The sum of the probabilities of the regions where the posterior probabilities are not the highest)

## Bayes Error $ \epsilon * $
- Bayes’ error is the minimum attainable error.
- In practice, we do not have the true class distribution, so we cannot calculate the Bayes error.
- The Bayes’ error does not depend on the classification rule that you apply, but on the distribution of the data.

In general you can not compute the Bayes’ error:
    - you don’t know the true class conditional probabilities
    - the (high) dimensional integrals are very complicated.

## Miss classification cost
Sometimes the cost of misclassification is not the same for all classes. In this case, the decision rule should be adapted to minimize the cost.

### Loss $ \lambda_{ij} $
- The loss of classifying an object of class $ w_i $ as class $ w_j $

### Conditional risk
Conditional risk, also known as conditional expected risk, is the expected loss given that a specific condition or event occurs. In the context of classification, it is the expected misclassification cost given that the input falls into a particular region or class.


- The conditional risk of classifying an object x as class $ w_i $ is the sum of the losses of classifying x as class $ w_j $ times the posterior probability of class $ w_j $:
    - $ l^i(x) = \sum_{j=1}^{C} \lambda_{ij} P(w_j|x) $

### Average risk over a region
Average risk, also known as expected risk, is the overall expected loss across all possible conditions or events. It is the weighted average of the conditional risks, where the weights are the probabilities of the respective conditions or events.

The average risk of classifying an object x in a region R as class $ w_i $ is the sum of the conditional risks of classifying x as class $ w_i $ times the probability of x in the region R:

- $ r^i = \int_\Omega l^i(x) p(x) dx $
    - $ \Omega_i = \{ x \in \Omega | y(x) = w_i \} $
    - $ \Omega_i $ here is the region or subset of the feature space where class $ w_i $ is the most probable class.

### Overall risk (total risk)
The overall risk, also known as the total risk or global risk, is the expected loss across the entire input space, considering all possible classifications and their associated costs. It is the sum of the average risks over all regions or classes, weighted by the probabilities of those regions or classes.


The overall risk can be expressed as: 
$R = \sum_{i=1}^C r^i = \sum_{i=1}^C \int_\Omega l^i(x) p(x) dx $

### Minimizing total risk
We minimize the risk when we define the regions $ \Omega_i $ are chosen such that each of the integrals is minimized.


So $ x \in \Omega_i $ if: 

$ \sum_{j=1}^{C} \lambda_{ji} P(w_j|x) <= \sum_{j=1}^{C} \lambda_{jk} P(w_j|x) for k = 1,2,..C $


#### Minimum risk two class example:
When you pridict class $ w_1 $ for an object from class $ w_1 $ usually $\lambda_{11} = 0$.

- Therefore when we have two classes we comapre:

$ \sum_{j=1}^2 \lambda_{j1} P(w_j|x) => \lambda_{11} P(w_1|x) + \lambda_{21} P(w_2|x)  => 0 + \lambda_{21} P(w_2|x) $

and

$ \sum_{j=1}^2 \lambda_{j2} P(w_j|x) => \lambda_{12} P(w_1|x) + \lambda_{22} P(w_2|x)  => \lambda_{12} P(w_1|x) + 0 $

and choose the class with the minimum value.

# Lecture 2: Density  based classification
## Bayes Classifier
In many cases the posterior is hard to estimate, often a functional (parametric) form is assumed. Using bayes theorem we can rewrite the posterior as:
$ P(y|x) = \frac{P(x|y)P(y)}{P(x)} $

Where:
- $ P(y|x) $ is the posterior
- $ P(x|y) $ is the likelihood or class conditional
- $ P(y) $ is the  class prior
- $ P(x) $ is the unconditioned distribution
    - which can be estimated by the sum of the likelihoods times the priors:
    - $ P\^(x) = \sum_{i=1}^{C} P\^(x|y_i)P\^(y_i) $


## Estimating the class conditional probability
- The model for the class condtional probability is now the crucial choice.

### Very Common model: Gaussian distribution
The Gaussian distribution is a continuous probability distribution that is symmetric around its mean, showing that data near the mean are more frequent in occurrence than data far from the mean. The Gaussian distribution is defined by the probability density function:

$ p(x) = \frac{1}{(2\pi)^{p/2} \sqrt{\det(\Sigma)}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right) $

Where:
- $ \mu $ is the mean vector
- $ \Sigma $ is the covariance matrix the elliptical shape of the distribution
- $ p $ is the number of dimensions
- $ \det(\Sigma) $ is the determinant of the covariance matrix
    - The determinant of a matrix is a value that can be computed from its elements. It is a measure of the space spanned by the matrix's columns.
    - $ det(\Sigma) = \prod_{i=1}^{p} \lambda_i $ where $ \lambda_i $ are the eigenvalues of the covariance matrix
- $ \exp $ is the exponential function

#### 2D Gaussian distribution
$$ \mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix} $$

$$ \Sigma = \begin{bmatrix} \sigma_1^2 & \rho \sigma_1 \sigma_2 \\ \rho \sigma_1 \sigma_2 & \sigma_2^2 \end{bmatrix} $$

where:
- $ \sigma_1^2 $ and $ \sigma_2^2 $ are the variances of the two dimensions
- $ \rho $ is the correlation coefficient between the two dimensions

$ p(x) = \frac{1}{\sqrt{2\pi det(\Sigma)}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right) $



#### Plug-in Gaussian Distribution
Now we use the Gaussian distribution for each class:

$ p\^(x|y) = \frac{1}{\sqrt{2\pi^p det(\Sigma_y\^)}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_y\^)^\top \Sigma_y\^(\mathbf{x} - \boldsymbol{\mu}_y\^)\right) $

We have to estimate the parameters $ \mu_y $ and $ \Sigma_y $ for each class using some training data.

$ \mu = \frac{1}{N} \sum_{i=1}^{N} x_i $

$ \Sigma = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu) (x_i - \mu)^\top $   <- outer product

#### The Two-Class case
- Define the discriminant:
$ f(x) = \log P(y_1|x) - \log P(y_2|x) $ it means that $ f(x) = 0 $ is the decision boundary

- Rewrite the discriminant:
$ f(x) = x^\top W x + w^\top x + w_0 $

where:
- W is the matrix of the quadratic terms of the discriminant function
- w is the vector of the linear terms of the discriminant function
- w_0 is the bias term

this is called a quadratic discriminant function of x

##### Rewriting the Discriminant Function

1. **Original Discriminant Function**:
   $$ f(x) = \log p(y_1|x) - \log p(y_2|x) $$

   This function compares the logarithms of the posterior probabilities of the two classes $ ( y_1 ) and ( y_2 ) $. 

2. **Gaussian Assumption**:
   If we assume that the class-conditional densities $ ( p(x|y_1) ) and ( p(x|y_2) ) $ are Gaussian, we can express these probabilities in terms of their means and covariances. For a Gaussian distribution, the log-probability can be written as:
   $$ \log p(x|y) = -\frac{1}{2} (x - \mu_y)^T \Sigma_y^{-1} (x - \mu_y) - \frac{1}{2} \log |\Sigma_y| + \text{constant} $$

3. **Simplifying the Difference**:
   When we take the difference $ ( \log p(y_1|x) - \log p(y_2|x)) $, the constant terms cancel out, leaving us with:
   $$ f(x) = -\frac{1}{2} (x - \mu_1)^T \Sigma_1^{-1} (x - \mu_1) + \frac{1}{2} (x - \mu_2)^T \Sigma_2^{-1} (x - \mu_2) $$

4. **Quadratic Form**:
   This expression can be rewritten in a quadratic form:
   $$ f(x) = x^T W x + w^T x + w_0 $$
   where:
   - \( W \) is a matrix that combines the inverse covariances \( \Sigma_1^{-1} \) and \( \Sigma_2^{-1} \),
        - $ W = \frac{1}{2} (\Sigma_1^{-1} - \Sigma_2^{-1}) $
   - \( w \) is a vector that combines the means \( \mu_1 \) and \( \mu_2 \),
        - $ w = \Sigma_1^{-1} \mu_1 - \Sigma_2^{-1} \mu_2 $
   - \( w_0 \) is a scalar term that includes the log-determinants of the covariances and other constants.
        - $ w_0 = -\frac{1}{2} \mu_1^T \Sigma_1^{-1} \mu_1 + \frac{1}{2} \mu_2^T \Sigma_2^{-1} \mu_2 - \frac{1}{2} \log \frac{|\Sigma_1|}{|\Sigma_2|} + \log \frac{P(y_1)}{P(y_2)} $

The key idea is that by assuming Gaussian distributions for the class-conditional densities, we can express the discriminant function as a quadratic function of \( x \). This allows for more complex decision boundaries than a linear classifier.


#### Class Posterior probability for the Gaussian case:
Combining:

$ p\^(x|y) = \frac{1}{\sqrt{2\pi^p det(\Sigma_y\^)}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_y\^)^\top \Sigma_y\^(\mathbf{x} - \boldsymbol{\mu}_y\^)\right) $

$ p(y|x) = \frac{p\^(x|y)P(y)}{p\^(x)} $

We can derive for class $ y_i $ the $ log(p(y|x)): $

$ log (p\^(y_i|x)) = -\frac{p}{2} \log(2\pi) - \frac{1}{2} \log(det(\Sigma_i)) - \frac{1}{2} (x - \mu_i)^\top \Sigma_i^{-1} (x - \mu_i) + \log(P(y_i)) $

PS: We use log to avoid numerical problems with numbers very close to zero.

#### Estimating the covariance matrix:
- if one of the variances is zero, the covariance matrix is singular and the inverse does not exist.
- We can estimate the covariance matrix by the sample covariance matrix:
    - $ \Sigma_k = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu) (x_i - \mu)^\top $

for each class k.

- When there is insufficient data, this covariance matrix can not be inverted. 

- Alternatively, we can use a diagonal covariance matrix:
    $ \Sigma_k = \frac{1}{C} \sum_{i=1}^{C} \Sigma_k $


#### No Estimated Covariance Matrix
- In some cases even a full average of the covariance matrices is not possible.
- As simplification one could assume that all features have the same variance:
    - $ \Sigma_k = \sigma^2 I $ where I is the identity matrix

The decision rule is then:
- $ g_i(x) = -\frac{1}{\sigma^2} (\frac{1}{2} \mu_i^\top \mu_i - \mu_i^\top x) + \log(P(y_i)) $

 SZ
- Define the discriminant:
$ f(x) = \log P(y_1|x) - \log P(y_2|x) $ it means that $ f(x) = 0 $ is the decision boundary

We get 
$ f(x) = w^\top x + w_0 $

where:
- $ w = \mu_1 - \mu_2 $ 
- $ w_0 = \frac{1}{2} \mu_1^\top \mu_1 - \frac{1}{2} \mu_2^\top \mu_2 + \log \frac{P(y_1)}{P(y_2)} $

A linear classifier that uses the difference of the class means as the discriminant function is called the nearest mean classifier.
