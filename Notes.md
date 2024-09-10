
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
