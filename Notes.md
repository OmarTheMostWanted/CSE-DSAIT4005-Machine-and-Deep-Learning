# Lecture 1: Introduction

## Objects in feature space
We can interpret the measurements as a vector in a vector space:
    
    x = (x1 , x2 , ..., xp )

This originates, in principle, from a probability density over  the whole feature space

    p(x, y) 

## Classification
- Given labeled data: X
- Assign to each object a class label
- In effect splits the feature space in separate regions

## Class Posterior Probabilities
- The probability of an object x belonging to class y: P(y|x)
  - The sum of the posterior probabilities of all classes is 1: P(y|x) + P(y'|x) = 1      


## Bayes theorem
- P(y|x) = P(x|y)P(y) / P(x)
- class (conditional) distribution P(x|y) is the likelihood of the data given the class
- P(y) is the prior probability of the class
- unconditioned distribution P(x) is the data distribution

## Classification
- Calculate (estimate) the class posterior probabilities
- Multiply by the prior probabilities
- Choose the class with the highest probability

## Decision boundaries
- The boundaries that separate the regions of different classes
- The decision boundaries are defined by the points where the posterior probabilities are equal

## Bayes Error
- The error rate of the Bayes classifier
- The minimum error rate that can be achieved
- The Bayes error is the sum of the probabilities of the regions where the posterior probabilities are not the highest
- P(error) = ∫ P(error|x)P(x)dx or P(error) = ∑ P(error|x)P(x) 
- Its the sum of the probabilities of the regions where the posterior probabilities are not the highest

## Missclassification cost
Introduce a loss that measures the cost of misclassification of an object x as class y when it actually belongs to class y'.