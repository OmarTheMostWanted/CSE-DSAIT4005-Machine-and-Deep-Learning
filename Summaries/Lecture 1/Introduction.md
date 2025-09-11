# Sensor  
- Representation  
- Generalization  
- Feature Space  
- Classification  
- Test object classified as ‘A’  

## Recognition problem  
- What is happening? Where is this?  
- Who is who?  
- What is what?  

## Pattern recognition: find the cat  
- Everyday tasks are deceptively difficult  
- Need to recognize places, situations, objects, people  
- Automatic pattern recognition: learning from examples  

## Example tasks in ML & DL  
- Structure discovery  
- Gender detection  
- Classification  
- Classification + localization  
- Object detection  
- Instance segmentation  
- Feature extraction  
- Text classification  
- Lane finding  
- Fraud detection  
- Image classification  
- Medical diagnosis  
- Music generation  
- Generative networks  
- Unsupervised learning  
- Supervised learning  
- Pattern modeling  
- Image generation  
- Weather forecasting  
- Regression  
- Clustering  
- Game AI  
- Sales growth prediction  
- Customer segmentation  
- Reinforcement learning  
- Recommender systems  
- Market forecasting  
- Skill acquisition  
- Estimating life expectancy  
- Targeted marketing  
- Robot navigation  

## Pattern recognition pipeline  
1. Dataset (measurements)  
2. Feature extraction  
3. Learning  
4. Classifier  

## Classification: What is the label?  
- What is the label of this object? (e.g., cat? green bean?)  
- Labels may come from the input or an external source  
- Ensure all relevant information is available for classification  
- Experts provide labels/targets  

## Feature quality  
- Good features separate classes clearly  
- Poor features lead to confusion and noisy measurements  
- In many problems, experts are unsure which features to choose  

## Overfitting & evaluation  
- Overfitting: model memorizes training data patterns  
- Use independent test data to evaluate generalization  

## This week’s topics  
- Introduction, definitions, learning from examples  
- Classification  
- Bayes rule & Bayes error  
- Misclassification costs  
- Parametric classifiers: quadratic, linear (LDA), nearest mean  
- Non-parametric classifiers: Parzen, k-NN  
- (Logistic regression?)  

## Objects in feature space  
- Represent measurements as a vector  
  

\[X = (x_1, x_2, \dots, x_p)\]

  
- Assume an underlying density \(p(x, y)\) over feature space  

## Classification in feature space  
- Given labeled samples, assign class labels to partition the space  
- Regression: predict real values instead of classes  
- Clustering: discover structure without labels  

## Model output & decision rule  
- Estimate posterior probabilities \(p(y \mid x)\) or fit \(f(x)\)  
- Decision: assign \(x\) to class \(y_1\) if \(p(y_1 \mid x) > p(y_2 \mid x)\)  
- Equivalently compare  
  

\[
  p(y_1 \mid x) - p(y_2 \mid x) > 0,\quad
  \frac{p(y_1 \mid x)}{p(y_2 \mid x)} > 1,\quad
  \log p(y_1 \mid x) - \log p(y_2 \mid x) > 0
  \]

  

## Bayes’ theorem & total probability  


\[
p(y \mid x) = \frac{p(x \mid y)\,p(y)}{p(x)}
\]

  


\[
p(x) = \sum_{i} p(x \mid y_i)\,p(y_i)
\]

  
- Two-class:  
  

\[
  p(x) = p(x \mid y_1)\,p(y_1) + p(x \mid y_2)\,p(y_2)
  \]

  

## Bayes classification steps  
1. Estimate class-conditional densities \(p(x \mid y_i)\)  
2. Multiply by priors \(p(y_i)\)  
3. Compute posteriors \(p(y_i \mid x)\)  
4. Assign to class with highest posterior  

## Classification error & Bayes error  
- Total error:  
  

\[
  p(\text{error}) = \sum_{i} p(\text{error} \mid y_i)\,p(y_i)
  \]

  
- Bayes error: minimum attainable (typically \(>0\))  
- In practice cannot compute (unknown densities, high-dim integrals)  

## Misclassification costs & risk  
- Introduce cost matrix \(\lambda_{ij}\) for assigning \(y_j\) when true class is \(y_i\)  
- Conditional risk:  
  

\[
  R_i(x) = \sum_{j} \lambda_{ji}\,p(y_j \mid x)
  \]

  
- Total risk over region \(L_i\):  
  

\[
  \int_{L_i} R_i(x)\,p(x)\,dx
  \]

  
- Decision rule: choose class minimizing expected risk  
- Two-class comparison:  
  

\[
  \lambda_{21}\,p(y_2 \mid x) \;\gtrless\; \lambda_{12}\,p(y_1 \mid x)
  \]

  

## Example cost scenario  
- Classes: apple, pear, banana  
- Cost matrix defines penalties for misclassification  
- Decision boundaries shift according to costs  

## Thought questions  
- Does better posterior estimation reduce Bayes error?  
- How does estimated classification error compare to Bayes error?  
- What if \(p(y_1) = 2\,p(y_2)\)?  

## Conclusions  
- Machine learning: learn from examples  
- Classification: predict object labels  
- Bayes classifier is optimal  
- Next lecture: quadratic, linear (LDA), nearest mean, more flexible classifiers  
