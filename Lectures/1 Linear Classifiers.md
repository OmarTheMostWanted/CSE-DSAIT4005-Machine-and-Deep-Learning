# Linear Classifiers
Linear Classifiers is about estimating the class conditional probabilities  $ (P(Y|X)) $ based on $ P(X|Y) $ and $ P(Y) $.

Density estimation is the process of estimating the probability density function of the random variable that generated the data.

## Linear Discriminant
Classifiers that do not do density estimation but directly estimate the class conditional probabilities are called discriminative classifiers.

Describe the decision boundary as a line or a hyperplane.

$$ g(x) = W^T X + W_0 $$

Where $ W $ is the weight vector and $ W_0 $ is the bias term.

Classify $ x $ to $ \begin{cases} w_1 & \text{if } W^T X + W_0 \ge 0 \\ w_2 & \text{if } W^T X + W_0 \lt 0 \end{cases} $

In the most general sense, this is called linear discriminant analysis.

- Classifier is a linear function of the input features.
- The classification depends if the weighted sum of the input features is greater or less than a threshold.

Incoporating the bias term into the weight vector, we can write the equation as:

$$ g(x) = W^T X > 0 $$

Redefine the feature vector as $ X = \begin{bmatrix} X \\ 1 \end{bmatrix} $ and the weight vector as $ W = \begin{bmatrix} W^T \\ W_0 \end{bmatrix} $.

$$ g(x) = \begin{bmatrix} W^T & W_0 \end{bmatrix} * \begin{bmatrix} X \\ 1 \end{bmatrix} = w^T x $$

### Examples:
- Perceptron: Works by updating the weights based on the misclassified points.
- Fisher classifier: Maximizes the distance between the means of the classes and minimizes the variance within the classes.
- Logistic classifier: Uses the logistic function to estimate the class conditional probabilities.
- Least squares: Minimizes the squared error between the predicted and the actual values.

### Bias-Vfariance Tradeoff
The bias-variance tradeoff is the tradeoff between the bias of the estimator and the variance of the estimator.

## Nearest Mean
How to find $ W $ and $ W_0 $?

- Previously, we assumed a Gaussian distribution per class. When we assume $ \Sigma = \sigma^2 I $ (where $ \Sigma $ is the covariance matrix and $ \sigma^2 I $ is a scalar multiple of the identity matrix), we can find:

  $$ w = \mu_1 - \mu_2 $$

  - Here, $ \mu_1 $ and $ \mu_2 $ are the mean vectors for the two classes.
  - $ w $ is the vector that points in the direction of the greatest separation between the classes.

  $$ w_0 = \frac{1}{2} (\mu_1 + \mu_2) $$

  - $ w_0 $ represents the midpoint between the two means, acting as the threshold.

In this context, $ w $ forms the weight vector, and $ w_0 $ is the bias term.

## The Perceptron Algorithm
The perceptron algorithm works by updating the weights based on the misclassified points. Here’s how it goes:

1. **Assume Linearly Separable Classes**: Assume that the classes are linearly separable, meaning there exists a hyperplane that perfectly separates the classes. This means there is an optimal weight vector $ W^* $ that separates the classes:
   
   $$ w^{*T} x > 0 \quad \forall x \in C_1 (y = +1) $$
   $$ w^{*T} x < 0 \quad \forall x \in C_2 (y = -1) $$

   - $ C_1 $ and $ C_2 $ represent the two classes.
   - $ w^{*T} x $ denotes the dot product of the weight vector $ w^* $ and the input vector $ x $.
   - $ y $ is the class label (+1 or -1).

2. **Define the Perceptron Error/Loss Function**:
   
   The perceptron aims to minimize the following error function:
   
   $$ J(W) = \sum_{\text{misclassified } x_i} -y_i w^T x_i $$

   - $ J(W) $ is the error function.
   - The summation runs over all misclassified points $ x_i $.
   - $ y_i $ is the actual class label of $ x_i $.
   - $ w^T x_i $ denotes the dot product of the weight vector $ w $ and the misclassified point $ x_i $.

### Perceptron Error Function

The error function, often called the cost or loss function, measures how well the perceptron is classifying the training data. It helps the algorithm understand how far it is from correctly classifying all the points.

For the perceptron algorithm, the error function is defined as:

$$
J(W) = \sum_{\text{misclassified } x_i} -y_i w^T x_i
$$

Here's what each part of this formula means:

- **J(W)**: This is the error function itself, which depends on the current weight vector \( W \).

- **Summation over misclassified \( x_i \)**: This part indicates that we sum the errors for all the misclassified points \( x_i \).

- **\( y_i \)**: This is the actual label of the point \( x_i \). It can be either +1 or -1, indicating which class the point belongs to.

- **\( w^T x_i \)**: This is the dot product of the weight vector \( w \) and the feature vector \( x_i \). It represents the decision boundary of the perceptron.

- **\(-y_i w^T x_i \)**: This term calculates the error for a misclassified point. If a point is correctly classified, this value would be positive. However, since we are summing errors for misclassified points, these values are negative, indicating that the points are on the wrong side of the decision boundary.

### Why It Matters

The perceptron updates its weights by minimizing this error function. The lower the value of \( J(W) \), the better the perceptron is doing at correctly classifying the training data. During each iteration, the algorithm adjusts the weights to reduce this error, moving closer to the optimal decision boundary.

### Cost function optimization
Assume a cost function $ J(\theta) $ that we want to minimize.

#### One way to set the derivative to zero and solve
$$ \frac{\partial J(\theta)}{\partial \theta} = 0 $$

This is very hard to solve for complex functions.

#### Gradient Descent
Follow the gradient of the cost function to reach a (local) minimum.

$$ \theta_{t+1} = \theta_t - \rho \frac{\partial J(\theta)}{\partial \theta} $$

Where $ \rho $ is the learning rate.
$ \theta $ is the parameter vector.


Gradient Descent is an optimization algorithm used to minimize the cost function in various machine learning algorithms. The primary goal is to find the parameters (weights) that minimize the cost function.

##### Key Concepts

1. **Cost Function**: A function that measures how well the model's predictions match the actual data. Commonly used cost functions include Mean Squared Error (MSE) for regression and Cross-Entropy Loss for classification.

2. **Gradient**: The gradient of the cost function with respect to the model's parameters indicates the direction and rate of the steepest ascent. Since we aim to minimize the cost function, we move in the opposite direction of the gradient.

3. **Learning Rate**: A hyperparameter that determines the step size at each iteration while moving towards a minimum of the cost function. Too large a learning rate can cause overshooting, while too small a learning rate can slow down convergence.

##### Types of Gradient Descent

1. **Batch Gradient Descent**: Calculates the gradient using the entire dataset. It provides a stable and precise direction but can be slow and computationally expensive for large datasets.
   
   $$ w := w - \eta \nabla J(w) $$

   Where:
   - \( w \) is the vector of parameters.
   - \( \eta \) is the learning rate.
   - \( \nabla J(w) \) is the gradient of the cost function.

2. **Stochastic Gradient Descent (SGD)**: Updates the parameters using only one training example at a time. It is much faster and can handle large datasets efficiently but introduces more noise in the update steps.
   
   $$ w := w - \eta \nabla J(w; x^{(i)}, y^{(i)}) $$

   Where:
   - \( (x^{(i)}, y^{(i)}) \) is the \( i \)-th training example.

3. **Mini-Batch Gradient Descent**: A compromise between Batch Gradient Descent and SGD, it divides the dataset into small batches and performs an update for each batch. It balances speed and precision.
   
   $$ w := w - \eta \nabla J(w; x^{(i:i+n)}, y^{(i:i+n)}) $$

   Where:
   - \( (x^{(i:i+n)}, y^{(i:i+n)}) \) are the mini-batch examples.

##### Steps in Gradient Descent

1. **Initialize Weights**: Begin with random weights or zero weights.
2. **Compute Gradient**: Calculate the gradient of the cost function with respect to the weights.
3. **Update Weights**: Adjust the weights in the opposite direction of the gradient.
4. **Repeat**: Iterate the process until convergence or for a predefined number of epochs.

##### Example

For a simple linear regression problem:

1. **Cost Function** (Mean Squared Error):
   
   $$ J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})^2 $$

   Where:
   - \( h_w(x^{(i)}) = w x^{(i)} + b \) is the hypothesis.
   - \( m \) is the number of training examples.

2. **Gradient Descent Update Rules**:
   
   $$ w := w - \eta \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)}) x^{(i)} $$
   $$ b := b - \eta \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)}) $$

By following these steps, we iteratively adjust \( w \) and \( b \) until we find the values that minimize the cost function.

### The Perceptron
Just a simple linear classifier that is trained incrementall or in batches (applicable to very large datasets)
When the data is separable, the perceptron converges to a solution.
When the data is not separable, the perceptron will not converge and will run indefinitely.

Perceptron is the basis for many other algorithms like neural networks.

