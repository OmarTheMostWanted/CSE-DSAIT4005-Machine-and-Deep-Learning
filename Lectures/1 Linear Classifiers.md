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

- **J(W)**: This is the error function itself, which depends on the current weight vector $ W $.

- **Summation over misclassified $ x_i $**: This part indicates that we sum the errors for all the misclassified points $ x_i $.

- **$ y_i $**: This is the actual label of the point $ x_i $. It can be either +1 or -1, indicating which class the point belongs to.

- **$ w^T x_i $**: This is the dot product of the weight vector $ w $ and the feature vector $ x_i $. It represents the decision boundary of the perceptron.

- **$-y_i w^T x_i $**: This term calculates the error for a misclassified point. If a point is correctly classified, this value would be positive. However, since we are summing errors for misclassified points, these values are negative, indicating that the points are on the wrong side of the decision boundary.

### Why It Matters

The perceptron updates its weights by minimizing this error function. The lower the value of $ J(W) $, the better the perceptron is doing at correctly classifying the training data. During each iteration, the algorithm adjusts the weights to reduce this error, moving closer to the optimal decision boundary.

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
   - $ w $ is the vector of parameters.
   - $ \eta $ is the learning rate.
   - $ \nabla J(w) $ is the gradient of the cost function.

2. **Stochastic Gradient Descent (SGD)**: Updates the parameters using only one training example at a time. It is much faster and can handle large datasets efficiently but introduces more noise in the update steps.
   
   $$ w := w - \eta \nabla J(w; x^{(i)}, y^{(i)}) $$

   Where:
   - $ (x^{(i)}, y^{(i)}) $ is the $ i $-th training example.

3. **Mini-Batch Gradient Descent**: A compromise between Batch Gradient Descent and SGD, it divides the dataset into small batches and performs an update for each batch. It balances speed and precision.
   
   $$ w := w - \eta \nabla J(w; x^{(i:i+n)}, y^{(i:i+n)}) $$

   Where:
   - $ (x^{(i:i+n)}, y^{(i:i+n)}) $ are the mini-batch examples.

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
   - $ h_w(x^{(i)}) = w x^{(i)} + b $ is the hypothesis.
   - $ m $ is the number of training examples.

2. **Gradient Descent Update Rules**:
   
   $$ w := w - \eta \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)}) x^{(i)} $$
   $$ b := b - \eta \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)}) $$

By following these steps, we iteratively adjust $ w $ and $ b $ until we find the values that minimize the cost function.

### The Perceptron
Just a simple linear classifier that is trained incrementall or in batches (applicable to very large datasets)
When the data is separable, the perceptron converges to a solution.
When the data is not separable, the perceptron will not converge and will run indefinitely.

Perceptron is the basis for many other algorithms like neural networks.


### Python Implementation
```python
import os
import random
from typing import List, Tuple
import matplotlib.pyplot as plt

# Function to generate training and test data with distinct classes
def generate_data(num_samples) -> List[Tuple[float, float, int]]:
    """
    Generates a dataset with two distinct classes.

    Parameters:
    num_samples (int): The total number of samples to generate. Half will belong to class 1 and half to class -1.

    Returns:
    list: A list of tuples where each tuple contains two features (x1, x2) and a class label (1 or -1).
    """
    data = []
    for _ in range(num_samples // 2):
        x1 = random.uniform(-1, 10)
        x2 = random.uniform(-1, 10)
        data.append((x1, x2, 1))  # Class 1
        
    for _ in range(num_samples // 2):
        x1 = random.uniform(-10, 1)
        x2 = random.uniform(-10, 1)
        data.append((x1, x2, -1))  # Class -1
        
    return data

# Function to initialize weights
def initialize_weights() -> Tuple[List[float], float]:
    """
    Initializes the weights and bias for the perceptron.

    Returns:
    tuple: A tuple containing a list of two weights and a bias term.
    """
    return [random.uniform(-1, 1), random.uniform(-1, 1)], random.uniform(-1, 1)

# Function to predict the class label
def predict(weights, bias, x1, x2) -> int:
    """
    Predicts the class label for a given input using the perceptron model.

    Parameters:
    weights (list): The weights of the perceptron.
    bias (float): The bias term of the perceptron.
    x1 (float): The first feature of the input.
    x2 (float): The second feature of the input.

    Returns:
    int: The predicted class label (1 or -1).
    """
    return 1 if weights[0] * x1 + weights[1] * x2 + bias > 0 else -1

# Function to update weights and bias by gradient descent
def update_weights(weights, bias, x1, x2, label, learning_rate) -> Tuple[List[float], float]:
    """
    Updates the weights and bias of the perceptron based on the prediction error.

    Parameters:
    weights (list): The current weights of the perceptron.
    bias (float): The current bias term of the perceptron.
    x1 (float): The first feature of the input.
    x2 (float): The second feature of the input.
    label (int): The true class label of the input.
    learning_rate (float): The learning rate for weight updates.

    Returns:
    tuple: The updated weights and bias.
    """
    weights[0] += learning_rate * label * x1
    weights[1] += learning_rate * label * x2
    bias += learning_rate * label
    return weights, bias

# Function to train the perceptron
def train_perceptron(data, learning_rate, epochs) -> Tuple[List[float], float]:
    """
    Trains the perceptron model using the provided training data.

    Parameters:
    data (list): The training data, a list of tuples containing features and class labels.
    learning_rate (float): The learning rate for weight updates.
    epochs (int): The number of times to iterate over the training data.

    Returns:
    tuple: The trained weights and bias.
    """
    weights, bias = initialize_weights()
    for _ in range(epochs):
        for x1, x2, label in data:
            prediction = predict(weights, bias, x1, x2)
            if prediction != label:
                weights, bias = update_weights(weights, bias, x1, x2, label, learning_rate)
    return weights, bias

# Function to test the perceptron
def test_perceptron(data, weights, bias):
    """
    Tests the perceptron model using the provided test data.

    Parameters:
    data (list): The test data, a list of tuples containing features and class labels.
    weights (list): The weights of the trained perceptron.
    bias (float): The bias term of the trained perceptron.

    Returns:
    float: The accuracy of the perceptron on the test data.
    """
    correct_predictions = 0
    for x1, x2, label in data:
        prediction = predict(weights, bias, x1, x2)
        if prediction == label:
            correct_predictions += 1
    accuracy = correct_predictions / len(data)
    return accuracy

# Function to plot the data and decision boundary
def plot_decision_boundary(data, weights, bias):
    """
    Plots the data points and the decision boundary of the perceptron.

    Parameters:
    data (list): The data points to plot, a list of tuples containing features and class labels.
    weights (list): The weights of the trained perceptron.
    bias (float): The bias term of the trained perceptron.
    """
    plt.figure(figsize=(10, 6))

    for x1, x2, label in data:
        if label == 1:
            plt.scatter(x1, x2, color='blue')
        else:
            plt.scatter(x1, x2, color='red')

    # Decision boundary: w1 * x1 + w2 * x2 + b = 0
    x_values = [min(x[0] for x in data), max(x[0] for x in data)]
    y_values = [-(weights[0] * x + bias) / weights[1] for x in x_values]
    
    plt.plot(x_values, y_values, 'k--')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/perceptron_linear_classifier.png')
    plt.close()  # Close the plot to free up memory

# Main function
def main():
    """
    The main function to execute the perceptron training and testing process.
    """
    num_samples = 100
    learning_rate = 0.01
    epochs = 1000

    # Generate training and test data
    training_data = generate_data(num_samples)
    test_data = generate_data(num_samples)

    # Train the perceptron
    weights, bias = train_perceptron(training_data, learning_rate, epochs)

    # Test the perceptron
    accuracy = test_perceptron(test_data, weights, bias)

    print(f"Trained weights: {weights}")
    print(f"Trained bias: {bias}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Plot the data and decision boundary
    plot_decision_boundary(test_data, weights, bias)

if __name__ == "__main__":
    main()
```

## Fisher Linear Discriminant

The aim is to find the weights that maximize the separability between classes. This is done by maximizing the Fisher criterion, which involves projecting the data into a lower-dimensional space to enhance class separation.

### Projection into Lower Dimensions

Intuitively, projecting data into a lower dimension means finding a line (or hyperplane in higher dimensions) where the data points of different classes are most separated.

- **Why Project?**: Imagine two classes of points in 2D space. If you plot them, they might be overlapping in some areas. By projecting onto a line where the separation is maximized, you can make the classes more distinct along that line.
- **Does it Lose Data?**: While you are reducing dimensions, you aren't necessarily losing the important information about class separation. Instead, you're focusing on the dimensions that matter most for distinguishing between classes.

### Fisher Criterion

The Fisher criterion is the ratio of between-class variance to within-class variance. It’s used to find the optimal linear discriminant function for separating two classes:
- A higher Fisher criterion value indicates better class separation.
- A lower Fisher criterion value indicates poor class separation.

Mathematically:
$$ J_F = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{within}}} = \frac{|\mu_A - \mu_B|^2}{\sigma_A^2 + \sigma_B^2} $$

Where:
- $ J_F $ is the Fisher criterion.
- $ \mu_A $ and $ \mu_B $ are the means of classes A and B.
- $ \sigma_A^2 $ and $ \sigma_B^2 $ are the variances of classes A and B.

### Optimization

Along the weight vector $ W $, the Fisher criterion is maximized when the means are far apart and the variances are small:
$$ J_F = \frac{|W^T \mu_A - W^T \mu_B|^2}{W^T \Sigma_A W + W^T \Sigma_B W} $$

Where:
- $ W $ is the weight vector.
- $ \Sigma_A $ and $ \Sigma_B $ are the covariance matrices of classes A and B.

To optimize the Fisher criterion:
$$ J_F = \frac{W^T \Sigma_{\text{between}} W}{W^T \Sigma_{\text{within}} W} $$

Take the derivative of the Fisher criterion with respect to the weight vector and set it to zero to find the optimal weight vector:
$$ W = \Sigma_{\text{within}}^{-1} (\mu_A - \mu_B) $$

Thus, the classifier becomes:
$$ g(x) = x^T W + W_0 = x^T \Sigma_{\text{W}}^{-1} (\mu_A - \mu_B) + W_0 $$


### Understanding the Classifier Formula

The formula given is:
$$ g(x) = x^T W + W_0 = x^T \Sigma_{\text{W}}^{-1} (\mu_A - \mu_B) + W_0 $$

#### Components Explained

1. **$ g(x) $**:
   - This is the output of the classifier for the input $ x $. It helps determine which class $ x $ belongs to.

2. **$ x^T W $**:
   - $ x $ is the input feature vector (in column form).
   - $ x^T $ is the transpose of $ x $, making it a row vector.
   - $ W $ is the weight vector that defines the direction along which the data is projected.
   - $ x^T W $ is the dot product of $ x $ and $ W $, resulting in a single scalar value. This value represents the projection of $ x $ onto the weight vector $ W $.

3. **$ W_0 $**:
   - This is the bias term, or intercept. It shifts the decision boundary.

4. **Expanded Form**:
   - The expanded form of $ W $ is given by:
     $$ W = \Sigma_{\text{W}}^{-1} (\mu_A - \mu_B) $$
     - $ \Sigma_{\text{W}} $ is the within-class scatter matrix (also known as the within-class covariance matrix).
     - $ \Sigma_{\text{W}}^{-1} $ is the inverse of the within-class scatter matrix.
     - $ \mu_A $ and $ \mu_B $ are the mean vectors of classes A and B, respectively.
     - $ \mu_A - \mu_B $ is the vector difference between the class means.

#### Between-Class Scatter

The between-class scatter $ S_B $ is part of the Fisher criterion, but it is implicit in the weight vector $ W $. Here's how:

1. **Between-Class Scatter Matrix $ S_B $**:
   - Defined as: 
     $$ S_B = (\mu_A - \mu_B)(\mu_A - \mu_B)^T $$
   - It measures the distance between the class means.

2. **Role in the Weight Vector**:
   - When calculating $ W $:
     $$ W = \Sigma_{\text{W}}^{-1} (\mu_A - \mu_B) $$
   - The term $ \mu_A - \mu_B $ is directly influenced by $ S_B $.
   - $ S_B $ ensures that the weight vector $ W $ points in the direction that maximizes the separation between class means.

### Decision Rule

The classifier uses $ g(x) $ to decide the class of $ x $:
- If $ g(x) > 0 $, classify as Class 1.
- If $ g(x) \leq 0 $, classify as Class -1.

### Summary

- **Maximizing Class Separation**: The formula $ x^T W $ ensures that the projection maximizes the separation between the classes.
- **Shift with Bias**: The bias term $ W_0 $ adjusts the decision boundary to further enhance the separation.
- **Between-Class Scatter**: While not explicitly shown in the final decision function, it is implicitly considered through $ \mu_A - \mu_B $, which impacts $ W $.

Hope this makes things clearer! Let me know if you need further details.


### Comparison with Linear Discriminant Analysis (LDA)

LDA also aims to separate classes by assuming Gaussian distributions for each class and equal covariance matrices:
$$ f(x) = (\mu_2 - \mu_1)^T \Sigma^{-1} x + w_0 $$

- **Fisher Linear Discriminant**: Optimizes the Fisher criterion directly, focusing on maximizing class separation.
- **LDA**: Assumes a Gaussian distribution per class and equal covariance matrices, leading to a similar optimization problem.

### Key Points

1. **Fisher Linear Discriminant**:
   - Focuses on the ratio of between-class variance to within-class variance.
   - Projects data onto a line that maximizes this ratio.

2. **Projection**:
   - Reduces dimensions by focusing on the most informative axis for class separation.
   - Does not necessarily lose important information, as it aims to enhance the separation between classes.

3. **Both Methods**:
   - Use the inverse of the within-class covariance matrix ($ S_W $).
   - Can encounter issues if $ S_W $ is not invertible, which happens with insufficient data.

By optimizing the Fisher criterion, the Fisher Linear Discriminant finds the projection that best separates the classes, even in a lower-dimensional space. This enhances our ability to classify new data points effectively.



## Expectation
The expectation $ E[\cdot] $ is the average value of a function over all possible outcomes. In the context of the cost function, it represents the average error between the true and predicted labels.
To calculate the expectation, you sum the product of the function and the probability of each outcome.

// more explantion here about expectation

## Least Squares
Define the cost function:
$ J(w) = E[\lvert y - w^T x \rvert^2] $

Where:
- $ J(w) $ is the cost function.
- $ y $ is the true label.
- $ w^T x $ is the predicted label.
- $ E[\cdot] $ denotes the expected value. Its over the joint distribution of $ x $ and $ y $.

Least squares means how good does $ w^T x $ predict $ y $.

Using the definition of E[.] to derive the cost function:

$ \hat{w} = R^{-1}_x E[xy] $

Where:
- $ \hat{w} $ is the optimal weight vector.
- $ R_x $ is the covariance matrix of $ x $.
- $ E[xy] $ is the expected value of the product of $ x $ and $ y $.


$ R_x = E[xx^T] $
$$ = \begin{bmatrix} E[x_1 x_1] & E[x_1 x_2] & \cdots & E[x_1 x_d] \\ E[x_2 x_1] & E[x_2 x_2] & \cdots & E[x_2 x_d] \\ \vdots & \vdots & \ddots & \vdots \\ E[x_d x_1] & E[x_d x_2] & \cdots & E[x_d x_d] \end{bmatrix} $$
for a d-dimensional input vector $ x $. explain more

The cross-correlation is:

$$ E[xy] = E \begin{bmatrix} \begin{bmatrix} E[x_1 y] \\ E[x_2 y] \\ \vdots \\ E[x_d y] \end{bmatrix} \end{bmatrix} $$ explain more
