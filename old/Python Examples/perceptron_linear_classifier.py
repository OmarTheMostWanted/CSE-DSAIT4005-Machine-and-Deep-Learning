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