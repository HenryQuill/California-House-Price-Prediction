
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df

# ReLU activation function
def relu(x):
    return np.maximum(0, x)
# Derivative of ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Mean Squared Error (MSE) loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

# Feed Forward Neural Network (FNN) class
class FNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize network parameters
        # input_size: number of input features (8 for California Housing dataset)
        # hidden_size: number of neurons in the hidden layer
        # output_size: number of output neurons (1 for regression)
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)  # Hidden layer weights, He initialization for stable gradients
        self.b1 = np.zeros((hidden_size, 1))  # Hidden layer bias, initialized to 0
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)  # Output layer weights, He initialization
        self.b2 = np.zeros((output_size, 1))  # Output layer bias, initialized to 0
        self.learning_rate = learning_rate

    def forward(self, X):
        # Forward propagation: compute network output from input X
        # X: input matrix with shape (input_size, m), m is the number of samples
        self.Z1 = np.dot(self.W1, X) + self.b1  # Hidden layer input: W1 * X + b1
        self.A1 = relu(self.Z1)  # Apply ReLU to hidden layer for non-linearity
        self.Z2 = np.dot(self.W2, self.A1) + self.b2  # Output layer input: W2 * A1 + b2
        self.A2 = self.Z2  # No activation at output (regression)
        return self.A2  # Return predictions: matrix (output_size, m)

    def backward(self, X, y, y_pred):
        # Backpropagation: update weights and biases based on error
        # X: input, y: true values, y_pred: predicted values
        m = X.shape[1]  # Number of samples
        # Output layer gradient
        delta2 = 2 * (y_pred - y) / m  # Derivative of MSE: 2 * (y_pred - y) / m
        dW2 = np.dot(delta2, self.A1.T)  # Gradient of W2: delta2 * A1^T
        db2 = np.sum(delta2, axis=1, keepdims=True)  # Gradient of b2: sum delta2 over samples
        # Hidden layer gradient
        delta1 = np.dot(self.W2.T, delta2) * relu_derivative(self.Z1)  # Backpropagate gradient to hidden layer, multiply by ReLU derivative
        dW1 = np.dot(delta1, X.T)  # Gradient of W1: delta1 * X^T
        db1 = np.sum(delta1, axis=1, keepdims=True)  # Gradient of b1: sum delta1 over samples
        # Update weights and biases using gradient descent
        self.W1 -= self.learning_rate * dW1  # Update W1: W1 = W1 - learning_rate * dW1
        self.b1 -= self.learning_rate * db1  # Update b1
        self.W2 -= self.learning_rate * dW2  # Update W2
        self.b2 -= self.learning_rate * db2  # Update b2

    def train(self, X, y, epochs):
        # Train the model for a fixed number of epochs
        # X: training data, y: labels, epochs: number of iterations
        losses = []  # List to store loss over epochs
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward(X)  # Compute predictions
            # Compute MSE loss
            loss = mse_loss(y, y_pred)  # Compare predictions with true values
            losses.append(loss)  # Store loss
            # Backpropagation to update parameters
            self.backward(X, y, y_pred)
            # Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
        return losses  # Return loss list

# Load and preprocess California Housing dataset
def load_data():
    # Load California Housing dataset from sklearn
    data = fetch_california_housing()
    X = data.data.T  # Transpose feature matrix to (n_features, n_samples)
    y = data.target.reshape(1, -1)  # Reshape labels to (1, n_samples)

    # Standardize features to mean=0, variance=1
    scaler = StandardScaler()
    X = scaler.fit_transform(X.T).T  # Standardize by samples, then transpose back

    # Split data into training (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2, random_state=42)
    X_train, X_test = X_train.T, X_test.T  # Transpose back to (n_features, n_samples)
    y_train, y_test = y_train.T, y_test.T  # Reshape labels to (1, n_samples)
    return X_train, X_test, y_train, y_test

# Main function
def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()  # Get training and test data

    # Initialize neural network
    input_size = X_train.shape[0]  # Number of features (8)
    hidden_size = 10  # Number of neurons in hidden layer
    output_size = 1  # Number of output neurons (1 for regression)
    learning_rate = 0.001  # Learning rate
    epochs = 1000  # Number of epochs

    # Create neural network model
    model = FNN(input_size, hidden_size, output_size, learning_rate)

    # Train the network
    losses = model.train(X_train, y_train, epochs)  # Train and store loss

    # Evaluate on test set
    y_pred = model.forward(X_test)  # Predict on test set
    test_loss = mse_loss(y_test, y_pred)  # Compute MSE loss on test set
    print(f'Test Loss: {test_loss:.4f}')  # Print test loss

    # Plot training loss
    print()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

if __name__ == "__main__":
    main()  