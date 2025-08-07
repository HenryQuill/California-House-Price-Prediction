# Early stopping application
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)  # Hidden layer weights
        self.b1 = np.zeros((hidden_size, 1))  # Hidden layer bias
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)  # Output layer weights
        self.b2 = np.zeros((output_size, 1))  # Output layer bias
        self.learning_rate = learning_rate

    def forward(self, X):
        # Forward propagation: compute network output from input X
        self.Z1 = np.dot(self.W1, X) + self.b1  # Hidden layer input
        self.A1 = relu(self.Z1)  # Apply ReLU to hidden layer
        self.Z2 = np.dot(self.W2, self.A1) + self.b2  # Output layer input
        self.A2 = self.Z2  # No activation at output (regression)
        return self.A2  # Return predictions

    def backward(self, X, y, y_pred):
        # Backpropagation: update weights and biases based on error
        m = X.shape[1]  # Number of samples
        delta2 = 2 * (y_pred - y) / m  # Output layer gradient (derivative of MSE)
        dW2 = np.dot(delta2, self.A1.T)  # Gradient of W2
        db2 = np.sum(delta2, axis=1, keepdims=True)  # Gradient of b2
        delta1 = np.dot(self.W2.T, delta2) * relu_derivative(self.Z1)  # Hidden layer gradient
        dW1 = np.dot(delta1, X.T)  # Gradient of W1
        db1 = np.sum(delta1, axis=1, keepdims=True)  # Gradient of b1
        # Update weights and biases using gradient descent
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X_train, y_train, X_val, y_val, epochs, patience=50, min_delta=0.0):
        # Train the network with Early Stopping
        train_losses = []  # Store training loss
        val_losses = []  # Store validation loss
        best_val_loss = float('inf')  # Best validation loss
        best_weights = {}  # Store best weights
        patience_counter = 0  # Counter for Early Stopping

        for epoch in range(epochs):
            # Forward propagation on training set
            y_pred_train = self.forward(X_train)
            train_loss = mse_loss(y_train, y_pred_train)
            train_losses.append(train_loss)

            # Backpropagation to update parameters
            self.backward(X_train, y_train, y_pred_train)

            # Compute loss on validation set
            y_pred_val = self.forward(X_val)
            val_loss = mse_loss(y_val, y_pred_val)
            val_losses.append(val_loss)

            # Check Early Stopping
            if val_loss < best_val_loss - min_delta:  # If validation loss improves
                best_val_loss = val_loss
                best_weights = {          # Save best weights to best_weights
                    'W1': self.W1.copy(),
                    'b1': self.b1.copy(),
                    'W2': self.W2.copy(),
                    'b2': self.b2.copy()
                }
                patience_counter = 0  # Reset counter
            else:
                patience_counter += 1  # Increase counter if no improvement

            # Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Stop early if no improvement after 'patience' epochs
            if patience_counter >= patience and best_weights:
                print(f'Early stopping at epoch {epoch + 1}, Best Val Loss: {best_val_loss:.4f}')
                self.W1 = best_weights['W1']  # Restore best weights
                self.b1 = best_weights['b1']
                self.W2 = best_weights['W2']
                self.b2 = best_weights['b2']
                break

        return train_losses, val_losses

# Load and preprocess California Housing dataset
def load_data():
    # Load data from sklearn
    data = fetch_california_housing()
    X = data.data.T  # Shape (n_features, n_samples)
    y = data.target.reshape(1, -1)  # Shape (1, n_samples)

    # Standardize features to mean=0, variance=1
    scaler = StandardScaler()
    X = scaler.fit_transform(X.T).T

    # Split data into training, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X.T, y.T, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    X_train, X_val, X_test = X_train.T, X_val.T, X_test.T  # Shape (n_features, n_samples)
    y_train, y_val, y_test = y_train.T, y_val.T, y_test.T
    return X_train, X_val, X_test, y_train, y_val, y_test

# Main function
def main():
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Initialize network
    input_size = X_train.shape[0]  # Number of features (8)
    hidden_size = 10  # Number of neurons in hidden layer
    output_size = 1  # Number of output neurons (regression)
    learning_rate = 0.001
    epochs = 1000
    patience = 50  # Number of patience epochs for Early Stopping
    min_delta = 0.0  # Minimum improvement in validation loss

    # Create neural network object
    model = FNN(input_size, hidden_size, output_size, learning_rate)

    # Train with Early Stopping
    train_losses, val_losses = model.train(X_train, y_train, X_val, y_val, epochs, patience, min_delta)

    # Evaluate on test set
    y_pred_test = model.forward(X_test)
    test_loss = mse_loss(y_test, y_pred_test)
    print(f'Final Test Loss: {test_loss:.4f}')
    print()

    # Plot loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Early Stopping)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()  