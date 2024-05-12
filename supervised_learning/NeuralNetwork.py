from Predictor import Predictor
import numpy as np

class NeuralNetworkClassifier(Predictor):
    def __init__(self, input_size, hidden_size=100, output_size=2, learning_rate=0.01, epochs=100):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # Forward pass
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            output_output = self.softmax(output_input)

            # Backward pass
            loss = -np.sum(y * np.log(output_output))  # Cross-entropy loss
            error = output_output - y

            # Update weights and biases
            self.weights_hidden_output -= self.learning_rate * np.dot(hidden_output.T, error)
            self.bias_output -= self.learning_rate * np.sum(error, axis=0)
            hidden_error = np.dot(error, self.weights_hidden_output.T) * (hidden_output * (1 - hidden_output))
            self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_error)
            self.bias_hidden -= self.learning_rate * np.sum(hidden_error, axis=0)

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output_output = self.softmax(output_input)
        return np.argmax(output_output, axis=1)

class NeuralNetworkRegressor(Predictor):
    def __init__(self, input_size, hidden_size=100, output_size=1, learning_rate=0.01, epochs=100):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # Forward pass
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            output = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output

            # Backward pass
            error = output - y
            loss = np.mean(error ** 2)  # Mean squared error loss

            # Update weights and biases
            self.weights_hidden_output -= self.learning_rate * np.dot(hidden_output.T, error)
            self.bias_output -= self.learning_rate * np.sum(error, axis=0)
            hidden_error = np.dot(error, self.weights_hidden_output.T) * (hidden_output * (1 - hidden_output))
            self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_error)
            self.bias_hidden -= self.learning_rate * np.sum(hidden_error, axis=0)

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        output = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        return output
