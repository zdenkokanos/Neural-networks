import numpy as np


# Sigmoid activation function with clipping to avoid overflow
def sigmoid(x):
    # Clip the values to a range to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

class Linear:
    def __init__(self, input_size=2, output_size=1):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))  # Initialize bias
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        input_gradient = np.dot(output_gradient, self.weights.T)

        # Update weights and bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient


class NeuralNetwork:
    def __init__(self):
        # Define the architecture of the network
        self.layers = [
            Linear(2, 4),  # Input layer to hidden layer (2 inputs -> 4 hidden neurons)
            sigmoid,  # Activation function
            Linear(4, 1)  # Hidden layer to output layer (4 hidden neurons -> 1 output)
        ]
        self.activations = []  # To store outputs of layers and activation functions
        self.layer_outputs = []  # To store the outputs of linear layers

    def forward(self, X):
        input_data = X
        self.activations = []  # Reset activations for each forward pass
        self.layer_outputs = []  # Reset layer outputs

        # Forward pass through layers
        for layer in self.layers:
            if isinstance(layer, Linear):
                output = layer.forward(input_data)
                self.layer_outputs.append(output)
            elif callable(layer):  # For activation functions
                output = layer(input_data)

            self.activations.append(output)
            input_data = output

        return input_data

    def backward(self, X, y, learning_rate):
        output = self.activations[-1]
        loss = output - y
        loss_derivative = loss * 2  # For MSE, derivative of loss is 2 * (pred - true)

        grad = loss_derivative
        # Backpropagate through layers in reverse order
        for i in range(len(self.layers) - 1, -1, -1):
            if isinstance(self.layers[i], Linear):
                grad = self.layers[i].backward(grad, learning_rate)
            elif self.layers[i] == sigmoid:
                grad = grad * sigmoid_derivative(self.activations[i - 1])
            elif self.layers[i] == tanh:
                grad = grad * tanh_derivative(self.activations[i - 1])
            elif self.layers[i] == relu:
                grad = grad * relu_derivative(self.activations[i - 1])


def train(model, X_train, y_train, epochs, learning_rate):
    for epoch in range(epochs):
        output = model.forward(X_train)
        model.backward(X_train, y_train, learning_rate)

        if epoch % 100 == 0:
            mse = np.mean((output - y_train) ** 2)
            print(f'Epoch {epoch}, MSE: {mse}')


# XOR dataset
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Create and train the model
model = NeuralNetwork()
train(model, X_train, y_train, epochs=500, learning_rate=0.01)