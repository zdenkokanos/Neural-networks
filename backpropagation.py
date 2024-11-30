import numpy as np
# How to structure the code was inspired by Chat-GPT (putting it into classes and calling forward and backward)

# MSE class for Mean Squared Error loss
class MSE:
    def compute_loss(self, predicted, target):
        return np.mean((predicted - target) ** 2)

    def compute_gradient(self, predicted, target):
        return 2 * (predicted - target) / predicted.shape[0]

class TanH:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, learning_rate):
        x = np.tanh(x)
        return x * (1 - x)


class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, learning_rate):  # derivation
        s = self.forward(x)
        return s * (1 - s)
################################################################################################################

class Linear:  # fully-connected layer in practice usually named Linear
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)  # when initializing, assign random weights
        self.bias = np.random.rand(1, output_size)
        self.input_val = None

    def forward(self, input_val):
        self.input_val = input_val
        return np.dot(input_val, self.weights) + self.bias

    def backward(self, gradient, learning_rate):
        # Compute gradients for weights and bias
        grad_weights = np.dot(self.input_val.T, gradient)
        grad_bias = np.sum(gradient, axis=0)
        grad_input = np.dot(gradient, self.weights.T)  # Gradient w.r.t. input

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input  # Return gradient to propagate backward


class NeuralNetwork:
    def __init__(self):
        self.layers = [Linear(2, 4),
                       TanH(),
                       Linear(4, 1),
                       TanH()]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)


loss_array = []  # for plotting
loss_function = MSE()

def train(model, inputs, targets, epochs, learning_rate):
    for epoch in range(epochs):
        outputs = model.forward(inputs)
        loss = loss_function.compute_loss(outputs, targets)

        loss_array.append(loss)
        gradient = loss_function.compute_gradient(outputs, targets)

        model.backward(gradient, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


input_value = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [1]])
model = NeuralNetwork()
train(model, input_value, target, 500, 0.1)

for input in input_value:
    result = model.forward(input)
    print(f"Input: {input}, Output: {result}")



