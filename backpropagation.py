import numpy as np
import matplotlib.pyplot as plt
# How to structure the code was inspired by Chat-GPT (putting it into classes and calling forward and backward)

# MSE class for Mean Squared Error loss
class MSE:
    def __init__(self):
        self.output = None

    def compute_loss(self, predicted, target):
        self.output = (predicted - target)
        return np.mean(self.output ** 2)

    def compute_gradient(self, predicted, target):
        return 2 * self.output / self.output.size

class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, x, _):
        return x * (1 - self.output ** 2)

class ReLU:
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, gradient, _):
        return gradient * (self.output > 0)


class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, learning_rate):  # derivation
        s = self.forward(x)
        return x * s * (1 - s)
################################################################################################################

class Linear:  # fully-connected layer in practice usually named Linear
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.input_val = None

    def forward(self, input_val):
        self.input_val = input_val
        return np.dot(input_val, self.weights) + self.bias

    def backward(self, gradient, learning_rate):
        # Compute gradients for weights and bias
        grad_weights = np.dot(self.input_val.T, gradient)
        grad_bias = np.sum(gradient, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return np.dot(gradient, self.weights.T)

class NeuralNetwork:
    def __init__(self):
        self.layers = [Linear(2, 4),
                       Tanh(),
                       Linear(4, 1),
                       Tanh()]

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
target = np.array([[0], [1], [1], [0]])
model = NeuralNetwork()
train(model, input_value, target, 500, 0.1)

for input in input_value:
    result = model.forward(input)
    print(f"Input: {input}, Output: {result}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, 500 + 1), loss_array, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Testing Loss Over Epochs")
plt.legend()
plt.show()
