import numpy as np
import matplotlib.pyplot as plt
# How to structure the code was inspired by Chat-GPT (putting it into classes and calling forward and backward)

# MSE class for Mean Squared Error loss
class MSE:
    def __init__(self):
        self.forward_output = None

    def compute_loss(self, predicted, target):
        self.forward_output = (predicted - target)
        return np.mean(self.forward_output ** 2)

    def compute_gradient(self):
        return 2 * self.forward_output / self.forward_output.size

class Tanh:
    def __init__(self):
       self.forward_output = None

    def forward(self, x):
        self.forward_output = np.tanh(x)
        return self.forward_output

    def backward(self, x, _):
        return x * (1 - self.forward_output ** 2)

class ReLU:
    def __init__(self):
        self.forward_output = None

    def forward(self, x):
        self.forward_output = np.maximum(0, x)
        return self.forward_output

    def backward(self, gradient, _):
        return gradient * (self.forward_output > 0)


class Sigmoid:
    def __init__(self):
        self.forward_output = None

    def forward(self, x):
        self.forward_output = 1 / (1 + np.exp(-x))
        return self.forward_output

    def backward(self, x, learning_rate):  # derivation
        s = self.forward(x)
        return x * self.forward_output * (1 - self.forward_output)
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
    def __init__(self, layers):
        self.layers = layers

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
        gradient = loss_function.compute_gradient()

        model.backward(gradient, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


function = input("Zadaj 'AND', 'OR' alebo 'XOR': ")
if function == "AND":
    target = np.array([[0], [0], [0], [1]])
elif function == "OR":
    target = np.array([[0], [1], [1], [1]])
else:
    target = np.array([[0], [1], [1], [0]])

num_layers = int(input("Zadaj 1 pre 1 skrytú vrstvu alebo 2 pre dve skryté vrstvy: "))
if num_layers == 1:
    layers = [Linear(2, 4),
              Tanh(),
              Linear(4, 1),
              Tanh()]
else:
    layers = [Linear(2, 4),
              Tanh(),
              Linear(4, 4),
              Tanh(),
              Linear(4, 1),
              Tanh()]

input_value = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
model = NeuralNetwork(layers)
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
