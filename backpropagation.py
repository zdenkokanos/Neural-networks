import numpy as np
# How to structure the code was inspired by Chat-GPT (putting it into classes and calling forward and backward)

# MSE class for Mean Squared Error loss
class MSE:
    def compute_loss(self, predicted, target):
        return np.mean((predicted - target) ** 2)

    def compute_gradient(self, predicted, target):
        return 2 * (predicted - target) / predicted.shape[0]

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):  # derivation
        s = self.forward(x)
        return s * (1 - s)
################################################################################################################

class Linear:  # fully-connected layer in practice usually named Linear
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)  # when initializing, assign random weights
        self.bias = np.random.rand(1, output_size)
        self.output = None

    def forward(self, input_val):
        self.output = np.dot(input_val, self.weights) + self.bias
        return self.output

    def backward(self, gradient, learning_rate):
        grad_input = np.dot(gradient, self.weights.T)  # Gradient w.r.t. input
        grad_weights = np.dot(self.input_val.T, gradient)  # Gradient w.r.t. weights
        grad_bias = np.sum(gradient, axis=0, keepdims=True)  # Gradient w.r.t. bias

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias


class NeuralNetwork:
    def __init__(self):
        self.layers = [Linear(2, 4),
                       Sigmoid(),
                       Linear(4, 1),
                       Sigmoid()]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


loss_array = []  # for plotting

def train(model, inputs, targets, epochs, learning_rate):
    for epoch in range(epochs):
        outputs = model.forward(inputs)
        loss = MSE.compute_loss(outputs, targets)

        loss_array.append(loss)
        gradient = MSE.compute_gradient(outputs, targets)

        model.backward(gradient, learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")


input_value = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [0], [1], [1]])
model = NeuralNetwork()
train(model, input_value, target, 500, 0.01)



