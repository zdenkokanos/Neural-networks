import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
INPUT_SIZE = 28*28
FC1_SIZE = 512
FC2_SIZE = 256
OUTPUT_SIZE = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_OF_EPOCHS = 10

# Download the MNIST Dataset if not downloaded yet
transform = transforms.ToTensor()  # Converts image to tensor with values in range [0, 1]
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Load the datasets into DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, FC1_SIZE)  # First hidden layer (784 -> 256)
        self.fc2 = nn.Linear(FC1_SIZE, FC2_SIZE)  # Second hidden layer (256 -> 64)
        self.fc3 = nn.Linear(FC2_SIZE, OUTPUT_SIZE)  # Output layer (64 -> 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten into 1D vector
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = NeuralNetwork()

criterion = nn.CrossEntropyLoss()
optimize_ADAM = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimize_SGD = optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimize_SGD_Momentum = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

optimizer = optimize_ADAM

time_s = time.time()
for epoch in range(NUM_OF_EPOCHS):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()  # Track loss

        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Calculate and print training stats
    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{NUM_OF_EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Evaluate on the test set after each epoch
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.2f}%\n")

print(time_s - time.time())

