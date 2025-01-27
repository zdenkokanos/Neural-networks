import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
INPUT_SIZE = 28*28
FC1_SIZE = 256
FC2_SIZE = 64
OUTPUT_SIZE = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_OF_EPOCHS = 6

#####################################################################################################################
# Part of downloading and loading dataset is from website:
# https://medium.com/@bpoyeka1/building-simple-neural-networks-nn-cnn-using-pytorch-for-mnist-dataset-31e459d17788
# Download the MNIST Dataset if not downloaded yet
transform = transforms.ToTensor()  # Converts image to tensor with values in range [0, 1]
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Load the datasets into DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#####################################################################################################################
# https://medium.com/@bpoyeka1/building-simple-neural-networks-nn-cnn-using-pytorch-for-mnist-dataset-31e459d17788
# Structure was inspired by website above, except that everything is done by me

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, FC1_SIZE)  # First hidden layer (784 -> 256)
        self.fc2 = nn.Linear(FC1_SIZE, FC2_SIZE)  # Second hidden layer (256 -> 64)
        self.fc3 = nn.Linear(FC2_SIZE, OUTPUT_SIZE)  # Output layer (64 -> 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten into 1D vector / Input layer
        x = self.relu(self.fc1(x))  # Passed into first hidden layer
        x = self.relu(self.fc2(x))   # Passed into second hidden layer
        x = self.fc3(x)  # Passed into output layer
        return x

def initialize_optimizer():
    optimizer_type = input("What optimizing algorithm would you like to use? ('adam', 'sgd', 'sgd_momentum'): \n")
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_type == 'sgd_momentum':
        return optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()  # default loss function to use for multi-class classification problems
optimizer = initialize_optimizer()

# Track losses for plotting
training_losses = []
testing_losses = []
testing_accuracies = []

for epoch in range(NUM_OF_EPOCHS):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for image_inputs, targets in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(image_inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Track loss

        # Track accuracy
        _, predicted_num = torch.max(outputs, 1)
        total += targets.size(0)

        for i in range(len(predicted_num)):
            if predicted_num[i] == targets[i]:  # Compare predictions with targets
                correct += 1

    training_loss = running_loss / len(train_loader)
    training_losses.append(training_loss)
    # Calculate and print training stats
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{NUM_OF_EPOCHS}], Accuracy: {train_accuracy:.2f}%")

    # Evaluate on the test set after each epoch
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for image_inputs, targets in test_loader:
            output = model(image_inputs)
            loss = criterion(output, targets)
            _, predicted_num = torch.max(output, 1)  # returns the maximum values and its index
            total += targets.size(0)

            test_loss += loss.item()  # Accumulate test loss
            for i in range(len(predicted_num)):  # Loop over the indices of the batch
                if predicted_num[i] == targets[i]:  # Compare predictions with targets
                    correct += 1

    test_loss /= len(test_loader)
    testing_losses.append(test_loss)
    test_accuracy = 100 * correct / total
    testing_accuracies.append(test_accuracy)
    print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.2f}%\n")

plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_OF_EPOCHS + 1), training_losses, label="Training Loss")
plt.plot(range(1, NUM_OF_EPOCHS + 1), testing_losses, label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Testing Loss Over Epochs")
plt.legend()
plt.show()
