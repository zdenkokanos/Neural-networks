import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  # needed for heatmap for confusion matrix

# Hyperparameters
INPUT_SIZE = 28*28
FC1_SIZE = 512
FC2_SIZE = 128
OUTPUT_SIZE = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_OF_EPOCHS = 20

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
        self.fc1 = nn.Linear(INPUT_SIZE, FC1_SIZE)  # first hidden layer (784 -> 512)
        self.fc2 = nn.Linear(FC1_SIZE, FC2_SIZE)  # second hidden layer (512 -> 128)
        self.fc3 = nn.Linear(FC2_SIZE, OUTPUT_SIZE)  # output layer (128 -> 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten into 1D vector / Input layer
        x = self.relu(self.fc1(x))  # passed into first hidden layer
        x = self.relu(self.fc2(x))   # passed into second hidden layer
        x = self.fc3(x)  # passed into output layer
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
optimizer = initialize_optimizer()
loss_function = nn.CrossEntropyLoss()  # default loss function to use for multi-class classification problems

# track losses and accuracies for plotting
training_losses = []
testing_losses = []
testing_accuracies = []

for epoch in range(NUM_OF_EPOCHS):
    model.train()  # set model to training mode
    train_loss = 0.0
    correct = 0
    total = 0

    for image_inputs, targets in train_loader:
        optimizer.zero_grad()  # set gradients to zero
        outputs = model(image_inputs)  # forward pass
        loss = loss_function(outputs, targets)  # compute loss
        loss.backward()  # backpropagation
        optimizer.step()  # update weights according to gradients

        train_loss += loss.item()  # track loss

        _, predicted_num = torch.max(outputs, 1)
        total += targets.size(0)

        for i in range(len(predicted_num)):
            if predicted_num[i] == targets[i]:  # compare predictions with targets
                correct += 1

    training_loss = train_loss / len(train_loader)
    training_losses.append(training_loss)
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{NUM_OF_EPOCHS}], Accuracy: {train_accuracy:.2f}%")

    # evaluate the neural network on test set after each epoch
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():  # do not compute gradients during evaluation
        for image_inputs, targets in test_loader:
            output = model(image_inputs)
            loss = loss_function(output, targets)
            _, predicted_num = torch.max(output, 1)  # returns the maximum values and their indexes
            total += targets.size(0)

            test_loss += loss.item()  # accumulate test loss
            for i in range(len(predicted_num)):  # compare predictions with targets to check accuracy
                if predicted_num[i] == targets[i]:
                    correct += 1

    test_loss /= len(test_loader)
    testing_losses.append(test_loss)
    test_accuracy = 100 * correct / total
    testing_accuracies.append(test_accuracy)
    print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.2f}%\n")

#####################################################################################################################
# Used Chat-GPT to display graphs and confusion matrix #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(range(1, NUM_OF_EPOCHS + 1), training_losses, label="Training Loss")
ax1.plot(range(1, NUM_OF_EPOCHS + 1), testing_losses, label="Testing Loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Testing Loss Over Epochs")
ax1.legend()

ax2.plot(range(1, NUM_OF_EPOCHS + 1), testing_accuracies, label="Testing Accuracy", color='orange')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Testing Accuracy Over Epochs")
ax2.legend()

plt.tight_layout()
plt.show()

# targets and predicted labels for the confusion matrix
all_predicted = []
all_targets = []

model.eval()  
with torch.no_grad():
    for image_inputs, targets in test_loader:
        outputs = model(image_inputs)
        _, predicted_num = torch.max(outputs, 1)  
        all_predicted.extend(predicted_num.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Compute the confusion matrix
conf_matrix = confusion_matrix(all_targets, all_predicted)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(10)), yticklabels=list(range(10)))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
#####################################################################################################################
