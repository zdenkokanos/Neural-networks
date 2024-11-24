import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the transformation to convert images to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download the training and test dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Load the datasets into DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 1. Inspect the first batch of images and labels
for images, labels in train_loader:
    print("First batch shapes:")
    print(f"Images shape: {images.shape}")  # Shape of the batch of images
    print(f"Labels shape: {labels.shape}")  # Shape of the batch of labels
    break  # Just inspect the first batch

# 2. Visualize the first image and its label
image, label = train_dataset[2]
image = image.numpy().squeeze()  # Remove the single channel dimension
plt.imshow(image, cmap='gray')
plt.title(f'Label: {label}')
plt.show()

# 4. Check the length of the training and test datasets
print(f'Training dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

