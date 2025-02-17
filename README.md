# Grayscale-to-Color Image Processing Using Deep Learning

This project demonstrates how to use a deep learning model to convert grayscale images into colored images. The model is a Convolutional Neural Network (CNN) that learns to predict the RGB channels from a grayscale input, essentially performing grayscale-to-color image conversion.

## Table of Contents
- Overview
- Technologies Used
- Dataset
- Model Architecture
- Data Preprocessing
- Training the Model
- Testing the Model
- Results
- How to Run
- Embedding Setup
- License

## Overview
The project uses a deep learning model to colorize grayscale images. It takes grayscale images as input and generates their corresponding colored versions. The model is trained using PyTorch on a large image dataset, with colorization tasks enhanced through feature extraction and augmentation.

## Technologies Used
- **Deep Learning Framework**: PyTorch
- **Preprocessing & Augmentation**: OpenCV, PyTorch torchvision
- **GPU Support**: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- **Visualization**: Matplotlib

## Dataset
For demonstration purposes, the model uses the CIFAR-10 dataset to train the grayscale-to-color transformation model. You can embed your custom dataset by modifying the data loading portion of the code.

If you use the CIFAR-10 dataset:
- **Size**: 60,000 32x32 color images in 10 classes.
- **Source**: CIFAR-10 dataset.

## Model Architecture
The model is a simple convolutional neural network (CNN) designed to take grayscale images as input and output three channels of color information (RGB).

- **Input**: A 32x32 grayscale image (single channel).
- **Conv Layers**: Series of convolutional layers with increasing filters (32, 64, 128, etc.).
- **Output**: A 32x32 image with three channels (RGB).

## Data Preprocessing
The preprocessing involves:
1. **Grayscale Conversion**: Convert color images into grayscale to use them as input for the network.
2. **Normalization**: Normalize pixel values to the range [0, 1].
3. **Augmentation**: Random transformations such as rotation, shifting, flipping, etc.

### Embedding Data Preprocessing
Here’s an embedded setup for preprocessing and augmentation:

```python
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import CIFAR10

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # Convert image to grayscale
    transforms.Resize((32, 32)),                    # Resize to 32x32
    transforms.ToTensor(),                          # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])     # Normalize to [-1, 1]
])

# Download CIFAR-10 dataset and apply transformations
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

# Download CIFAR-10 dataset and apply transformations

```
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
Training the Model
The model is trained using an Adam optimizer and Mean Squared Error (MSE) Loss. The training process utilizes GPU acceleration for faster computation.
Code for Training the Model
python
CopyEdit
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
```

# Model

```
class ColorizerCNN(nn.Module):
    def __init__(self):
        super(ColorizerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256 * 32 * 32, 3 * 32 * 32)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x.view(x.size(0), 3, 32, 32)  # Output 3 color channels
```

# Initialize Model
```
model = ColorizerCNN().to(device)
```

# Loss & Optimizer
```
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

# Train Model
```
for epoch in range(20):
    model.train()
    for gray_images, color_images in train_loader:
        gray_images, color_images = gray_images.to(device), color_images.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(gray_images)
        
        # Compute loss
        loss = criterion(outputs, color_images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/20], Loss: {loss.item():.4f}')

    
    # Save the model periodically
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'colorizer_epoch_{epoch+1}.pth')
```
# Testing the Model
After training, test the model by visualizing the output from grayscale input. Display the following images:
1.	Original Color Image
2.	Grayscale Image
3.	Colorized Image (Output)

# How to Run
1.	Clone this repository.
2.	Install the required libraries:
bash
```
pip install torch torchvision matplotlib opencv-python
```
3.	Run the model training:
bash
```
python train_model.py
```
4.	After training, test the model:
bash
```
python test_model.py
```
# License
This project is licensed under the MIT License - see the LICENSE file for details
