import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Define the image size to resize images for VGG16
IMG_SIZE = 224

# Define directories for training and validation datasets
train_dir = 'iRoads/train'  # Change to the path where your training images are stored
validation_dir = 'iRoads/validation'  # Change to the path where your validation images are stored

# Define the transformations to apply to the dataset (normalizing to ImageNet mean and std)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean and std
])

# Load the datasets with ImageFolder, assuming each class is in a subfolder (road, non-road)
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
validation_dataset = datasets.ImageFolder(validation_dir, transform=transform)

# Create DataLoader for both training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Load the pre-trained VGG16 model without the top layers (classifier)
model = models.vgg16(pretrained=True)

# Freeze the layers of VGG16 (except the final classifier layers)
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier part of VGG16 to fit our binary classification (road vs non-road)
model.classifier[6] = nn.Linear(4096, 1)  # For binary classification (use 2 for multi-class classification)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define the loss function (binary cross entropy for binary classification)
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer (Adam optimizer, only optimizing the classifier part)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

# Function for training the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())  # Remove singleton dimension and calculate loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Convert output to binary predictions
            predicted = (torch.sigmoid(outputs).squeeze() > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print statistics for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Function for evaluating the model
def evaluate_model(model, validation_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item()

            # Convert output to binary predictions
            predicted = (torch.sigmoid(outputs).squeeze() > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {running_loss/len(validation_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model
evaluate_model(model, validation_loader, criterion)

# Save the trained model
torch.save(model.state_dict(), 'vgg16_iroads_model.pth')
