import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from PIL import Image
import os
from configs import *
import copy
import argparse

# Define the transforms for training and testing datasets
transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(padding=4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}


# Custom dataset class to load the images
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_weights = []

        classes = sorted(os.listdir(self.folder_path))
        for class_index, class_name in enumerate(classes):
            class_path = os.path.join(self.folder_path, class_name)
            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                self.image_paths.extend([os.path.join(class_path, img) for img in images])
                self.labels.extend([class_index] * len(images))
                self.class_weights.append(1./len(images))
        
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32) / sum(self.class_weights)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='faceswap')
    args = parser.parse_args()

    task = args.task
    DIR = {'faceswap': FACESWAP, 'deepfake': DEEPFAKE}[task]
    print(DIR)

    # Define the paths to the folders
    folder_A = os.path.join(DIR, 'train')
    folder_C = os.path.join(DIR, 'test')

    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create training and testing datasets
    train_dataset = ImageDataset(os.path.join(folder_A), transform=transform['train'])
    test_dataset = ImageDataset(os.path.join(folder_C), transform=transform['test'])

    # Define the class weights for training
    class_weights = train_dataset.class_weights.to(device)

    # Create data loaders for training and testing datasets
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Two output classes
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

        # Evaluate the model
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total    
        print(f"Test Accuracy: {accuracy}%")
        
    torch.save(model, os.path.join(DIR, 'model.pt'))