import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from PIL import Image
import os
from configs import *
from train_vgg import ImageDataset, transform
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import copy
from six.moves import cPickle as pkl
import seaborn as sns
sns.set_theme()
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='faceswap')
args = parser.parse_args()

task = args.task
DIR = {'faceswap': FACESWAP, 'deepfake': DEEPFAKE}[task]

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
Model = models.vgg16_bn()
Model.classifier[6] = torch.nn.Linear(4096, 2)  # Assuming the output layer has 2 classes
Model = torch.load(os.path.join(DIR, 'model_vgg.pt'))

L = 2

model = copy.deepcopy(torch.nn.Sequential( *list(list(Model.children())[0].children())[:37] ))
model2 = copy.deepcopy(torch.nn.Sequential( *(list(list(Model.children())[0].children())[37:] + list(Model.children())[1:2] + [nn.Flatten()] + list(Model.children())[2:]) ))
del Model

Feats, Labels = [], []
Feats_t, Labels_t = [], []
model.eval()

# since phi is fixed, extract train-test data features

for images, labels in train_loader:
    images = images.to(device)
    Feats.append(model(images).detach().cpu())
    Labels.append(labels)
    
for images, labels in test_loader:
    images = images.to(device)
    Feats_t.append(model(images).detach().cpu())
    Labels_t.append(labels)
    
Feats, Labels = torch.cat(Feats), torch.cat(Labels)

del model 

trials = 5
ALL_AUROC = []
ALL_std_at_fixed_alpha = []
std = np.linspace(0, 20, 21)
no_noises = 10
# Orig_outputs = []

start_time = time.time()

for trial in range(trials):
    
    AUROC = []
    std_at_fixed_alpha = []

    for s in std:     
        
        fc = copy.deepcopy(model2).to(device)
        fc.train()                
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(fc.parameters(), lr=0.0001)

        # Train the model
        num_epochs = 10
        y_true = []
        y_scores = []
        
        for epoch in range(num_epochs):
            fc.train()
            idx = np.arange(len(Feats))
            np.random.shuffle(idx)
            Feats, Labels = Feats[idx], Labels[idx]
            
            # for images, labels in zip(Feats, Labels):
            for i in range(0, len(Feats), batch_size):
                images = copy.deepcopy(Feats[i: i+batch_size]).to(device)
                labels = copy.deepcopy(Labels[i: i+batch_size]).to(device)
                optimizer.zero_grad()
                noise = (torch.randn_like(images) * s).to(device)
                outputs = fc(images + noise)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Noise: {s}")

        # Evaluate the model
        fc.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in zip(Feats_t, Labels_t):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = fc(images)                
                _, predicted = torch.max(outputs.data, 1)
                # Orig_outputs.append(predicted.detach().cpu())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                probabilities = torch.nn.Softmax(dim=-1)(outputs)
                y_scores.extend(probabilities[:, 1].tolist())
                y_true.extend(labels.tolist())

        accuracy = 100 * correct / total       
        auroc = roc_auc_score(y_true, y_scores)
        AUROC.append(auroc)
        print(f"Trial: {trial}, AUROC: {auroc}, STD: {s}, Test Accuracy: {accuracy}%, Time: {(time.time() - start_time)//60} mins")
        
        # find inference sigma at which alpha = 1% is achieved.
        tmp = 0
        for s_ in np.arange(0., s+0.2, 0.2):
            fc.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                # ii = 0
                for images, labels in zip(Feats_t, Labels_t):
                    
                    images = images.to(device)  
                                  
                    orig_outputs = fc(images)   
                    _, orig_predicted = torch.max(orig_outputs.data, 1)     
                    
                    for ii in range(no_noises):                    
                        noise = (torch.randn_like(images) * s_).to(device)
                        outputs = fc(images + noise)             
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == orig_predicted).sum().item()

            accuracy = 100 * correct / total   
            
            if 100 - accuracy <= 1:
                tmp = s_
            else:
                break 
            
        std_at_fixed_alpha.append(tmp)
                
    ALL_AUROC.append(AUROC)
    ALL_std_at_fixed_alpha.append(std_at_fixed_alpha)

print([ALL_AUROC, ALL_std_at_fixed_alpha, std])

with open("results/result_L={}_task={}_10_vgg.pkl".format(L, task), "wb") as f:
    pkl.dump([ALL_AUROC, ALL_std_at_fixed_alpha, std], f)