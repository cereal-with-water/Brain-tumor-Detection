# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from PIL import Image

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#read data
data_dir = './data'
sets = ['yes', 'no']

# Define transformations to apply to the images
c_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])


# Load the dataset
dataset = {x: datasets.ImageFolder(root=r'C:/Users/aytem/Desktop/testgit/5820-Final-Project/data', transform= c_transform)
           for x in sets
}

dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=6, shuffle=True) for x in sets}

dataset_name = 'yes'
data_loader = dataloaders[dataset_name]

# Randomly select a batch from the data loader
images, labels = next(iter(data_loader))

# Print out the shape
print(images[0].shape)


class_labels = dataset[dataset_name].classes

# Display the images
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for i, ax in enumerate(axes.flatten()):
    image = images[i].permute(1, 2, 0)  # Change the tensor shape from (C, H, W) to (H, W, C)
    label = labels[i].item()
    class_name = class_labels[label]  # Get the class label name

    #ax.imshow(image, cmap='gray')
    ax.set_title(f"Label: {class_name}")
    ax.axis('off')

plt.tight_layout()
plt.show()


num_epochs = 25
learning_rate = 0.001
num_classes = len(class_labels)


# create the model still working on this part

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

# # Input and output numbers needs an adjustment based on picture size, eg. 32x32x3 should have 3 channels and 32 input
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
#         self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)

#         self.fc1 = nn.Linear(in_features=64*16, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=64)
#         self.fc3 = nn.Linear(in_features=64, out_features=num_classes)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.pool(F.relu(self.conv4(x)))

#         x = x.view(x.size(0), -1) # Flatten

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x

# train the model 
# test the model


