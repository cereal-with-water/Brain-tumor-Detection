import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.io import read_image
import os


import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])

# Load the dataset
dataset_yes = datasets.ImageFolder(root=r'C:\Users\aytem\Desktop\testgit\5820-Final-Project\brain_tumor_dataset\yes',
                                   transform=transform)

dataset_no = datasets.ImageFolder(root=r'C:\Users\aytem\Desktop\testgit\5820-Final-Project\brain_tumor_dataset\no',
                                  transform=transform)

# Display a few images from the "yes" class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, (image, _) in enumerate(dataset_yes):
    if i >= 10:  # Display 10 images
        break
    ax = axes[i // 5, i % 5]
    ax.imshow(image.permute(1, 2, 0))  # PyTorch tensors are (C, H, W), so we permute dimensions for display
    ax.axis('off')
    ax.set_title('Yes')
plt.show()

# Display a few images from the "no" class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, (image, _) in enumerate(dataset_no):
    if i >= 10:  # Display 10 images
        break
    ax = axes[i // 5, i % 5]
    ax.imshow(image.permute(1, 2, 0))  # PyTorch tensors are (C, H, W), so we permute dimensions for display
    ax.axis('off')
    ax.set_title('No')
plt.show()


print(os.listdir(r'C:\Users\aytem\Desktop\testgit\5820-Final-Project\brain_tumor_dataset\yes'))
print(os.listdir(r'C:\Users\aytem\Desktop\testgit\5820-Final-Project\brain_tumor_dataset\no'))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #reading data
# data_dir = './data' 
# sets = ['test', 'train', 'valid']

# compared_transform = transforms.Compose([
#     transforms.Grayscale(), 
#     transforms.Resize((128, 128)), 
#     transforms.ToTensor()
#     ])

# image_datasets = {
#     x: datasets.ImageFolder(os.path.join(data_dir, x), transform=composed_transform)}
# for x in sets
# }

# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=6, shuffle = True)}
    

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