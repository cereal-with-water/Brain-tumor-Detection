Brain Tumor Detection
Overview
This repository contains code for a deep learning model to detect brain tumors using MRI images. The model is trained on a dataset consisting of MRI images labeled with tumor presence or absence.

Dataset
The dataset used for training and evaluation consists of MRI images categorized into two classes:

"Yes" indicating the presence of a brain tumor.
"No" indicating the absence of a brain tumor.
The dataset is divided into training and validation sets to train and evaluate the model's performance.

Model Architecture
The model architecture is designed using PyTorch, a popular deep learning framework. It consists of convolutional neural network layers followed by fully connected layers, with a final sigmoid activation to predict the probability of tumor presence.

Training
The model is trained using the Adam optimizer with binary cross-entropy loss. Hyperparameters such as learning rate and batch size are tuned to optimize the model's performance. Training is performed over multiple epochs, with periodic evaluation on the validation set to monitor model progress.

Results
The training and validation results are logged after each epoch, including loss and accuracy metrics. These results provide insights into the model's training progress and performance on unseen data.

Requirements
Python 3.x
PyTorch
torchvision
numpy
matplotlib
