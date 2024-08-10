import numpy as np
import torch
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

### Load Data
# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X  # Input data
        self.y = y  # Labels
        self.transform = transform  # Data transformation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # Extract the input data at the specified index
        y = self.y[idx]  # Extract the label at the specified index

        if self.transform:
            x = self.transform(x)  # Apply transformation if specified

        return x, y
    
class data_loader():
    def __init__(self, directory, data_type='Train', input_size = (32,32),  batch_size=256, validation_ratio=0.1):

        self.directory = directory  # Directory containing the dataset
        self.data_type = data_type  # Type of data: Train, Test
        self.input_size = input_size
        self.batch_size = batch_size  # Batch size for DataLoader
        self.validation_ratio = validation_ratio  # Validation split ratio

        # Convert the labels to categorical names
        cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']

        # Initialize LabelEncoder
        self.label_encoder = LabelEncoder()

        # Fit LabelEncoder on the training labels
        self.label_encoder.fit(cifar10_labels)

        # Define augmentations for training and testing

        # Ref: https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py 

        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def load(self):
        classes = os.listdir(self.directory)  # List of classes (subdirectories)
        images = []
        lbl = []

        # Iterate through each class directory
        for c in classes:
            path = os.path.join(self.directory, c)  # Path to class directory
            file_names = os.listdir(path)  # List of file names in the class directory
            for file_name in file_names:
                if file_name.endswith(".png"):  # Consider only PNG files
                    img = Image.open(os.path.join(path, file_name))  # Open image
                    images.append(np.array(img))  # Append image to list
                    lbl.append(c)  # Append label to list

        X = np.array(images)  # Convert images to numpy array
        Y = np.array(lbl)  # Convert labels to numpy array
        Y = self.label_encoder.transform(Y.ravel())  # Encode labels

        if self.data_type == 'Train':
            # Split data into training and validation sets
            X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=self.validation_ratio, stratify=Y)

            # Convert training and validation data to PyTorch tensors
            Y_train_tensor = torch.tensor(Y_train).long()
            Y_valid_tensor = torch.tensor(Y_valid).long()

            # Create datasets and data loaders for training and validation sets
            train_dataset = CustomDataset(X_train, Y_train_tensor, transform=self.transform_train)
            valid_dataset = CustomDataset(X_valid, Y_valid_tensor, transform=self.transform_test)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size)

            return train_loader, valid_loader

        if self.data_type == 'Test':
            # Convert test data to PyTorch tensors
            Y_test_tensor = torch.tensor(Y).long()

            # Create dataset and data loader for test set
            test_dataset = CustomDataset(X, Y_test_tensor, transform=self.transform_test)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            return test_loader
