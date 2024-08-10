# Import necessary libraries
import yaml 
import time  
import os  
import sys  
from PIL import Image 
import re

import numpy as np  
import pandas as pd  
from matplotlib import pyplot as plt  

import torch  
import torch.nn as nn  
from torch.utils.data import TensorDataset, DataLoader, Dataset  
from torchvision import transforms  
import torchvision

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder 

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(parent_dir, 'data'))  # Append data directory to system path
sys.path.append(os.path.join(parent_dir, 'models'))  # Append models directory to system path
sys.path.append(os.path.join(parent_dir, 'config'))  # Append config directory to system path
sys.path.append(os.path.join(parent_dir, 'utils'))  # Append utils directory to system path

# Import necessary modules from other directories
from data_loader import data_loader  # Custom data loader
from model import Vision_Transformer, Transformer_encoder, MultiheadAttention
from train import train  # Custom training function
from evaluation import evaluation  # Custom evaluation function
from Layer_visualization import Layer_visualization 
from VIT_b_16_visualization import VIT_b_16_visualization  # Custom layer visualization function

# Load configuration settings from YAML file
with open(os.path.join(parent_dir, 'config', 'config.yml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Define directories for training and testing data
train_directory = os.path.join(parent_dir, 'data', 'dataset', 'train')
test_directory = os.path.join(parent_dir, 'data', 'dataset', 'test')

print('Data is loading ... !')

# Initialize data loaders for training and testing data
input_size = config['Input_size']
input_size = tuple([input_size,input_size])
batch_size = config['Batch_size']
validation_ratio = config['Validation_ratio']

train_loader = data_loader(train_directory, input_size = input_size, data_type='Train', batch_size=batch_size, validation_ratio=validation_ratio)
train_data_loader, valid_data_loader = train_loader.load()

test_loader = data_loader(test_directory,input_size = input_size, batch_size=batch_size, data_type='Test')
test_data_loader = test_loader.load()

print('Data is loaded !')
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration settings for training
learning_rate = config['learning_rate']
stp_size = config['stp_size']
gama = config['gama']
epochs = config['epochs']

# Configuration settings for the neural network
patch_size= config['patch_size'] 
embed_dim = config['embed_dim']
num_heads = config['num_heads']
num_encoders  = config['num_encoders']
Hard_Label_Distillation = config['Hard_Label_Distillation']

# Define loss function
loss_function = nn.CrossEntropyLoss()

# Load pre-trained model or initialize new one
if config['Load_Trained_Model']:

    ModelPath = f'Model_{patch_size}_{embed_dim}_{num_heads}_{num_encoders}_{input_size[0]}_{Hard_Label_Distillation}.pt'

    print(ModelPath)
    PATH = os.path.join(parent_dir, 'models', 'saved_models', ModelPath)
    model = torch.load(PATH , map_location = device) 

else:
        #Initialize neural network model, loss function, optimizer, scheduler, and device
    model = Vision_Transformer(patch_size = patch_size, embed_dim = embed_dim , num_heads = num_heads, num_encoders = num_encoders, img_size = input_size, Hard_Label_Distillation = Hard_Label_Distillation)

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stp_size, gamma=gama)
    train(model, device, train_data_loader, valid_data_loader, loss_function, optimizer, scheduler, epoch_number=epochs , saved_model_dir = os.path.join(parent_dir, 'models', 'saved_models'))

print()

#Evaluate the model on test data
print('Model Performance on test set')
evaluation(model, device,loss_function, test_data_loader)
if not(config['Hard_Label_Distillation']):
    Layer_visualization(model,device,test_data_loader,num_images_to_show = 2, layers_to_show=[0,1,3])
    

# Display any generated plots
plt.show()
