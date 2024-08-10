import numpy as np
import sys
import time
import os
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn


def train(model, device, train_loader, valid_loader, lossfun1, optimizer, scheduler2, epoch_number=15 , saved_model_dir = None):
  
  # Count the number of parameters in the model
  number_of_params = sum(p.numel() for p in model.parameters())
  print(f'Model_{model.patch_size}_{model.embed_dim}_{model.num_heads}_{model.num_encoders}_{model.img_size}_{model.Hard_Label_Distillation}.pt , number Of params = {number_of_params}')  # Printing the network architecture and number of parameters

  # Move the model to the specified device
  model.to(device)

  # Linear learning rate scheduler
  scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001 , total_iters=10)
  
  # If hard label distillation is enabled, use additional loss function and load pre-trained CNN model
  if model.Hard_Label_Distillation:
      lossfun2 = nn.CrossEntropyLoss()
      CNN_model  = torch.load(os.path.join(saved_model_dir, 'ResNet18_CIFAR10.pt'),map_location = device)
      
      # Freeze the parameters of the pre-trained CNN model
      for p in CNN_model.parameters():
          p.requires_grad = False
      CNN_model.eval()
      CNN_model.to(device)

  # Lists to store training and validation metrics
  loss_train_epoch = []
  acc_train_epoch = []
  loss_val_epoch = []
  acc_val_epoch = []
  training_time = 0
  
  # Loop over epochs
  for epoch in range(epoch_number):
      start_time = time.time()

      # Training phase
      model.train()
      batch_itr = 0
      for X, Y in train_loader:
          batch_itr += 1

          # Move the inputs and targets to GPU if available
          X = X.to(device)
          Y = Y.to(device)
          
          # Perform forward pass
          if model.Hard_Label_Distillation:
              yHat , yHat_HLD = model(X)
              with torch.no_grad():
                Hard_labels = CNN_model(X)
                Hard_labels = torch.nn.functional.softmax(Hard_labels,1)
              loss = 0.5*lossfun1(yHat, Y) + 0.5*lossfun2(yHat_HLD, Hard_labels)
              
          else:
              yHat = model(X)
              loss = lossfun1(yHat, Y)
          loss_train_epoch.append(loss.item())

          # Compute accuracy
          y_true = Y.detach().cpu()
          y_predict = np.argmax(yHat.detach().cpu(), 1)  # one-hot to normal labels
          accuracy = accuracy_score(y_true, y_predict)
          acc_train_epoch.append(accuracy)

          # Backpropagation and optimization
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # Print training progress
          
          sys.stdout.write('\r' + f'epoch = {epoch+1}, iter = {batch_itr}, Loss Train = {loss.item():.4f}, Accuracy = {accuracy:.4f}')
      # Adjust learning rate using the linear scheduler
      if epoch < 10:
          scheduler1.step()
      else:
          scheduler2.step()
          
      end_time = time.time() - start_time
      training_time += end_time
      
      # Validation phase
      model.eval()
      val_loss = 0.0
      correct = 0
      total = 0
      with torch.no_grad():
          for X_val, Y_val in valid_loader:
              X_val = X_val.to(device)
              Y_val = Y_val.to(device)
              
              # Forward pass during validation
              if model.Hard_Label_Distillation:
                  y_predict_val , yHat_HLD = model(X_val)
                  Hard_labels = CNN_model(X_val)
                  Hard_labels = torch.nn.functional.softmax(Hard_labels,1)
                  loss_val = lossfun1(y_predict_val, Y_val) + lossfun2(yHat_HLD, Hard_labels)
                  loss_val /= 2
              else:
                  y_predict_val = model(X_val)
                  loss_val = lossfun1(y_predict_val, Y_val)
              val_loss += loss_val.item()

              # Compute validation accuracy
              y_true_val = Y_val.detach().cpu()
              y_predict_val = torch.argmax(y_predict_val.detach().cpu(), 1)
              correct += (y_predict_val == y_true_val).sum().item()
              total += y_true_val.size(0)

      # Calculate average validation loss and accuracy
      avg_val_loss = val_loss / len(valid_loader)
      avg_val_accuracy = correct / total
      flag = avg_val_accuracy < 0.7
      loss_val_epoch.append(avg_val_loss)
      acc_val_epoch.append(avg_val_accuracy)

      # Print validation metrics
      sys.stdout.write(f'\n Time: {training_time:.4f} , Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}\n')

  # Plotting the training and validation loss
  plt.figure()
  plt.plot(loss_train_epoch, '-', color='blue', label='Train Loss')
  plt.plot(np.linspace(0, len(loss_train_epoch), len(loss_val_epoch)), loss_val_epoch, '-', color='red', label='Validation Loss')
  plt.xlabel('Iteration')
  plt.title('Loss')
  plt.legend()

  # Plotting the training and validation accuracy
  plt.figure()
  plt.plot(acc_train_epoch, '-', color='blue', label='Train Accuracy')
  plt.plot(np.linspace(0, len(acc_train_epoch), len(acc_val_epoch)), acc_val_epoch, '-', color='red', label='Validation Accuracy')
  plt.xlabel('Iteration')
  plt.title('Accuracy')
  plt.legend()

