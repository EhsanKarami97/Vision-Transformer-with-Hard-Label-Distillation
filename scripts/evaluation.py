import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import torch

def evaluation(model, device,lossfun, test_loader):
    # Initialize lists to store true labels and predictions
    all_y_true = []
    all_y_predict = []
    loss_test_epoch = 0
    itr = 0
    # Loop through mini-batches
    for X, Y in test_loader:
        itr += 1
        X = X.to(device)
        Y = Y.to(device)  # Move Y to device if necessary

        # Perform forward pass
        with torch.no_grad():
            model.eval()
            if model.Hard_Label_Distillation:
                yHat, _ = model(X)
            else:
                yHat = model(X)
        
        # Compute loss
        loss_test = lossfun(yHat, Y)
        loss_test_epoch += (loss_test.item())
                 
        # Store true labels and predictions
        all_y_true.append(Y.detach().cpu().numpy())
        all_y_predict.append(np.argmax(yHat.detach().cpu().numpy(), axis=1))

    # Concatenate all mini-batch results
    all_y_true = np.concatenate(all_y_true)
    all_y_predict = np.concatenate(all_y_predict)

    # Calculate metrics
    accuracy = accuracy_score(all_y_true, all_y_predict)
    f1 = f1_score(all_y_true, all_y_predict, average='macro')
    precision = precision_score(all_y_true, all_y_predict, average='macro')
    recall = recall_score(all_y_true, all_y_predict, average='macro')

     # Print metrics
    print(f'loss : {loss_test_epoch/itr:.4f}')
    print(f'accuracy : {accuracy:.4f}')
    print(f'f1 : {f1:.4f}')
    print(f'precision : {precision:.4f}')
    print(f'recall : {recall:.4f}')
    print()

    # Confusion matrix
    confusion_mat = confusion_matrix(all_y_true, all_y_predict)
    disp = ConfusionMatrixDisplay(confusion_mat)
    disp.plot()

