from matplotlib import pyplot as plt
import numpy as np
import torch

def Layer_visualization(model,device,data,num_images_to_show,layers_to_show):

  # Visualize sample images from the dataset
  Number_of_patches = model.patch_num

  # Define the mean and std used in normalization
  mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
  std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

  # Function to denormalize a tensor
  def denormalize(tensor, mean, std):
      tensor = tensor * std + mean
      return tensor

  for i, (X, y) in enumerate(data):
      for j in range(num_images_to_show):
          plt.figure(figsize = (10,11))
          plt.subplot(3,3,5)
          org_img = denormalize(X[j], mean, std).permute(1, 2, 0)
          plt.imshow(org_img)
          rows, cols = Number_of_patches , Number_of_patches
          height, width, _ = org_img.shape
          for p in range(1, rows):
              plt.axhline(p * height / rows, color='white', linewidth=1 , alpha = 0.3)
          for q in range(1, cols):
              plt.axvline(q * width / cols, color='white', linewidth=1, alpha = 0.3)
          plt.axis('off')
          plt.title('Image')

          # Forward pass through the model
          if model.Hard_Label_Distillation:
              output , _ , attn_output_weights_total = model(X[j].unsqueeze(0).to(device),Visualize = True)
              attn_output_weights_total = attn_output_weights_total[:,:,:-1,:-1]
          else:
              output , attn_output_weights_total = model(X[j].unsqueeze(0).to(device),Visualize = True)
          prd = torch.argmax(output)

          plt.subplot(3,3,4)
          plt.text(0.5, 0.5, f'Prediction = {prd}',
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=15)
          plt.axis('off')
          
          plt.subplot(3,3,6)
          plt.text(0.5, 0.5, f'Label = {y[j]}',
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=15)
          plt.axis('off')


          # Plot attention maps for each layer

          plt.subplot(3,3,1)
          attn_weights = attn_output_weights_total[layers_to_show[0]].squeeze()[0][0][1:].reshape(Number_of_patches,Number_of_patches)
          plt.imshow(attn_weights)
          plt.axis('off')
          plt.title('L1 (H0)')

          plt.subplot(3,3,7)
          attn_weights = np.mean(attn_output_weights_total[layers_to_show[0]].squeeze(),0)[0][1:].reshape(Number_of_patches,Number_of_patches)
          plt.imshow(attn_weights)
          plt.axis('off')
          plt.title('L1 (Mean)')

          plt.subplot(3,3,2)
          attn_weights = attn_output_weights_total[layers_to_show[1]].squeeze()[0][0][1:].reshape(Number_of_patches,Number_of_patches)
          plt.imshow(attn_weights)
          plt.axis('off')
          plt.title('L2 (H0)')

          plt.subplot(3,3,8)
          attn_weights = np.mean(attn_output_weights_total[layers_to_show[1]].squeeze(),0)[0][1:].reshape(Number_of_patches,Number_of_patches)
          plt.imshow(attn_weights)
          plt.axis('off')
          plt.title('L2 (Mean)')

          plt.subplot(3,3,3)
          attn_weights = attn_output_weights_total[layers_to_show[2]].squeeze()[0][0][1:].reshape(Number_of_patches,Number_of_patches)
          plt.imshow(attn_weights)
          plt.axis('off')
          plt.title('L4 (H0)')

          plt.subplot(3,3,9)
          attn_weights = np.mean(attn_output_weights_total[layers_to_show[2]].squeeze(),0)[0][1:].reshape(Number_of_patches,Number_of_patches)
          plt.imshow(attn_weights)
          plt.axis('off')
          plt.title('L4 (Mean)')

          print()

      break 


