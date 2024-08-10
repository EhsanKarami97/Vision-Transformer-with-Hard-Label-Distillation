from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

def VIT_b_16_visualization(model,device,data,num_images_to_show,layers_to_show):


# Freeze all the parameters of the model to prevent them from being updated during training
    for p in model.parameters():
        p.requires_grad = False

    # Initialize a list to store attention weights
    attn_output_weights_total = []

    # Define a hook function to extract attention weights
    def get_attention_weights(module, input, output):
        # Initialize a MultiheadAttention layer
        multihead_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)

        # Compute the attention output and weights
        attn_output, attn_output_weights = multihead_attn(output, output, output, need_weights=True, average_attn_weights=False)

        # Detach and append the attention weights to the list
        attn_output_weights_total.append(attn_output_weights.detach())

    # Set the model to evaluation mode
    model.eval()

    # Register forward hooks to specified MultiheadAttention layers in the encoder
    for i in range(len(model.encoder.layers)):
        # Select layers 4, 8, and 12 to register hooks
        if i + 1 in layers_to_show:
            layer = model.encoder.layers[i]
            # Register the hook function to the layer's first layer normalization component
            layer.ln_1.register_forward_hook(get_attention_weights)

    # Visualize sample images from the dataset
    Number_of_patches = 14

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

            input = X[j]
            org_img = denormalize(input, mean, std).permute(1, 2, 0)
            plt.imshow(org_img)
            rows, cols = Number_of_patches , Number_of_patches
            height, width, _ = org_img.shape
            for i in range(1, rows):
                plt.axhline(i * height / rows, color='white', linewidth=1 , alpha = 0.3)
            for j in range(1, cols):
                plt.axvline(j * width / cols, color='white', linewidth=1, alpha = 0.3)
            plt.axis('off')
            plt.title('Image')

            # Forward pass through the model
            attn_output_weights_total = []
            output = model(input.unsqueeze(0))

            # Plot attention maps for each layer
            plt.subplot(3,3,1)
            attn_weights = attn_output_weights_total[0].squeeze()[0][0][1:].reshape(Number_of_patches,Number_of_patches)
            plt.imshow(attn_weights)
            plt.axis('off')
            plt.title('L4 (H0)')

            plt.subplot(3,3,7)
            attn_weights = torch.mean(attn_output_weights_total[0].squeeze(),0)[0][1:].reshape(Number_of_patches,Number_of_patches)
            plt.imshow(attn_weights)
            plt.axis('off')
            plt.title('4 (Mean)')

            plt.subplot(3,3,2)
            attn_weights = attn_output_weights_total[1].squeeze()[0][0][1:].reshape(Number_of_patches,Number_of_patches)
            plt.imshow(attn_weights)
            plt.axis('off')
            plt.title('L8 (H0)')

            plt.subplot(3,3,8)
            attn_weights = torch.mean(attn_output_weights_total[1].squeeze(),0)[0][1:].reshape(Number_of_patches,Number_of_patches)
            plt.imshow(attn_weights)
            plt.axis('off')
            plt.title('L8 (Mean)')

            plt.subplot(3,3,3)
            attn_weights = attn_output_weights_total[2].squeeze()[0][0][1:].reshape(Number_of_patches,Number_of_patches)
            plt.imshow(attn_weights)
            plt.axis('off')
            plt.title('L12 (H0)')

            plt.subplot(3,3,9)
            attn_weights = torch.mean(attn_output_weights_total[2].squeeze(),0)[0][1:].reshape(Number_of_patches,Number_of_patches)
            plt.imshow(attn_weights)
            plt.axis('off')
            plt.title('L12 (Mean)')

            print()

        break 


