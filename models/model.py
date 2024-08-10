import numpy as np
import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.split_dim = model_dim // num_heads

        # Linear transformations for Q, K, and V
        self.Linear_Q = nn.Linear(model_dim, model_dim)
        self.Linear_K = nn.Linear(model_dim, model_dim)
        self.Linear_V = nn.Linear(model_dim, model_dim)
        self.Linear_O = nn.Linear(model_dim, model_dim)

    def split_heads(self, x):
        batch_size, seq_length, model_dim = x.size()
        # Reshape the input tensor to split into multiple heads
        x = x.view(batch_size, seq_length, self.num_heads, self.split_dim)
        x = x.transpose(1, 2)  # Swap dimensions for matrix multiplication : (batch_size, num_heads, seq_length, split_dim)
        return x

    def forward(self, Q, K, V):
        # Split Q, K, and V into multiple heads
        Q = self.split_heads(self.Linear_Q(Q))  # (batch_size, num_heads, seq_length, split_dim)
        K = self.split_heads(self.Linear_K(K))  # (batch_size, num_heads, seq_length, split_dim)
        V = self.split_heads(self.Linear_V(V))  # (batch_size, num_heads, seq_length, split_dim)

        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.split_dim, dtype=torch.float32))
        # Apply softmax to get attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # Calculate output using attention probabilities and V
        output = torch.matmul(attn_probs, V)

        output = output.transpose(1, 2)  # (batch_size, seq_length, num_heads, split_dim)
        output = output.reshape(output.size(0), -1, self.model_dim)  # (batch_size, seq_length, model_dim)

        # Linear transformation for the output
        output = self.Linear_O(output)

        return output, attn_probs



class Transformer_encoder(nn.Module):
  def __init__(self,embed_dim,num_heads,FFN_hidden):
    super().__init__()
    # Layer normalization for the input
    self.LayerNorm1 = nn.LayerNorm(embed_dim)
    # Multi-head attention mechanism
    self.attention = MultiheadAttention(embed_dim , num_heads)
    # Layer normalization for the attention output
    self.LayerNorm2 = nn.LayerNorm(embed_dim)
    # Feedforward neural network
    self.FFN = nn.Sequential(
        nn.Linear(embed_dim, FFN_hidden),
        nn.GELU(),  # Activation function
        nn.Linear(FFN_hidden, embed_dim),
    )

  def forward(self,X):
    # Layer normalization for the input
    out = self.LayerNorm1(X)
    # Multi-head attention mechanism
    attn_output, attn_output_weights = self.attention(out,out,out)
    # Residual connection and layer normalization
    X = attn_output + X
    # Feedforward neural network
    out = self.LayerNorm2(X)
    # Residual connection
    out = self.FFN(out)

    return out + X , attn_output_weights
  

class Vision_Transformer(nn.Module):
  def __init__(self,patch_size, embed_dim, num_heads ,num_encoders, FFN_hidden = 128, class_num = 10 , img_size = 32 , Hard_Label_Distillation = False):
    super().__init__()

    self.embed_dim = embed_dim  # Dimensionality of the token embeddings
    self.num_heads = num_heads  # Number of attention heads
    self.FFN_hidden = FFN_hidden  # Hidden layer size in the feedforward network
    self.num_encoders = num_encoders  # Number of transformer encoder layers
    self.Hard_Label_Distillation = Hard_Label_Distillation  # Flag for hard label distillation


    # Handling different types of input values
    if (type(patch_size) == list) | (type(patch_size) == tuple):
          patch_size = patch_size[0]
            
    self.patch_size = patch_size # Patch size for dividing the image into patches
    
    if (type(img_size) == list) | (type(img_size) == tuple):
      img_size = img_size[0]
    
    self.img_size = img_size # Size of the input image
        
    self.patch_num = int(img_size/patch_size) # Number of patches in one dimension

    # Learnable token embeddings
    self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    if self.Hard_Label_Distillation:
        self.dis_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    # Position embeddings for each patch
    if self.Hard_Label_Distillation:
      self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num**2 + 2 , embed_dim))
    else:
      self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num**2 + 1 , embed_dim))

    # Transformer encoder layers    
    self.transformer_encoders = nn.Sequential()
    for i in range(self.num_encoders):
       self.transformer_encoders.add_module(f"encoder{i}" , Transformer_encoder(self.embed_dim, self.num_heads, self.FFN_hidden))

    # Layer normalization
    self.LayerNorm1 = nn.LayerNorm(embed_dim)

    # Final classification head
    self.mlp_head = nn.Linear(embed_dim , class_num)

    if self.Hard_Label_Distillation:
        self.dist_head = nn.Linear(embed_dim , class_num)
    
    # Dropout layer for regularization
    self.dropout = nn.Dropout(0.1)

    # Convolutional layer for patching the input image
    self.patcher =  nn.Conv2d(3 , self.embed_dim, kernel_size=patch_size, stride=patch_size)


  def forward(self, X , Visualize = False):
    # Patching the input image and flattening the patches
    output = self.patcher(X).flatten(2).transpose(1, 2)
    output = torch.cat((self.cls_token.expand(output.shape[0], -1, -1), output), 1)
    
    if self.Hard_Label_Distillation:
      output = torch.cat((output, self.dis_token.expand(output.shape[0], -1, -1)) , 1)
    
    # Adding positional embeddings to the patches
    output = output +  self.pos_embedding
    
    # Forward pass through each transformer encoder layer
    attn_output_weights_total = []
    for encoder in self.transformer_encoders:
        output , attn_output_weights = encoder(output)
        attn_output_weights_total.append(attn_output_weights.detach().cpu().numpy())
    
    # Layer normalization
    output = self.LayerNorm1(output)
    # Applying the final classification head
    output1 = self.mlp_head(output[:,0]) # Output for the CLS token
    
    if self.Hard_Label_Distillation:
      output2 = self.dist_head(output[:,-1]) # Output for the DIS token
   
    # Return outputs based on the mode (with or without visualization)
    if Visualize:
      if self.Hard_Label_Distillation:
        return output1 , output2 , np.array(attn_output_weights_total)
      else:
        return output1 , np.array(attn_output_weights_total)
    else:
      if self.Hard_Label_Distillation:
        return output1 , output2
      else:
        return output1