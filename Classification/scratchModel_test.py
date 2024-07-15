import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple
from predictions import pred_and_plot_image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the PatchEmbedding class
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int=3, patch_size:int=16, embedding_dim:int=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

# Define the MultiheadSelfAttentionBlock class
class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim:int=768, num_heads:int=12, attn_dropout:float=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return attn_output

# Define the MLPBlock class
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim:int=768, mlp_size:int=3072, dropout:float=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

# Define the TransformerEncoderBlock class
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim:int=768, num_heads:int=12, mlp_size:int=3072, mlp_dropout:float=0.1, attn_dropout:float=0):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

# Define the ViT class
class ViT(nn.Module):
    def __init__(self, img_size:int=224, in_channels:int=3, patch_size:int=16, num_transformer_layers:int=12, embedding_dim:int=768, mlp_size:int=3072, num_heads:int=12, attn_dropout:float=0, mlp_dropout:float=0.1, embedding_dropout:float=0.1, num_classes:int=2):
        super().__init__()
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
        self.num_patches = (img_size * img_size) // patch_size**2
        self.class_token = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.pos_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)
        self.encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_size=mlp_size, mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.heads = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.pos_embedding + x
        x = self.embedding_dropout(x)
        x = self.encoder(x)
        x = self.ln(x[:, 0])
        x = self.heads(x)
        return x



# Load the model
model_path = r"C:\Users\bhara\OneDrive\Desktop\Temp\Model_trials\scratch.pth"
model = ViT(num_classes=3)  # Adjust the number of classes as needed
# model.load_state_dict(torch.load(model_path, map_location=device))
pretrained_vit_state_dict = torch.load(model_path,map_location=device)

new_pretrained_vit_state_dict = {}
for k, v in pretrained_vit_state_dict.items():
    # print(k,"     --->     ",v)
    if 'head' in k:
        # print("replaced   ",k )
        k = k.replace('head', 'heads')  # Rename keys related to the classifier head
    new_pretrained_vit_state_dict[k] = v

model.load_state_dict(new_pretrained_vit_state_dict,strict=False)

model.to(device)
model.eval()

# Define class names
class_names = ["benign", "malignant","normal"]

# Image path
image_path = r"C:\Users\bhara\OneDrive\Desktop\Temp\augmented-1300\augmentedDATASET\test\malignant\malignant1081.png"

# Predict and plot
pred_and_plot_image(model, class_names, image_path)
