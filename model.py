import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the multimodal model with ResNet18 backbone
class CNNMultimodalModel(nn.Module):
    def __init__(self, backbone_name='resnet18', text_feature_dim=2, hidden_dim=512, num_classes=2, dropout_rate=0.3):
        super().__init__()

        # Image encoder using ResNet18
        self.image_encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        # Get feature dimension from the backbone
        if backbone_name == 'resnet18':
            self.image_feature_dim = 512  # ResNet18 feature dimension
        elif 'mobilenetv2' in backbone_name:
            self.image_feature_dim = 1280  # MobileNetV2 feature dimension
        elif 'squeezenet' in backbone_name:
            self.image_feature_dim = 512  # SqueezeNet feature dimension
        elif 'shufflenet_v2' in backbone_name:
            self.image_feature_dim = 1024  # ShuffleNetV2 feature dimension
        else:
            # For other backbones, determine feature dim dynamically
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224)
                dummy_output = self.image_encoder(dummy_input)
                self.image_feature_dim = dummy_output.shape[1]

        # Adapt first convolutional layer for grayscale images (1 channel)
        if 'resnet' in backbone_name:
            # For ResNet models
            orig_conv = self.image_encoder.conv1
            new_conv = nn.Conv2d(1, orig_conv.out_channels,
                                kernel_size=orig_conv.kernel_size,
                                stride=orig_conv.stride,
                                padding=orig_conv.padding,
                                bias=orig_conv.bias is not None)
            # Initialize with mean across RGB channels
            with torch.no_grad():
                if hasattr(orig_conv, 'weight'):
                    new_conv.weight[:, 0:1, :, :] = orig_conv.weight.mean(dim=1, keepdim=True)
            self.image_encoder.conv1 = new_conv
        elif 'mobilenet' in backbone_name:
            # For MobileNet models
            orig_conv = self.image_encoder.conv_stem
            new_conv = nn.Conv2d(1, orig_conv.out_channels,
                                kernel_size=orig_conv.kernel_size,
                                stride=orig_conv.stride,
                                padding=orig_conv.padding,
                                bias=orig_conv.bias is not None)
            # Initialize with mean across RGB channels
            with torch.no_grad():
                if hasattr(orig_conv, 'weight'):
                    new_conv.weight[:, 0:1, :, :] = orig_conv.weight.mean(dim=1, keepdim=True)
            self.image_encoder.conv_stem = new_conv

        # Text projection network with batch normalization
        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        # Image projection network with batch normalization
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        # Final classification layers with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images, text_features):
        # Process images
        image_features = self.image_encoder(images)
        image_features = self.image_projection(image_features)

        # Process text features
        text_features = self.text_projection(text_features)

        # Reshape for attention if needed
        if len(image_features.shape) == 2:
            image_features = image_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Cross-modal attention
        # Using image as query, text as key/value
        attn_output, _ = self.cross_attn(
            query=image_features,
            key=text_features,
            value=text_features
        )

        # Combine features
        combined_features = torch.cat([
            image_features.squeeze(1),
            attn_output.squeeze(1)
        ], dim=1)

        # Classification
        output = self.classifier(combined_features)

        return output