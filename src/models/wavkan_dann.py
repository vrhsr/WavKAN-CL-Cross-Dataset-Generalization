import torch
import torch.nn as nn
from src.models.dann import GradientReversalLayer

class DANNDomainDiscriminator(nn.Module):
    """Domain discriminator head with gradient reversal."""
    def __init__(self, feature_dim=64):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=1.0)
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),  # Binary: source=0 vs target=1
        )

    def forward(self, x):
        x = self.grl(x)
        return self.discriminator(x)

    def set_alpha(self, alpha):
        self.grl.alpha = alpha

class HybridDANN(nn.Module):
    """
    Wraps any baseline or KAN model inside a DANN architecture.
    Extracts features dynamically based on model_name.
    """
    def __init__(self, backbone, model_name, feature_dim=64):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.domain_discriminator = DANNDomainDiscriminator(feature_dim)
        
    def forward(self, x, alpha=1.0):
        self.domain_discriminator.set_alpha(alpha)
        features = self.backbone.extract_features(x)
        
        # Dynamically route features through the specific backbone's classifier
        if self.model_name in ['wavkan', 'spline_kan']:
            class_output = self.backbone.classifier(features)
        elif self.model_name == 'resnet':
            class_output = self.backbone.fc(features)
        elif self.model_name == 'vit':
            class_output = self.backbone.mlp_head(features)
        elif self.model_name == 'simple_mlp':
            class_output = self.backbone.net[-1](features)
        else:
            raise ValueError(f"Unknown model_name for HybridDANN: {self.model_name}")
            
        domain_output = self.domain_discriminator(features)
        return class_output, domain_output

    def predict(self, x):
        """Inference-only forward"""
        return self.backbone(x)
