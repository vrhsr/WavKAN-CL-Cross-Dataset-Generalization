"""
Domain-Adversarial Neural Network (DANN) for 1D ECG signals.

Implements Ganin et al. (2016) "Domain-Adversarial Training of Neural Networks"
with a Gradient Reversal Layer (GRL) for unsupervised domain adaptation.

Architecture:
  - Feature Extractor: 1D CNN backbone (Conv1D + ResBlocks)
  - Label Classifier: FC head for source-domain classification
  - Domain Discriminator: FC head with GRL for domain-invariant features
"""
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Reverses gradients during backpropagation (Ganin & Lempitsky, 2015)."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for the gradient reversal function."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha):
        self.alpha = alpha


class DANNFeatureExtractor(nn.Module):
    """1D CNN feature extractor for ECG signals.
    
    Designed to be roughly parameter-comparable to the other baselines.
    Uses Conv1D blocks with batch normalization and residual connections.
    """
    def __init__(self, in_channels=12, feature_dim=256):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1: 1 -> 64 channels
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Block 2: 64 -> 128 channels
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Block 3: 128 -> 256 channels
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # Block 4: 256 -> feature_dim
            nn.Conv1d(256, feature_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Dynamic stem output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 1000) # Default seq_len
            self.feature_dim = self.conv_blocks(dummy).view(1, -1).shape[1]

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        return x.view(x.size(0), -1)  # (B, feature_dim)


class DANNClassifier(nn.Module):
    """Label classifier head."""
    def __init__(self, feature_dim=256, num_classes=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class DANNDomainDiscriminator(nn.Module):
    """Domain discriminator head with gradient reversal."""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=1.0)
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # Binary: source vs target domain
        )

    def forward(self, x):
        x = self.grl(x)
        return self.discriminator(x)

    def set_alpha(self, alpha):
        self.grl.set_alpha(alpha)


class DANN(nn.Module):
    """Complete Domain-Adversarial Neural Network.
    
    Combines feature extractor, label classifier, and domain discriminator.
    The GRL ensures the feature extractor learns domain-invariant representations.
    
    Args:
        in_channels: Number of input channels (1 for single-lead ECG)
        num_classes: Number of classification labels
        feature_dim: Dimensionality of the feature representation
    """
    def __init__(self, backbone='dann', in_channels=12, num_classes=5, feature_dim=256, **kwargs):
        super().__init__()
        self.backbone = backbone
        
        if backbone == 'wavkan':
            from src.models.wavkan import WavKANClassifier
            self.feature_extractor = WavKANClassifier(in_channels=in_channels, hidden_dim=64)
            real_feature_dim = 64
        elif backbone == 'wavkan_multiscale':
            from src.models.wavkan_multiscale import MultiScaleWavKANClassifier
            self.feature_extractor = MultiScaleWavKANClassifier(in_channels=in_channels, hidden_dim=64)
            real_feature_dim = 64
        elif backbone == 'resnet':
            from src.models.baselines import ResNet1D
            self.feature_extractor = ResNet1D(in_channels=in_channels, num_classes=num_classes)
            real_feature_dim = 1024
        elif backbone == 'inception':
            from src.models.baselines import InceptionTime
            self.feature_extractor = InceptionTime(in_channels=in_channels, num_classes=num_classes)
            real_feature_dim = 128
        else:
            self.feature_extractor = DANNFeatureExtractor(in_channels, feature_dim)
            real_feature_dim = self.feature_extractor.feature_dim
            
        self.label_classifier = DANNClassifier(real_feature_dim, num_classes)
        self.domain_discriminator = DANNDomainDiscriminator(real_feature_dim)

    def forward(self, x, alpha=1.0):
        """
        Forward pass through all components.
        
        Args:
            x: Input tensor (B, seq_len) or (B, 1, seq_len)
            alpha: GRL scaling factor (scheduled during training)
            
        Returns:
            class_output: Classification logits (B, num_classes)
            domain_output: Domain prediction logits (B, 1)
        """
        self.domain_discriminator.set_alpha(alpha)
        
        if hasattr(self.feature_extractor, 'extract_features'):
            features = self.feature_extractor.extract_features(x)
        else:
            features = self.feature_extractor(x)
            
        class_output = self.label_classifier(features)
        domain_output = self.domain_discriminator(features)
        return class_output, domain_output

    def predict(self, x):
        """Inference-only forward (no domain discriminator)."""
        if hasattr(self.feature_extractor, 'extract_features'):
            features = self.feature_extractor.extract_features(x)
        else:
            features = self.feature_extractor(x)
        return self.label_classifier(features)
