import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SplineLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        """
        B-Spline Linear Layer for KAN using RBF approximation.
        """
        super(SplineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Spline coefficients (weights)
        self.spline_weights = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        
        # Base weights (residual connection like SiLU)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.uniform_(self.spline_weights, -0.1, 0.1)

    def forward(self, x):
        # Base activation (SiLU)
        base_output = F.silu(x) 
        base_out = F.linear(base_output, self.base_weight)
        
        # Spline (RBF approximation)
        grid = torch.linspace(-1, 1, self.grid_size).to(x.device)
        x_uns = x.unsqueeze(-1)
        
        # RBF: exp(-gamma * (x - c)^2)
        basis = torch.exp(-torch.pow(x_uns - grid, 2) * 5.0)  # (batch, in, grid)
        
        # Einsum: b i g, o i g -> b o
        spline_out = torch.einsum('big, oig -> bo', basis, self.spline_weights)
        
        return base_out + spline_out


class Conv1DStem(nn.Module):
    """Lightweight 1D convolutional feature extractor that preserves temporal structure."""
    def __init__(self, out_dim=64):
        super(Conv1DStem, self).__init__()
        self.stem = nn.Sequential(
            # Block 1: capture local morphology (QRS ~8-12 samples at 100Hz)
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),  # 250 -> 125
            
            # Block 2: capture broader waveform features
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),  # 125 -> 62
            
            # Block 3: abstract temporal features
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(out_dim // 4),  # -> out_dim//4 time steps
        )
        self.out_features = 128 * (out_dim // 4)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.stem(x)
        return features.view(features.size(0), -1)


class SplineKANClassifier(nn.Module):
    def __init__(self, input_dim=250, num_classes=2, hidden_dim=64, use_conv_stem=True):
        super(SplineKANClassifier, self).__init__()
        
        self.use_conv_stem = use_conv_stem
        
        if use_conv_stem:
            self.conv_stem = Conv1DStem(out_dim=hidden_dim)
            kan_input_dim = self.conv_stem.out_features
        else:
            self.conv_stem = None
            kan_input_dim = input_dim
        
        self.layer1 = SplineLinear(kan_input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.layer2 = SplineLinear(hidden_dim, hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        
        self.layer3 = SplineLinear(hidden_dim * 2, hidden_dim) 
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Contrastive Head (Projection)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, contrastive=False):
        if self.use_conv_stem and self.conv_stem is not None:
            x = self.conv_stem(x)
        else:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
        x = self.norm1(self.layer1(x)) 
        x = self.norm2(self.layer2(x))
        features = self.norm3(self.layer3(x))
        
        if contrastive:
            return self.projection_head(features)
            
        return self.classifier(features)
