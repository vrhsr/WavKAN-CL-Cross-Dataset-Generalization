import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WaveletLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WaveletLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        
        # Learnable parameters for scaling and translation of wavelets
        # Shape: (out_features, in_features)
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.uniform_(self.translation, -1, 1)
        nn.init.uniform_(self.scale, 0.1, 1)

    def forward(self, x):
        # x shape: (batch_size, in_features)
        # We need to compute the wavelet response for every input-output pair
        
        # Expand x to match output dimensions: (batch, out, in)
        x_expanded = x.unsqueeze(1).expand(-1, self.out_features, -1)
        
        # Apply wavelet transform: psi((x - b) / a)
        # s = (x - translation) / scale
        s = (x_expanded - self.translation) / self.scale
        
        if self.wavelet_type == 'mexican_hat':
            # Mexican Hat: (1 - t^2) * exp(-t^2 / 2)
            wavelet = (1 - s**2) * torch.exp(-0.5 * s**2)
        elif self.wavelet_type == 'morlet':
            # Morlet real part approx: cos(5*t) * exp(-t^2 / 2)
            wavelet = torch.cos(5 * s) * torch.exp(-0.5 * s**2)
        else:
            raise ValueError("Unknown wavelet type")
            
        # Weighted sum of wavelet responses
        # y = sum(w * psi(...))
        y = (self.weights * wavelet).sum(dim=2)
        
        return y

class WavKANClassifier(nn.Module):
    def __init__(self, input_dim=1000, num_classes=2, hidden_dim=64, wavelet_type='mexican_hat', depth=3):
        super(WavKANClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Input Layer
        self.layers.append(WaveletLinear(input_dim, hidden_dim, wavelet_type=wavelet_type))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden Layers
        for _ in range(depth - 2):
            self.layers.append(WaveletLinear(hidden_dim, hidden_dim, wavelet_type=wavelet_type))
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        # Output/Bottleneck Layer
        self.layers.append(WaveletLinear(hidden_dim, hidden_dim, wavelet_type=wavelet_type))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Contrastive Head (Projection)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Classification Head (Binary)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, contrastive=False):
        # x: (batch, seq_len) -> Flattened 1D signal
        x = x.view(x.size(0), -1)
        
        features = x
        for layer, norm in zip(self.layers, self.norms):
            features = norm(F.silu(layer(features)))
        
        if contrastive:
            return self.projection_head(features)
            
        return self.classifier(features)
