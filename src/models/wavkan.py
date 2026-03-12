import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WaveletLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat', init_mode='random'):
        super(WaveletLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.init_mode = init_mode
        self.a_min = 0.001  # Minimum scale floor to prevent collapse
        
        # Learnable parameters for scaling and translation of wavelets
        # Shape: (out_features, in_features)
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        # Use scale_raw to enforce positivity/stability via softplus in forward
        self.scale_raw = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.uniform_(self.translation, -1, 1)
        
        if self.init_mode == 'clinical':
            # Initialize scales to capture specific clinical frequency bands
            # Based on 100Hz sampling:
            # QRS (~25Hz) -> scale ~0.006
            # T-wave (~5Hz) -> scale ~0.032
            # P-wave (~2Hz) -> scale ~0.080
            scales = torch.zeros_like(self.scale_raw)
            n_in = self.in_features
            # Split input features into three clinical bands
            qrs_split = n_in // 3
            twave_split = 2 * n_in // 3
            
            # Helper to invert softplus: x = log(exp(y - a_min) - 1)
            def inv_softplus(y):
                diff = y - self.a_min
                return math.log(math.exp(max(diff, 1e-6)) - 1)
            
            scales[:, :qrs_split] = inv_softplus(0.006)
            scales[:, qrs_split:twave_split] = inv_softplus(0.032)
            scales[:, twave_split:] = inv_softplus(0.080)
            self.scale_raw.data.copy_(scales)
        else:
            # Default random initialization (log-space uniform approx)
            # Starting scales between 0.1 and 1.0
            nn.init.uniform_(self.scale_raw, -2.0, 0.0)

    def forward(self, x):
        # x shape: (batch_size, in_features)
        # Expand x to match output dimensions: (batch, out, in)
        x_expanded = x.unsqueeze(1).expand(-1, self.out_features, -1)
        
        # Apply softplus to ensure scale > a_min
        scale = self.a_min + F.softplus(self.scale_raw)
        
        # Apply wavelet transform: psi((x - b) / a)
        s = (x_expanded - self.translation) / (scale + 1e-8)
        
        if self.wavelet_type == 'mexican_hat':
            # Mexican Hat: (1 - t^2) * exp(-t^2 / 2)
            wavelet = (1 - s**2) * torch.exp(-0.5 * s**2)
        elif self.wavelet_type == 'morlet':
            # Morlet real part: cos(5*t) * exp(-t^2 / 2)
            wavelet = torch.cos(5 * s) * torch.exp(-0.5 * s**2)
        else:
            raise ValueError("Unknown wavelet type")
            
        # Weighted sum of wavelet responses
        y = (self.weights * wavelet).sum(dim=2)
        
        return y


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
            
            # Block 2: capture broader waveform features (P-wave, T-wave)
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
        # x: (batch, 1, seq_len) or (batch, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.stem(x)  # (batch, 128, T)
        return features.view(features.size(0), -1)  # flatten


class WavKANClassifier(nn.Module):
    def __init__(self, input_dim=250, num_classes=2, hidden_dim=64, 
                 wavelet_type='mexican_hat', depth=3, use_conv_stem=True):
        super(WavKANClassifier, self).__init__()
        
        self.use_conv_stem = use_conv_stem
        
        if use_conv_stem:
            self.conv_stem = Conv1DStem(out_dim=hidden_dim)
            kan_input_dim = self.conv_stem.out_features
        else:
            self.conv_stem = None
            kan_input_dim = input_dim
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Input Layer
        self.layers.append(WaveletLinear(kan_input_dim, hidden_dim, wavelet_type=wavelet_type))
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
        if self.use_conv_stem and self.conv_stem is not None:
            # x: (batch, 1, seq_len) or (batch, seq_len)
            features = self.conv_stem(x)
        else:
            # Legacy: flatten directly
            features = x.view(x.size(0), -1)
        
        for layer, norm in zip(self.layers, self.norms):
            features = norm(F.silu(layer(features)))
        
        if contrastive:
            return self.projection_head(features)
            
        return self.classifier(features)
