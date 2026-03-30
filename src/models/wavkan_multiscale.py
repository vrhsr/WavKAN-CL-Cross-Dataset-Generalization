import torch
import torch.nn as nn
from src.models.wavkan import WavKANClassifier

class MultiScaleWavKANClassifier(nn.Module):
    """
    Multi-Scale WavKAN Architecture.
    
    Processes ECG signals through three parallel WavKAN branches, each
    constrained to a specific wavelet scale (frequency) range:
      - High Frequency (HF): Fine details (e.g., QRS complex) => scale_min=0.001, scale_max=0.01
      - Mid Frequency (MF): Medium details (e.g., T-wave)  => scale_min=0.01, scale_max=0.05
      - Low Frequency (LF): Broad features (e.g., P-wave, respiratory wander) => scale_min=0.05, scale_max=1.0
      
    Features are fused using a learned attention mechanism.
    """
    def __init__(self, input_dim=1000, num_classes=5, hidden_dim=64, 
                 wavelet_type='mexican_hat', depth=3, use_conv_stem=True, in_channels=12):
        super(MultiScaleWavKANClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. High Frequency Branch (Sharper features)
        self.branch_hf = WavKANClassifier(
            input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim,
            wavelet_type=wavelet_type, depth=depth, use_conv_stem=use_conv_stem,
            in_channels=in_channels, scale_min=0.001, scale_max=0.01
        )
        
        # 2. Mid Frequency Branch (Standard features)
        self.branch_mf = WavKANClassifier(
            input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim,
            wavelet_type=wavelet_type, depth=depth, use_conv_stem=use_conv_stem,
            in_channels=in_channels, scale_min=0.01, scale_max=0.05
        )
        
        # 3. Low Frequency Branch (Broad features)
        self.branch_lf = WavKANClassifier(
            input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim,
            wavelet_type=wavelet_type, depth=depth, use_conv_stem=use_conv_stem,
            in_channels=in_channels, scale_min=0.05, scale_max=1.0
        )
        
        # Attention mechanism to learn branch importance dynamically
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.last_attn_weights = None

    def extract_features(self, x):
        feat_hf = self.branch_hf.extract_features(x)
        feat_mf = self.branch_mf.extract_features(x)
        feat_lf = self.branch_lf.extract_features(x)
        
        stacked = torch.stack([feat_hf, feat_mf, feat_lf], dim=1) # (B, 3, D)
        concat = torch.cat([feat_hf, feat_mf, feat_lf], dim=1)    # (B, 3*D)
        
        attn_weights = self.attention(concat) # (B, 3)
        self.last_attn_weights = attn_weights.detach()
        
        # Attention-weighted fusion
        fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1) # (B, D)
        return fused

    def forward(self, x):
        fused_features = self.extract_features(x)
        return self.classifier(fused_features)

    def get_scale_attention_weights(self):
        """
        Returns the branch attention weights for the most recent forward pass.
        Useful for interpretability: (B, 3) [HF_weight, MF_weight, LF_weight]
        """
        return self.last_attn_weights
