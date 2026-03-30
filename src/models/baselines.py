import torch
import torch.nn as nn
import math

# --- ResNet1D ---
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=5, seq_len=1000):
        super(ResNet1D, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = ResBlock1D(64, 64)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.layer3 = ResBlock1D(128, 256, stride=2)
        self.layer4 = ResBlock1D(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def extract_features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        return torch.flatten(x, 1)

# --- ViT-1D ---
class PatchEmbedding1D(nn.Module):
    def __init__(self, seq_len=1000, patch_size=50, in_channels=12, embed_dim=128):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 1, L)
        x = self.proj(x) # (B, E, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, E)
        return x

class ViT1D(nn.Module):
    def __init__(self, seq_len=1000, patch_size=50, num_classes=5, embed_dim=128, depth=4, heads=4, mlp_dim=256, in_channels=12):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(seq_len, patch_size, in_channels, embed_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
         # x: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        
        # Extract CLS token and classify
        x = x[:, 0]  # CLS token
        return self.mlp_head(x)

    def extract_features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        return x[:, 0]

# --- Simple MLP (Ablation Baseline) ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=1000, num_classes=5, hidden_dim=224, in_channels=12):
        super(SimpleMLP, self).__init__()
        # Designed to match WavKAN params (~114k)
        # 250*224 + 224 = 56k
        # 224*224 + 224 = 50k
        # Total ~106k
        flat_dim = input_dim * in_channels
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len) -> Flatten
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

    def extract_features(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        # return features before final Linear
        return self.net[:-1](x)

# --- InceptionTime ---
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=32):
        super().__init__()
        self.use_bottleneck = bottleneck > 0 and in_channels > 1
        bottleneck_channels = bottleneck if self.use_bottleneck else in_channels
        
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        else:
            self.bottleneck = nn.Identity()
            
        self.conv1 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=39, padding=19, bias=False)
        self.conv2 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=19, padding=9, bias=False)
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=9, padding=4, bias=False)
        
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        bottleneck = self.bottleneck(x)
        out1 = self.conv1(bottleneck)
        out2 = self.conv2(bottleneck)
        out3 = self.conv3(bottleneck)
        out4 = self.conv_pool(self.maxpool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return self.relu(self.bn(out))

class InceptionTime(nn.Module):
    def __init__(self, in_channels=12, num_classes=5, hidden_dim=32, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_in = in_channels
        for i in range(depth):
            self.blocks.append(InceptionBlock1D(current_in, hidden_dim))
            current_in = hidden_dim * 4
            
        self.shortcut = nn.ModuleList()
        current_in = in_channels
        for i in range(depth // 3):
            # A shortcut every 3 blocks
            self.shortcut.append(nn.Sequential(
                nn.Conv1d(current_in, hidden_dim * 4, 1, bias=False),
                nn.BatchNorm1d(hidden_dim * 4)
            ))
            current_in = hidden_dim * 4
            
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        
    def extract_features(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        res = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % 3 == 0:
                res = self.shortcut[i // 3](res)
                x = x + res
                x = torch.relu(x)
                res = x
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc(features)

# --- XResNet1D ---
class XResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * 4)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * 4)
            )
            
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return torch.relu(out)

class XResNet1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.layer1 = XResNetBlock(64, 64)
        self.layer2 = XResNetBlock(256, 128, stride=2)
        self.layer3 = XResNetBlock(512, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, num_classes)
        
    def extract_features(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.fc(self.extract_features(x))

# --- InceptionTime ---
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=32):
        super().__init__()
        self.use_bottleneck = bottleneck > 0 and in_channels > 1
        bottleneck_channels = bottleneck if self.use_bottleneck else in_channels
        
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        else:
            self.bottleneck = nn.Identity()
            
        self.conv1 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=39, padding=19, bias=False)
        self.conv2 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=19, padding=9, bias=False)
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=9, padding=4, bias=False)
        
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        bottleneck = self.bottleneck(x)
        out1 = self.conv1(bottleneck)
        out2 = self.conv2(bottleneck)
        out3 = self.conv3(bottleneck)
        out4 = self.conv_pool(self.maxpool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return self.relu(self.bn(out))

class InceptionTime(nn.Module):
    def __init__(self, in_channels=12, num_classes=5, hidden_dim=32, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_in = in_channels
        for i in range(depth):
            self.blocks.append(InceptionBlock1D(current_in, hidden_dim))
            current_in = hidden_dim * 4
            
        self.shortcut = nn.ModuleList()
        current_in = in_channels
        for i in range(depth // 3):
            # A shortcut every 3 blocks
            self.shortcut.append(nn.Sequential(
                nn.Conv1d(current_in, hidden_dim * 4, 1, bias=False),
                nn.BatchNorm1d(hidden_dim * 4)
            ))
            current_in = hidden_dim * 4
            
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        
    def extract_features(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        res = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % 3 == 0:
                res = self.shortcut[i // 3](res)
                x = x + res
                x = torch.relu(x)
                res = x
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.fc(features)

# --- XResNet1D ---
class XResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * 4)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * 4)
            )
            
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return torch.relu(out)

class XResNet1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        self.layer1 = XResNetBlock(64, 64)
        self.layer2 = XResNetBlock(256, 128, stride=2)
        self.layer3 = XResNetBlock(512, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, num_classes)
        
    def extract_features(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.fc(self.extract_features(x))
