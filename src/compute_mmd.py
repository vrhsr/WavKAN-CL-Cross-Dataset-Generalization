"""
MMD (Maximum Mean Discrepancy) Computation for Domain Alignment.

Quantitatively measures the distance between feature distributions of 
source (MIT-BIH) and target (PTB-XL) domains.

Metric: MMD with Gaussian RBF Kernel.
A lower MMD indicates better domain alignment.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from src.models.dann import DANN

def gaussian_kernel(x, y, sigma=1.0):
    """Computes the Gaussian RBF kernel matrix K(x, y)."""
    dist = torch.cdist(x, y)**2
    return torch.exp(-dist / (2 * sigma**2))

def compute_mmd(x, y, sigma=1.0):
    """Computes the Maximum Mean Discrepancy between two distributions x and y."""
    x_kernel = gaussian_kernel(x, x, sigma)
    y_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)
    
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

def extract_features(model, loader, device, model_name):
    """Extracts features before the final classifier layer using hooks."""
    features_list = []
    
    # Identify the feature layer
    if model_name == 'wavkan':
        # WavKAN: features after the last KAN layer before the classifier
        # forward(x) loop ends with: features = norm(F.silu(layer(features)))
        # classifier is self.classifier
        target_layer = model.norms[-1]
    elif model_name == 'resnet':
        target_layer = model.gap
    elif model_name == 'vit':
        target_layer = model.norm
    elif model_name == 'mlp':
        target_layer = model.net[-1] # Before classifier if sequential, but MLP has classifier as last layer of net?
        # Let's check SimpleMLP definition: self.net = nn.Sequential(...)
        # Last layer in SimpleMLP.net is the classifier. We want the layer BEFORE it.
        target_layer = model.net[-2] if len(model.net) > 1 else model.net[0]
    elif model_name == 'spline_kan':
        # SplineKAN has layers in nn.Sequential
        target_layer = model.layers[-2]
    elif model_name == 'dann':
        target_layer = model.feature_ext
    else:
        raise ValueError("Unknown model")

    extracted = []
    def hook(module, input, output):
        # Handle cases where output might be a tuple or have different shape
        if isinstance(output, tuple):
            output = output[0]
        # Flatten for MMD
        extracted.append(output.detach().cpu().view(output.size(0), -1))

    handle = target_layer.register_forward_hook(hook)
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc=f"Extracting {model_name}"):
            inputs = inputs.to(device).float()
            model(inputs)
            if len(extracted) > 1000: # Batch-wise extraction
                features_list.append(torch.cat(extracted, dim=0))
                extracted = []
                
    handle.remove()
    if extracted:
        features_list.append(torch.cat(extracted, dim=0))
        
    return torch.cat(features_list, dim=0)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    source_dataset = HarmonizedDataset(args.source_file)
    target_dataset = HarmonizedDataset(args.target_file)
    
    # Limit samples for MMD computation (it is O(N^2) memory-wise in kernel)
    n_samples = min(len(source_dataset), len(target_dataset), args.max_samples)
    
    # Use non-shuffled subset for consistency
    source_loader = DataLoader(torch.utils.data.Subset(source_dataset, range(n_samples)), batch_size=args.batch_size)
    target_loader = DataLoader(torch.utils.data.Subset(target_dataset, range(n_samples)), batch_size=args.batch_size)

    # 2. Initialize Model
    if args.model == 'wavkan':
        model = WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    elif args.model == 'resnet':
        model = ResNet1D(in_channels=1, num_classes=2).to(device)
    elif args.model == 'vit':
        model = ViT1D(seq_len=250, num_classes=2).to(device)
    elif args.model == 'mlp':
        model = SimpleMLP(input_dim=250, num_classes=2).to(device)
    elif args.model == 'spline_kan':
        model = SplineKANClassifier(input_dim=250, num_classes=2).to(device)
    elif args.model == 'dann':
        model = DANN(in_channels=1, num_classes=2, feature_dim=256).to(device)
    else:
        raise ValueError("Unknown model")

    # 3. Load Weights
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Computing raw feature MMD.")

    # 4. Extract Features
    source_feats = extract_features(model, source_loader, device, args.model)
    target_feats = extract_features(model, target_loader, device, args.model)
    
    # 5. Compute MMD (in smaller chunks if needed to avoid OOM)
    # RBF Sigma heuristic: median distance
    dist = torch.cdist(source_feats[:1000].to(device), target_feats[:1000].to(device))
    sigma = dist.median().item()
    print(f"Using RBF sigma: {sigma:.4f}")

    # Split into chunks of 1000 for kernel computation to avoid OOM
    mmd_scores = []
    chunk_size = 1000
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        s_chunk = source_feats[i:end].to(device)
        t_chunk = target_feats[i:end].to(device)
        mmd_scores.append(compute_mmd(s_chunk, t_chunk, sigma=sigma).item())

    avg_mmd = np.mean(mmd_scores)
    print(f"MMD for {args.model}: {avg_mmd:.6f}")

    # 6. Save result
    res_df = pd.DataFrame([{
        'model': args.model,
        'checkpoint': os.path.basename(args.checkpoint),
        'mmd': round(avg_mmd, 6),
        'sigma': round(sigma, 4)
    }])
    
    if os.path.exists(args.output):
        res_df.to_csv(args.output, mode='a', header=False, index=False)
    else:
        res_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--source_file', type=str, default='data/mitbih_processed.csv')
    parser.add_argument('--target_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--output', type=str, default='experiments/results_mmd.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    args = parser.parse_args()
    main(args)
