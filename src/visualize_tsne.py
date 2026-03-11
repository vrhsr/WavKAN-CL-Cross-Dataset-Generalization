"""
t-SNE / UMAP Visualization of Latent Representations.
Shows whether WavKAN learns domain-invariant features vs baselines.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
import os
import argparse

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from src.models.dann import DANN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'experiments/plots'


def extract_features(model, loader, device, model_name, max_samples=2000):
    """Extract penultimate layer features from a model."""
    model.eval()
    features_list = []
    labels_list = []
    
    # Register hook to capture features before classifier
    hook_features = {}
    
    def get_hook(name):
        def hook_fn(module, input, output):
            hook_features[name] = input[0].detach()  # input to classifier = features
        return hook_fn
    
    # Register hook on the classifier/fc layer
    if hasattr(model, 'classifier'):
        handle = model.classifier.register_forward_hook(get_hook('features'))
    elif hasattr(model, 'fc'):
        handle = model.fc.register_forward_hook(get_hook('features'))
    elif hasattr(model, 'mlp_head'):
        handle = model.mlp_head.register_forward_hook(get_hook('features'))
    else:
        print(f"Cannot find classifier layer for {model_name}")
        return None, None
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device).float()
            
            # Forward pass
            out = model(inputs)
            # DANN returns (class_out, domain_out), others return just out
            
            if 'features' in hook_features:
                feat = hook_features['features'].cpu().numpy()
                features_list.append(feat)
                labels_list.append(labels.numpy())
            
            if sum(len(f) for f in features_list) >= max_samples:
                break
    
    handle.remove()
    
    if not features_list:
        return None, None
    
    features = np.concatenate(features_list, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]
    return features, labels


def plot_tsne_single(features, labels, dataset_labels, model_name, save_path):
    """Create t-SNE plot for a single model, colored by class and shaped by dataset."""
    if features is None:
        return
    
    print(f"  Running t-SNE for {model_name} ({features.shape[0]} samples)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings = tsne.fit_transform(features)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Colored by class (Normal vs Abnormal)
    ax = axes[0]
    colors = ['#2ecc71', '#e74c3c']
    class_names = ['Normal', 'Abnormal']
    for cls_idx in [0, 1]:
        mask = labels == cls_idx
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   c=colors[cls_idx], label=class_names[cls_idx],
                   alpha=0.5, s=10)
    ax.set_title(f'{model_name} — Colored by Class')
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Plot 2: Colored by dataset (Source vs Target)
    ax = axes[1]
    ds_colors = ['#3498db', '#e67e22']
    ds_names = ['MIT-BIH (Source)', 'PTB-XL (Target)']
    for ds_idx in [0, 1]:
        mask = dataset_labels == ds_idx
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=ds_colors[ds_idx], label=ds_names[ds_idx],
                   alpha=0.5, s=10)
    ax.set_title(f'{model_name} — Colored by Domain')
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    plt.suptitle(f'Latent Space Visualization: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def load_model(name, device):
    """Re-instantiate and load model weights."""
    if name == 'wavkan':
        model = WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=128)
    elif name == 'spline_kan':
        model = SplineKANClassifier(input_dim=250, num_classes=2)
    elif name == 'resnet':
        model = ResNet1D(in_channels=1, num_classes=2)
    elif name == 'vit':
        model = ViT1D(seq_len=250, num_classes=2)
    elif name == 'mlp':
        model = SimpleMLP(input_dim=250, num_classes=2)
    elif name == 'dann':
        model = DANN(in_channels=1, num_classes=2, feature_dim=256)
    else:
        raise ValueError(f"Unknown model: {name}")
    
    path = f"experiments/{name}_endpoint.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        return model
    else:
        print(f"Warning: Checkpoint for {name} not found at {path}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading datasets...")
    mit_dataset = HarmonizedDataset('data/mitbih_processed.csv')
    ptb_dataset = HarmonizedDataset('data/ptbxl_processed.csv')
    
    # Sample subsets for visualization (1000 each)
    n_per_domain = 1000
    
    np.random.seed(42)
    mit_indices = np.random.choice(len(mit_dataset), min(n_per_domain, len(mit_dataset)), replace=False)
    ptb_indices = np.random.choice(len(ptb_dataset), min(n_per_domain, len(ptb_dataset)), replace=False)
    
    mit_subset = Subset(mit_dataset, mit_indices)
    ptb_subset = Subset(ptb_dataset, ptb_indices)
    
    mit_loader = DataLoader(mit_subset, batch_size=256, shuffle=False)
    ptb_loader = DataLoader(ptb_subset, batch_size=256, shuffle=False)
    
    models = ['wavkan', 'spline_kan', 'resnet', 'vit', 'dann']
    
    for model_name in models:
        print(f"\nProcessing {model_name}...")
        model = load_model(model_name, DEVICE)
        if model is None:
            continue
        
        # Extract features from both domains
        mit_features, mit_labels = extract_features(model, mit_loader, DEVICE, model_name, max_samples=n_per_domain)
        ptb_features, ptb_labels = extract_features(model, ptb_loader, DEVICE, model_name, max_samples=n_per_domain)
        
        if mit_features is None or ptb_features is None:
            print(f"  Skipping {model_name} — could not extract features")
            continue
        
        # Combine
        all_features = np.concatenate([mit_features, ptb_features], axis=0)
        all_labels = np.concatenate([mit_labels, ptb_labels], axis=0)
        dataset_labels = np.concatenate([
            np.zeros(len(mit_features)),  # 0 = MIT-BIH
            np.ones(len(ptb_features))    # 1 = PTB-XL
        ])
        
        save_path = os.path.join(OUTPUT_DIR, f'tsne_{model_name}.png')
        plot_tsne_single(all_features, all_labels, dataset_labels, model_name, save_path)
    
    # Create comparative grid
    print("\nGenerating comparison grid...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models):
        img_path = os.path.join(OUTPUT_DIR, f'tsne_{model_name}.png')
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(model_name, fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle('Latent Space Comparison Across Architectures', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Done. All t-SNE plots saved to experiments/plots/")


if __name__ == "__main__":
    main()
