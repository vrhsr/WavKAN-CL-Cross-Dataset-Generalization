"""
t-SNE / UMAP Visualization of Latent Representations.
Shows whether WavKAN learns domain-invariant features vs baselines.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Subset
import os
import argparse

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from src.models.dann import DANN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'paper/plots'


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
    elif hasattr(model, 'label_classifier'):
        handle = model.label_classifier.register_forward_hook(get_hook('features'))
    elif hasattr(model, 'net') and isinstance(model.net, nn.Sequential):
        handle = model.net[-1].register_forward_hook(get_hook('features'))
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
    
    embedding_path = save_path.replace('.png', '.npy')
    # Always recompute if we want the new perplexity, but assuming we delete the old files.
    if os.path.exists(embedding_path):
        embeddings = np.load(embedding_path)
    else:
        print(f"  Running t-SNE for {model_name} ({features.shape[0]} samples)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=2000)
        embeddings = tsne.fit_transform(features)
        np.save(embedding_path, embeddings)
        
    score = silhouette_score(embeddings, labels)
    print(f"  {model_name} Silhouette Score: {score:.3f}")
    
    # Save score to text file auxiliary
    with open(save_path.replace('.png', '.txt'), 'w') as f:
        f.write(str(score))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Colored by class (Normal vs Abnormal)
    ax = axes[0]
    colors = ['#2ecc71', '#e74c3c']
    class_names = ['Normal', 'Abnormal']
    for cls_idx in [0, 1]:
        mask = labels == cls_idx
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                   c=colors[cls_idx], label=class_names[cls_idx],
                   alpha=0.5, s=8)
    ax.set_title(f'Colored by Class')
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Add silhouette score box
    ax.text(0.05, 0.95, f'Sil={score:.3f}', transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    # Record limits to share with the domain plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Plot 2: Colored by dataset (Source vs Target)
    ax = axes[1]
    ds_colors = ['#3498db', '#e67e22']
    ds_names = ['MIT-BIH (Source)', 'PTB-XL (Target)']
    for ds_idx in [0, 1]:
        mask = dataset_labels == ds_idx
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=ds_colors[ds_idx], label=ds_names[ds_idx],
                   alpha=0.5, s=8)
    ax.set_title(f'Colored by Domain')
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return score


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
    
    models = ['wavkan', 'spline_kan', 'resnet', 'vit', 'dann', 'mlp']
    
    for model_name in models:
        print(f"\nProcessing {model_name}...")
        model = load_model(model_name, DEVICE)
        if model is None:
            continue
        
        # Extract features from both domains
        res_mit = extract_features(model, mit_loader, DEVICE, model_name, max_samples=n_per_domain)
        res_ptb = extract_features(model, ptb_loader, DEVICE, model_name, max_samples=n_per_domain)
        
        if res_mit[0] is None or res_ptb[0] is None:
            print(f"  Skipping {model_name} — could not extract features")
            continue
            
        mit_features, mit_labels = res_mit
        ptb_features, ptb_labels = res_ptb
        
        # Combine
        all_features = np.concatenate([mit_features, ptb_features], axis=0)
        all_labels = np.concatenate([mit_labels, ptb_labels], axis=0)
        dataset_labels = np.concatenate([
            np.zeros(len(mit_features)),  # 0 = MIT-BIH
            np.ones(len(ptb_features))    # 1 = PTB-XL
        ])
        
        save_path = os.path.join(OUTPUT_DIR, f'tsne_{model_name}.png')
        if not os.path.exists(save_path):
            plot_tsne_single(all_features, all_labels, dataset_labels, model_name, save_path)
        else:
            print(f"  Plot already exists for {model_name}, skipping computation.")
    
    # Create comparative grid
    print("\nGenerating comparison grid...")
    
    # Pretty names mapping
    pretty_names = {
        'wavkan': 'WavKAN',
        'spline_kan': 'Spline-KAN',
        'resnet': 'ResNet-1D',
        'vit': 'ViT-1D',
        'dann': 'DANN (Domain Adversarial)',
        'mlp': 'SimpleMLP'
    }
    
    # Figure 1: Main Architectures (2x2 grid)
    main_models = ['wavkan', 'spline_kan', 'resnet', 'vit']
    fig_main, axes_main = plt.subplots(2, 2, figsize=(20, 10))
    axes_main = axes_main.flatten()
    
    for idx, model_name in enumerate(main_models):
        img_path = os.path.join(OUTPUT_DIR, f'tsne_{model_name}.png')
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axes_main[idx].imshow(img)
            name_str = pretty_names.get(model_name, model_name)
            axes_main[idx].set_title(name_str, fontsize=18, fontweight='bold', pad=10)
        axes_main[idx].axis('off')
        
    plt.suptitle('Latent Space Comparison: Primary Architectures', fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_main.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Baselines (1x2 grid)
    print("\nGenerating baseline comparison grid...")
    baseline_models = ['dann', 'mlp']
    fig_base, axes_base = plt.subplots(1, 2, figsize=(20, 5))
    
    for idx, model_name in enumerate(baseline_models):
        img_path = os.path.join(OUTPUT_DIR, f'tsne_{model_name}.png')
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            axes_base[idx].imshow(img)
            name_str = pretty_names.get(model_name, model_name)
            axes_base[idx].set_title(name_str, fontsize=18, fontweight='bold', pad=10)
        axes_base[idx].axis('off')
        
    plt.suptitle('Domain Alignment and Baseline Analysis', fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_dann_mlp.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Done. All t-SNE plot grids saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
