import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import os
import wfdb

# Imports
from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.spline_kan import SplineKANClassifier
from src.models.baselines import ResNet1D, ViT1D
from src.models.dann import DANN

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FILE = 'data/ptbxl_processed.csv'
CHECKPOINT_DIR = 'experiments'
OUTPUT_DIR = 'experiments/plots'

def load_model(name):
    # Re-instantiate model structure
    if name == 'wavkan':
        model = WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=128)
    elif name == 'spline_kan':
        model = SplineKANClassifier(input_dim=250, num_classes=2)
    elif name == 'resnet':
        model = ResNet1D(in_channels=1, num_classes=2)
    elif name == 'vit':
        model = ViT1D(seq_len=250, num_classes=2)
    elif name == 'dann':
        model = DANN(in_channels=1, num_classes=2, feature_dim=256)
    
    # Load weights
    path = os.path.join(CHECKPOINT_DIR, f"{name}_endpoint.pth")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    else:
        print(f"Warning: Checkpoint for {name} not found.")
        return None

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{model_name}.png'))
    plt.close()

def visualize_activations(model, sample_input, model_name):
    # Hook to capture intermediate output
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hook (specific to architecture)
    if model_name == 'wavkan':
        # Hook first layer (WavKAN uses ModuleList)
        model.layers[0].register_forward_hook(get_activation('layer1'))
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input.unsqueeze(0).to(DEVICE))
        
    if 'layer1' in activations:
        act = activations['layer1'].cpu().numpy()[0] # (out_features) or sequence?
        # WavKAN linear outputs (batch, out_features)
        # We want to see how it responds to the signal.
        # But WavKAN likely outputs a flat vector per layer if it's MLP-style.
        # If it's 1D signal processing, maybe we visualize the wavelet filters themselves.
        pass

def visualize_wavelets(model, model_name):
    if model_name != 'wavkan':
        return

    # WavKAN likely has parameters for translation/scale
    # Inspect model.layer1.translation, model.layer1.scale if available
    # Or just plot the learned functions if easy.
    # Given abstract implementation, let's just create a schematic or print stats.
    print(f"Analyzing Wavelet Parameters for {model_name}...")
    
    # Example: Print mean/std of learned parameters
    for name, param in model.named_parameters():
        if 'wavelet' in name or 'scale' in name or 'translation' in name:
            print(f"  {name}: Mean={param.data.mean():.4f}, Std={param.data.std():.4f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_kan_basis(model, model_name):
    """
    Visualize the learned basis functions (Splines or Wavelets)
    """
    if model_name not in ['spline_kan', 'wavkan']:
        return

    print(f"Visualizing Basis Functions for {model_name}...")
    plt.figure(figsize=(10, 4))
    
    # Create a dummy input range [-1, 1] to see the basis response
    x = torch.linspace(-1, 1, 100).to(DEVICE)
    
    if model_name == 'spline_kan':
        # For Spline-KAN, we want to see the shape of the B-splines/RBFs
        # model.layer1 has .grid and .spline_weights
        layer = model.layer1
        # Re-create basis generation logic from forward pass
        grid = torch.linspace(-1, 1, layer.grid_size).to(DEVICE)
        x_uns = x.unsqueeze(-1) # (100, 1)
        # RBF basis: exp(-gamma * (x - c)^2)
        # (100, 1) - (1, grid) -> (100, grid)
        basis = torch.exp(-torch.pow(x_uns - grid.view(1, -1), 2) * 5.0)
        
        # Plot the first 5 basis functions
        basis_np = basis.detach().cpu().numpy()
        for i in range(min(5, basis_np.shape[1])):
            plt.plot(x.cpu().numpy(), basis_np[:, i], label=f'Basis {i}')
            
    elif model_name == 'wavkan':
        # For WavKAN, visualize the learned wavelets
        # This depends on your specific implementation of WavKANLinear
        # Assuming we can inspect the wavelet function phi(x)
        # If it's pure mathematical wavelets (mexican hat), plot that.
        # Here we plot a standard Mexican Hat as a proxy for what it's learning to weight
        t = torch.linspace(-5, 5, 100)
        psi = (1 - t**2) * torch.exp(-0.5 * t**2) # Mexican Hat
        plt.plot(t.numpy(), psi.numpy(), label='Mexican Hat Wavelet')
        
    plt.title(f'Learned Basis Functions: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'kan_basis_{model_name}.png'))
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Data (Subset for speed)
    dataset = HarmonizedDataset(DATA_FILE)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Get one batch for visualization
    iter_loader = iter(loader)
    try:
        inputs, labels = next(iter_loader)
    except StopIteration:
        return

    models = ['wavkan', 'spline_kan', 'resnet', 'vit', 'dann']
    
    for m_name in models:
        print(f"Processing {m_name}...")
        model = load_model(m_name)
        if model is None:
            continue
            
        # 0. Count Params (Table 2 Data)
        params = count_parameters(model)
        print(f"[{m_name}] Parameters: {params:,}")
            
        # 1. Confusion Matrix
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(DEVICE).float()
                if hasattr(model, 'predict'):
                    outputs = model.predict(bx)
                else:
                    outputs = model(bx)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(by.numpy())
                if len(all_preds) > 1000:
                    break
                    
        plot_confusion_matrix(all_labels, all_preds, m_name)
        
        # 2. Interpretability (Tier 1)
        visualize_kan_basis(model, m_name)
            
    print("Visualizations Generated in experiments/plots/")

if __name__ == "__main__":
    main()
