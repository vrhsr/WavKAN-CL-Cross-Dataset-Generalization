"""
Wavelet Interpretability Analysis Script.

Visualizes the learned wavelet basis functions from a trained WavKAN model
to understand which morphological features (QRS, P, T) the model is focusing on.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from src.models.wavkan import WavKANClassifier

def plot_wavelet(ax, wavelet_type, scale, translation, color='blue', label=None):
    """Plots a single wavelet in the time domain."""
    t = np.linspace(-2, 2, 400)
    s = (t - translation) / (scale + 1e-8)
    
    if wavelet_type == 'mexican_hat':
        y = (1 - s**2) * np.exp(-0.5 * s**2)
    elif wavelet_type == 'morlet':
        y = np.cos(5 * s) * np.exp(-0.5 * s**2)
    else:
        y = np.zeros_like(t)
        
    ax.plot(t, y, color=color, alpha=0.6, label=label)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = WavKANClassifier(input_dim=250, num_classes=2, 
                             hidden_dim=args.hidden_dim, 
                             wavelet_type=args.wavelet_type, 
                             depth=args.depth).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    # 2. Extract Parameters from the first WaveletLinear layer
    # This layer processes the raw signal (or conv stem features)
    first_layer = model.layers[0]
    
    # scale = a_min + softplus(scale_raw)
    a_min = 0.001
    learned_scales = (a_min + F.softplus(first_layer.scale_raw)).detach().cpu().numpy()
    learned_translations = first_layer.translation.detach().cpu().numpy()
    
    print(f"Extracted {learned_scales.shape[0]} base functions from Layer 0.")
    print(f"Scale range: [{learned_scales.min():.4f}, {learned_scales.max():.4f}]")
    print(f"Translation range: [{learned_translations.min():.4f}, {learned_translations.max():.4f}]")

    # 3. Plot a representative sample of wavelets
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    
    # Selection: pick wavelets with largest scales (broad features like T-wave) 
    # and smallest scales (sharp features like QRS)
    flat_scales = learned_scales.flatten()
    flat_trans = learned_translations.flatten()
    
    indices = np.argsort(flat_scales)
    smallest_idx = indices[:5]
    largest_idx = indices[-5:]
    
    for idx in smallest_idx:
        plot_wavelet(axes, args.wavelet_type, flat_scales[idx], flat_trans[idx], color='red')
    
    for idx in largest_idx:
        plot_wavelet(axes, args.wavelet_type, flat_scales[idx], flat_trans[idx], color='green')

    axes.set_title(f"Learned Wavelet Basis Functions ({args.wavelet_type})\nRed: Sharp (high freq), Green: Broad (low freq)")
    axes.set_xlabel("Normalized Time")
    axes.set_ylabel("Amplitude")
    axes.grid(True)
    
    save_path = f"paper/plots/wavelet_interpretability_{args.wavelet_type}.png"
    os.makedirs('paper/plots', exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

    # 4. Frequency Analysis (Optional but helpful for the paper)
    # Estimate frequency bands captured
    # F_center = f_c / scale
    # Mexico Hat: f_c ~ 0.25 (peak frequency)
    center_freqs = 0.25 / (learned_scales + 1e-8)
    plt.figure(figsize=(8, 4))
    plt.hist(center_freqs.flatten(), bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of Learned Center Frequencies ({args.wavelet_type})")
    plt.xlabel("Frequency (arbitrary units)")
    plt.ylabel("Count")
    plt.savefig(f"paper/plots/wavelet_frequencies_{args.wavelet_type}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--wavelet_type', type=str, default='mexican_hat')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    args = parser.parse_args()
    main(args)
