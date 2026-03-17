import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from src.models.wavkan import WavKANClassifier
from src.dataset import HarmonizedDataset
from torch.utils.data import DataLoader

def get_wavelet_activations(model, signal):
    """
    Extract wavelet scale activations for a given signal.
    Returns heatmap of scale magnitudes per time step.
    """
    model.eval()
    with torch.no_grad():
        # Get features before classification
        if model.use_conv_stem and model.conv_stem is not None:
            features = model.conv_stem(signal.unsqueeze(0))
        else:
            features = signal.view(1, -1)
        
        # Get activations from first layer
        layer = model.layers[0]  # WaveletLinear
        x = features
        
        # Compute wavelet responses
        scale = F.softplus(layer.scale_raw) + layer.a_min
        translation = layer.translation
        
        # For each output feature, compute wavelet response across input
        activations = []
        for out_idx in range(min(layer.out_features, 10)):  # Limit for visualization
            responses = []
            for in_idx in range(min(layer.in_features, 50)):  # Limit
                s = scale[out_idx, in_idx]
                t = translation[out_idx, in_idx]
                wavelet = layer._compute_wavelet((x[0, in_idx] - t) / s) * layer.weights[out_idx, in_idx]
                responses.append(wavelet.item())
            activations.append(responses)
        
        return np.array(activations)  # (out_features, in_features)


def generate_heatmaps(model_path, data_file, output_dir, num_samples=10):
    """
    Generate wavelet activation heatmaps for correct/incorrect predictions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=64)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Load data
    dataset = HarmonizedDataset(data_file)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    correct_activations = []
    incorrect_activations = []
    
    with torch.no_grad():
        for i, (signal, label) in enumerate(loader):
            if i >= num_samples:
                break
            
            signal = signal.to(device)
            output = model(signal)
            pred = torch.argmax(output, dim=1).item()
            
            # Get activations
            activations = get_wavelet_activations(model, signal.squeeze(0))
            
            if pred == label.item():
                correct_activations.append(activations)
            else:
                incorrect_activations.append(activations)
    
    # Average activations
    if correct_activations:
        avg_correct = np.mean(correct_activations, axis=0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_correct, cmap='viridis', cbar=True)
        plt.title('Average Wavelet Activations - Correct Predictions')
        plt.xlabel('Input Features (Time Steps)')
        plt.ylabel('Output Features')
        plt.savefig(os.path.join(output_dir, 'correct_predictions_heatmap.png'))
        plt.close()
    
    if incorrect_activations:
        avg_incorrect = np.mean(incorrect_activations, axis=0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_incorrect, cmap='viridis', cbar=True)
        plt.title('Average Wavelet Activations - Incorrect Predictions')
        plt.xlabel('Input Features (Time Steps)')
        plt.ylabel('Output Features')
        plt.savefig(os.path.join(output_dir, 'incorrect_predictions_heatmap.png'))
        plt.close()
    
    print(f"Heatmaps saved to {output_dir}")


def faithfulness_deletion(model, signal, baseline='zero', n_steps=10):
    """
    Deletion faithfulness: Remove most important features and measure drop in confidence.
    """
    model.eval()
    with torch.no_grad():
        original_output = model(signal.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0, 1].item()  # Assuming class 1 is positive
        
        # Get feature importance (absolute activations)
        activations = get_wavelet_activations(model, signal.squeeze(0))
        importance = np.abs(activations).sum(axis=0)  # Sum over output features
        
        # Sort features by importance
        sorted_indices = np.argsort(importance)[::-1]
        
        probs = [original_prob]
        for i in range(1, n_steps + 1):
            # Remove top i% features
            n_remove = int(len(sorted_indices) * i / n_steps)
            remove_indices = sorted_indices[:n_remove]
            
            modified_signal = signal.clone()
            if baseline == 'zero':
                modified_signal[remove_indices] = 0
            elif baseline == 'mean':
                modified_signal[remove_indices] = signal.mean()
            
            output = model(modified_signal.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0, 1].item()
            probs.append(prob)
        
        # Area under curve (higher AUC = more faithful)
        auc = np.trapz(probs, dx=1/n_steps)
        return auc


def generate_faithfulness_report(model_path, data_file, output_dir, num_samples=50):
    """
    Generate faithfulness metrics for interpretability evaluation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=64)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    dataset = HarmonizedDataset(data_file)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    aucs_zero = []
    aucs_mean = []
    
    with torch.no_grad():
        for i, (signal, label) in enumerate(loader):
            if i >= num_samples:
                break
            
            signal = signal.to(device).squeeze(0)
            
            auc_zero = faithfulness_deletion(model, signal, baseline='zero')
            auc_mean = faithfulness_deletion(model, signal, baseline='mean')
            
            aucs_zero.append(auc_zero)
            aucs_mean.append(auc_mean)
    
    # Summary stats
    results = {
        'auc_zero_mean': np.mean(aucs_zero),
        'auc_zero_std': np.std(aucs_zero),
        'auc_mean_mean': np.mean(aucs_mean),
        'auc_mean_std': np.std(aucs_mean),
        'num_samples': len(aucs_zero)
    }
    
    import json
    with open(os.path.join(output_dir, 'faithfulness_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Faithfulness metrics saved to {output_dir}/faithfulness_metrics.json")
    print(f"AUC Zero: {results['auc_zero_mean']:.3f} ± {results['auc_zero_std']:.3f}")
    print(f"AUC Mean: {results['auc_mean_mean']:.3f} ± {results['auc_mean_std']:.3f}")


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

    # Generate heatmaps if data file provided
    if args.data_file:
        generate_heatmaps(args.checkpoint, args.data_file, args.output_dir, args.num_samples)
        generate_faithfulness_report(args.checkpoint, args.data_file, args.output_dir, args.num_samples)

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
    parser.add_argument('--data_file', type=str, help='Path to test data for heatmaps')
    parser.add_argument('--output_dir', type=str, default='experiments/wavelet_analysis', help='Output directory for heatmaps')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples for heatmap analysis')
    args = parser.parse_args()
    main(args)
