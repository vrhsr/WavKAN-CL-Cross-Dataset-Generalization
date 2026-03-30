import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from scipy.signal import find_peaks

from src.models.wavkan import WavKANClassifier
from src.dataset import HarmonizedDataset
from torch.utils.data import DataLoader

def get_wavelet_activations(model, signal):
    model.eval()
    with torch.no_grad():
        if model.use_conv_stem and model.conv_stem is not None:
            features = model.conv_stem(signal.unsqueeze(0))
        else:
            features = signal.view(1, -1)
        
        layer = model.layers[0]
        x = features
        
        scale = layer.scale_min + (layer.scale_max - layer.scale_min) * torch.sigmoid(layer.scale_raw)
        translation = layer.translation
        
        activations = []
        for out_idx in range(min(layer.out_features, 10)):
            responses = []
            for in_idx in range(min(layer.in_features, 50)):
                s = scale[out_idx, in_idx]
                t = translation[out_idx, in_idx]
                z = (x[0, in_idx] - t) / (s + 1e-8)
                wavelet = layer._compute_wavelet(z) * layer.weights[out_idx, in_idx]
                responses.append(wavelet.item())
            activations.append(responses)
        
        return np.array(activations)


def generate_heatmaps(model_path, data_file, output_dir, num_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WavKANClassifier(input_dim=1000, num_classes=5, hidden_dim=64, in_channels=12)
    model.load_state_dict(torch.load(model_path)['model_state_dict'] if 'model_state_dict' in torch.load(model_path) else torch.load(model_path))
    model.to(device)
    model.eval()
    
    dataset = HarmonizedDataset(data_file)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    correct_activations = []
    incorrect_activations = []
    
    with torch.no_grad():
        for i, (signal, label) in enumerate(loader):
            if i >= num_samples: break
            
            signal = signal.to(device).float()
            output = model(signal)
            pred = torch.argmax(output, dim=1).item()
            true_cls = torch.argmax(label, dim=1).item()
            
            activations = get_wavelet_activations(model, signal.squeeze(0))
            
            if pred == true_cls:
                correct_activations.append(activations)
            else:
                incorrect_activations.append(activations)
    
    if correct_activations:
        avg_correct = np.mean(correct_activations, axis=0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_correct, cmap='viridis', cbar=True)
        plt.title('Average Wavelet Activations - Correct Predictions')
        plt.savefig(os.path.join(output_dir, 'correct_predictions_heatmap.png'))
        plt.close()
    
    if incorrect_activations:
        avg_incorrect = np.mean(incorrect_activations, axis=0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_incorrect, cmap='viridis', cbar=True)
        plt.title('Average Wavelet Activations - Incorrect Predictions')
        plt.savefig(os.path.join(output_dir, 'incorrect_predictions_heatmap.png'))
        plt.close()


def faithfulness_deletion(model, signal, baseline='zero', n_steps=10):
    model.eval()
    with torch.no_grad():
        original_output = model(signal.unsqueeze(0))
        original_prob = torch.sigmoid(original_output)[0, 0].item()
        
        activations = get_wavelet_activations(model, signal.squeeze(0))
        importance = np.abs(activations).sum(axis=0)
        sorted_indices = np.argsort(importance)[::-1]
        
        probs = [original_prob]
        for i in range(1, n_steps + 1):
            n_remove = int(len(sorted_indices) * i / n_steps)
            remove_indices = sorted_indices[:n_remove]
            
            modified_signal = signal.clone()
            # Approximation across channels:
            if baseline == 'zero': modified_signal[:, remove_indices] = 0
            elif baseline == 'mean': modified_signal[:, remove_indices] = signal.mean()
            
            output = model(modified_signal.unsqueeze(0))
            prob = torch.sigmoid(output)[0, 0].item()
            probs.append(prob)
            
        return np.trapz(probs, dx=1/n_steps)

def faithfulness_insertion(model, signal, baseline='zero', n_steps=10):
    model.eval()
    with torch.no_grad():
        activations = get_wavelet_activations(model, signal.squeeze(0))
        importance = np.abs(activations).sum(axis=0)
        sorted_indices = np.argsort(importance)[::-1]
        
        modified_signal = torch.zeros_like(signal) if baseline == 'zero' else torch.ones_like(signal) * signal.mean()
        
        probs = [torch.sigmoid(model(modified_signal.unsqueeze(0)))[0, 0].item()]
        
        for i in range(1, n_steps + 1):
            n_insert = int(len(sorted_indices) * i / n_steps)
            insert_indices = sorted_indices[:n_insert]
            
            modified_signal[:, insert_indices] = signal[:, insert_indices]
            
            output = model(modified_signal.unsqueeze(0))
            prob = torch.sigmoid(output)[0, 0].item()
            probs.append(prob)
            
        return np.trapz(probs, dx=1/n_steps)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None: class_idx = torch.argmax(output, dim=1).item()
        
        score = output[0, class_idx]
        score.backward()
        
        weights = torch.mean(self.gradients, dim=[0, 2], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2], mode='linear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

def qrs_iou_score(signal, saliency_map, fs=100):
    signal_np = signal.cpu().numpy().flatten()
    peaks, _ = find_peaks(signal_np, distance=int(0.3*fs), height=np.std(signal_np))
    qrs_mask = np.zeros_like(signal_np)
    for p in peaks:
        qrs_mask[max(0, p-int(0.05*fs)):min(len(signal_np), p+int(0.05*fs))] = 1
        
    saliency_mask = (saliency_map > np.percentile(saliency_map, 80)).astype(int)
    
    intersection = np.logical_and(qrs_mask, saliency_mask).sum()
    union = np.logical_or(qrs_mask, saliency_mask).sum()
    return intersection / union if union > 0 else 0.0

def generate_faithfulness_report(model_path, data_file, output_dir, num_samples=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WavKANClassifier(input_dim=1000, num_classes=5, hidden_dim=64, in_channels=12)
    model.load_state_dict(torch.load(model_path)['model_state_dict'] if 'model_state_dict' in torch.load(model_path) else torch.load(model_path))
    model.to(device)
    model.eval()
    
    dataset = HarmonizedDataset(data_file)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    metrics = {
        'audc_zero': [], 'audc_mean': [],
        'auic_zero': [], 'auic_mean': [],
        'qrs_iou': []
    }
    
    with torch.no_grad():
        for i, (signal, label) in enumerate(loader):
            if i >= num_samples: break
            signal = signal.to(device).float().squeeze(0)
            
            metrics['audc_zero'].append(faithfulness_deletion(model, signal, 'zero'))
            metrics['audc_mean'].append(faithfulness_deletion(model, signal, 'mean'))
            metrics['auic_zero'].append(faithfulness_insertion(model, signal, 'zero'))
            metrics['auic_mean'].append(faithfulness_insertion(model, signal, 'mean'))
            
            # Approximate QRS IOU from raw absolute importance map
            acts = get_wavelet_activations(model, signal)
            importance = np.abs(acts).sum(axis=0)
            importance_upsampled = np.repeat(importance, len(signal[0]) // len(importance))
            
            # Pad if sizes mismatch due to division
            rem = len(signal[0]) - len(importance_upsampled)
            if rem > 0: importance_upsampled = np.pad(importance_upsampled, (0, rem))
            elif rem < 0: importance_upsampled = importance_upsampled[:rem]
            
            metrics['qrs_iou'].append(qrs_iou_score(signal[0], importance_upsampled))
            
    results = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in metrics.items()}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'faithfulness_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Interpretability metrics saved to {output_dir}")
    print(json.dumps(results, indent=2))

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.data_file:
        generate_faithfulness_report(args.checkpoint, args.data_file, args.output_dir, args.num_samples)
        generate_heatmaps(args.checkpoint, args.data_file, args.output_dir, args.num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--wavelet_type', type=str, default='mexican_hat')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='experiments/wavelet_analysis')
    parser.add_argument('--num_samples', type=int, default=50)
    args = parser.parse_args()
    main(args)
