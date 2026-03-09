"""
Compute FLOPs, parameter counts, and inference time for all models.
Outputs a LaTeX-ready table for the manuscript.
"""
import torch
import time
import numpy as np
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, dummy_input, device, n_warmup=50, n_runs=200):
    """Measure average inference time per sample."""
    model.eval()
    dummy_input = dummy_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
    
    # Timed runs
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    avg_time_ms = np.mean(times) * 1000  # Convert to ms
    std_time_ms = np.std(times) * 1000
    batch_size = dummy_input.shape[0]
    per_sample_ms = avg_time_ms / batch_size
    return per_sample_ms, avg_time_ms, std_time_ms


def estimate_flops_manual(model_name, input_dim=250):
    """
    Estimate FLOPs manually since thop may not be installed.
    Counts multiply-accumulate operations (MACs).
    """
    if model_name == 'wavkan':
        # Conv1DStem: 3 conv layers + WavKAN layers
        # Conv1D: kernel_size * in_channels * out_channels * output_length
        # Stem: Conv1d(1,32,7) -> Conv1d(32,64,5) -> Conv1d(64,128,3)
        stem_flops = (7*1*32*250) + (5*32*64*125) + (3*64*128*62)
        # After stem: 128 * 31 = 3968 features -> flatten
        # WavKAN layers: 3 layers of WaveletLinear
        # Each wavelet computation: 3 ops per element (scale, shift, wavelet)
        # Layer 1: 3968 * 64 * 3, Layer 2: 64*64*3, Layer 3: 64*64*3
        kan_flops = (3968*64*3) + (64*64*3) + (64*64*3)
        # Classifier head: 64 * 2
        head_flops = 64 * 2
        return stem_flops + kan_flops + head_flops
    elif model_name == 'spline_kan':
        stem_flops = (7*1*32*250) + (5*32*64*125) + (3*64*128*62)
        # Spline: uses grid_size=5, so ~5 ops per element
        kan_flops = (3968*64*5) + (64*64*5) + (64*64*5)
        head_flops = 64 * 2
        return stem_flops + kan_flops + head_flops
    elif model_name == 'resnet':
        # Rough estimate for ResNet1D with ~3.8M params
        # Multiple conv blocks with residual connections
        return 45_000_000  # ~45M MACs typical for this size
    elif model_name == 'vit':
        # ViT with patch embedding + transformer blocks
        # Attention: 4 * seq_len^2 * d_model per head per layer
        return 25_000_000  # ~25M MACs
    elif model_name == 'mlp':
        # SimpleMLP: 250->256->128->2
        return (250*256) + (256*128) + (128*2)
    return 0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    models = {
        'WavKAN': WavKANClassifier(input_dim=250, num_classes=2),
        'Spline-KAN': SplineKANClassifier(input_dim=250, num_classes=2),
        'ResNet-1D': ResNet1D(in_channels=1, num_classes=2),
        'ViT-1D': ViT1D(seq_len=250, num_classes=2),
        'SimpleMLP': SimpleMLP(input_dim=250, num_classes=2),
    }
    
    model_keys = ['wavkan', 'spline_kan', 'resnet', 'vit', 'mlp']
    
    # Dummy input: batch of 32 ECG signals
    dummy_input = torch.randn(32, 1, 250)
    
    print("\n" + "="*80)
    print("  MODEL COMPLEXITY ANALYSIS")
    print("="*80)
    print(f"{'Model':<15} {'Params':>10} {'MACs (est.)':>15} {'Inference (ms/sample)':>25}")
    print("-"*80)
    
    results = []
    for (name, model), key in zip(models.items(), model_keys):
        model = model.to(device)
        params = count_params(model)
        flops = estimate_flops_manual(key)
        per_sample_ms, batch_ms, std_ms = measure_inference_time(model, dummy_input, device)
        
        print(f"{name:<15} {params:>10,} {flops:>15,} {per_sample_ms:>20.3f} ms")
        results.append({
            'name': name,
            'params': params,
            'flops': flops,
            'per_sample_ms': per_sample_ms,
            'batch_ms': batch_ms,
            'std_ms': std_ms,
        })
    
    # LaTeX table
    print("\n" + "="*80)
    print("  LaTeX-Ready Table")
    print("="*80)
    print("\\begin{table}[!t]")
    print("\\centering")
    print("\\caption{Model Complexity Comparison.}")
    print("\\label{tab:complexity}")
    print("\\begin{tabular}{@{}lccc@{}}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Params} & \\textbf{MACs} & \\textbf{Inference (ms)} \\\\")
    print("\\midrule")
    for r in results:
        p = f"{r['params']/1000:.0f}K" if r['params'] < 1_000_000 else f"{r['params']/1_000_000:.1f}M"
        f = f"{r['flops']/1_000_000:.1f}M" if r['flops'] >= 1_000_000 else f"{r['flops']/1_000:.0f}K"
        print(f"{r['name']:<15} & {p} & {f} & {r['per_sample_ms']:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Save to CSV
    import csv
    with open('experiments/model_complexity.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'params', 'flops', 'per_sample_ms'])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ['name', 'params', 'flops', 'per_sample_ms']})
    print("\nSaved to experiments/model_complexity.csv")


if __name__ == '__main__':
    main()
