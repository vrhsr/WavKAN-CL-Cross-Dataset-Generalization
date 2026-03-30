import argparse
import os
import tempfile
import time
import torch
import numpy as np
<<<<<<< HEAD
import pandas as pd
import torch.quantization
from thop import profile
=======
>>>>>>> 31960eb21812e61d0c2c98429f36f02f1ec30048

from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP, InceptionTime
from src.models.spline_kan import SplineKANClassifier

def build_model(name):
    in_channels = 12
    num_classes = 5
    if name == 'wavkan': return WavKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=64, in_channels=in_channels)
    if name == 'resnet': return ResNet1D(in_channels=in_channels, num_classes=num_classes, seq_len=1000)
    if name == 'vit': return ViT1D(seq_len=1000, num_classes=num_classes, in_channels=in_channels)
    if name == 'spline_kan': return SplineKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=64, in_channels=in_channels)
    if name == 'mlp': return SimpleMLP(input_dim=1000, num_classes=num_classes, in_channels=in_channels)
    if name == 'inception': return InceptionTime(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def state_dict_size_mb(model):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
    os.remove(f.name)
    return size

def get_flops(model, inp):
    model.eval()
    try:
        flops, _ = profile(model, inputs=(inp,), verbose=False)
        return flops / 1e6 # MFLOPs
    except Exception as e:
        print(f"FLOP calculation failed: {e}")
        return 0

def latency_ms(model, inp, runs=100, warmup=20):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(inp)
            
        if inp.is_cuda:
            torch.cuda.synchronize()
            
        st = time.perf_counter()
        for _ in range(runs):
            model(inp)
            
        if inp.is_cuda:
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
    return ((end_time - st) * 1000) / (runs * inp.size(0))

def main(args):
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_fp32 = torch.randn(args.batch_size, 12, 1000).to(device)
    x_fp16 = x_fp32.half().to(device) if device.type == 'cuda' else None
    x_int8_cpu = torch.randn(args.batch_size, 12, 1000).cpu()

    rows = []
    for name in models:
        print(f"Benchmarking {name}...")
        model_cpu = build_model(name).cpu()
        params = count_params(model_cpu)
        
        # 1. FLOPs
        macs = get_flops(model_cpu, torch.randn(1, 12, 1000))
        
        # 2. FP32
        model = model_cpu.to(device)
        fp32_lat = latency_ms(model, x_fp32)
        size_fp32 = state_dict_size_mb(model)
        
        # 3. FP16
        if device.type == 'cuda':
            model_fp16 = build_model(name).half().to(device)
            fp16_lat = latency_ms(model_fp16, x_fp16)
        else:
            fp16_lat = float('nan')
            
        # 4. INT8
        try:
            model_int8 = torch.quantization.quantize_dynamic(
                model_cpu, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
            )
            int8_lat = latency_ms(model_int8, x_int8_cpu)
            size_int8 = state_dict_size_mb(model_int8)
        except Exception as e:
            print(f"Quantization failed for {name}: {e}")
            int8_lat = float('nan')
            size_int8 = float('nan')

        rows.append({
            'model': name,
            'params_m': params / 1e6,
            'flops_m': macs,
            'fp32_ms_per_sample': fp32_lat,
            'fp16_ms_per_sample': fp16_lat,
            'int8_ms_per_sample': int8_lat,
            'fp32_size_mb': size_fp32,
            'int8_size_mb': size_int8,
            'int8_compression_pct': 100.0 * (size_fp32 - size_int8) / max(size_fp32, 1e-9) if not np.isnan(size_int8) else 0.0,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    
    print("\n" + "="*80)
    print("DEPLOYMENT BENCHMARK RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nSaved thoroughly benchmarked results to {args.out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='wavkan,spline_kan,resnet,vit,mlp,inception')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_csv', default='experiments/runs/deployment_benchmark.csv')
    main(parser.parse_args())
