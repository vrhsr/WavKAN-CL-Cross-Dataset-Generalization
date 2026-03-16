"""Deployment benchmark: params, model size, FP32/INT8 latency proxy, and memory footprint."""
import argparse
import os
import tempfile
import time
import torch
import numpy as np
import torch.quantization

from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from src.models.dann import DANN


def build_model(name):
    if name == 'wavkan':
        return WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=64)
    if name == 'resnet':
        return ResNet1D(in_channels=1, num_classes=2)
    if name == 'vit':
        return ViT1D(seq_len=250, num_classes=2)
    if name == 'spline_kan':
        return SplineKANClassifier(input_dim=250, num_classes=2)
    if name == 'mlp':
        return SimpleMLP(input_dim=250, num_classes=2)
    if name == 'dann':
        return DANN(in_channels=1, num_classes=2, feature_dim=256)
    raise ValueError(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def state_dict_size_mb(model):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
    os.remove(f.name)
    return size


def latency_ms(model, inp, runs=200, warmup=30):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inp)
        times = []
        for _ in range(runs):
            st = time.perf_counter()
            _ = model(inp)
            times.append((time.perf_counter() - st) * 1000)
    return float(np.mean(times) / inp.size(0))


def try_int8_dynamic(model):
    quantizable = {torch.nn.Linear, torch.nn.LSTM}
    q = torch.quantization.quantize_dynamic(model.cpu(), qconfig_spec=quantizable, dtype=torch.qint8)
    return q


def main(args):
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    x = torch.randn(args.batch_size, 1, 250)

    rows = []
    for name in models:
        model = build_model(name).cpu()
        fp32_lat = latency_ms(model, x)
        size_fp32 = state_dict_size_mb(model)

        q_model = try_int8_dynamic(model)
        int8_lat = latency_ms(q_model, x)
        size_int8 = state_dict_size_mb(q_model)

        rows.append({
            'model': name,
            'params': count_params(model),
            'fp32_ms_per_sample': fp32_lat,
            'int8_ms_per_sample': int8_lat,
            'fp32_size_mb': size_fp32,
            'int8_size_mb': size_int8,
            'size_reduction_pct': 100.0 * (size_fp32 - size_int8) / max(size_fp32, 1e-9),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs('experiments', exist_ok=True)
    out = args.out_csv
    df.to_csv(out, index=False)
    print(df)
    print(f"Saved deployment benchmark to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='wavkan,spline_kan,resnet,vit,mlp,dann')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_csv', default='experiments/deployment_benchmark.csv')
    main(parser.parse_args())
