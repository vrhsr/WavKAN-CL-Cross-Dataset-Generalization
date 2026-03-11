import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import numpy as np
from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from src.models.dann import DANN
from sklearn.metrics import f1_score

def evaluate_noise(model, test_file, snr_db, device, batch_size=32, normalize_input=False):
    """
    Evaluates model on a dataset with injected noise.
    """
    dataset = HarmonizedDataset(test_file, noise_snr_db=snr_db)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            if normalize_input:
                inputs = (inputs - inputs.mean(dim=-1, keepdim=True)) / (inputs.std(dim=-1, keepdim=True) + 1e-6)
            if isinstance(model, DANN):
                outputs = model.predict(inputs)
            else:
                outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    return f1_score(all_labels, all_preds)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize and Load Model
    if args.model == 'wavkan':
        model = WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=128).to(device)
    elif args.model == 'resnet':
        model = ResNet1D(in_channels=1, num_classes=2).to(device)
    elif args.model == 'vit':
        model = ViT1D(seq_len=250, num_classes=2).to(device)
    elif args.model == 'spline_kan':
        model = SplineKANClassifier(input_dim=250, num_classes=2).to(device)
    elif args.model == 'mlp':
        model = SimpleMLP(input_dim=250, num_classes=2).to(device)
    elif args.model == 'dann':
        model = DANN(in_channels=1, num_classes=2, feature_dim=256).to(device)
    else:
        raise ValueError("Unknown model")
        
    # Load weights
    model_path = f"experiments/{args.model}_endpoint.pth"
    print(f"Loading weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Train the model first.")
        return

    # 2. Run Stress Test
    snr_levels = [None, 20, 15, 10, 5, 0] # None = Clean
    results = {}
    
    print("\n--- Starting Noise Stress Test ---")
    print(f"Model: {args.model}")
    print(f"Test Set: {args.ptb_file}")
    
    for snr in snr_levels:
        label = "Clean" if snr is None else f"{snr}dB"
        print(f"Testing at {label}...")
        f1 = evaluate_noise(model, args.ptb_file, snr, device, normalize_input=args.normalize_input)
        results[label] = f1
        print(f"-> F1 Score: {f1:.4f}")
        
    # 3. Save Results
    df_res = pd.DataFrame([results])
    df_res.index = [args.model]
    filename = f"robustness_{args.model}_norm.csv" if args.normalize_input else f"robustness_{args.model}.csv"
    save_path = f"experiments/{filename}"
    df_res.to_csv(save_path)
    print(f"\nSaved results to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--model', type=str, required=True, choices=['wavkan', 'resnet', 'vit', 'spline_kan', 'mlp', 'dann'])
    parser.add_argument('--normalize_input', action='store_true')
    args = parser.parse_args()
    main(args)
