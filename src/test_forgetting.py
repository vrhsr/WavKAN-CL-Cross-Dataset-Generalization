"""
Catastrophic Forgetting Ablation Script.

Evaluates how much the model "forgets" the source domain (MIT-BIH)
after being fine-tuned on the target domain (PTB-XL).

Forgetting = (Initial Source F1) - (Final Source F1)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import argparse
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from src.models.dann import DANN

def get_k_shot_indices(dataset, k, seed=42):
    np.random.seed(seed)
    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    k_per_class = k // 2
    class_0_idx = np.where(labels == 0)[0]
    class_1_idx = np.where(labels == 1)[0]
    selected_0 = np.random.choice(class_0_idx, min(k_per_class, len(class_0_idx)), replace=False)
    selected_1 = np.random.choice(class_1_idx, min(k_per_class, len(class_1_idx)), replace=False)
    indices = np.concatenate([selected_0, selected_1])
    np.random.shuffle(indices)
    return indices

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return f1_score(all_labels, all_preds)

def fine_tune(model, train_loader, epochs=10, lr=1e-4, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Datasets
    source_dataset = HarmonizedDataset(args.source_file)
    target_dataset = HarmonizedDataset(args.target_file)
    
    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=False)
    
    k_shots = [10, 50, 100, 500]
    results = []
    
    for k in k_shots:
        print(f"\n--- Testing Forgetting after {k}-shot adaptation ({args.model}) ---")
        
        # 2. Initialize Model
        if args.model == 'wavkan':
            model = WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
        elif args.model == 'resnet':
            model = ResNet1D(in_channels=1, num_classes=2).to(device)
        elif args.model == 'vit':
            model = ViT1D(seq_len=250, num_classes=2).to(device)
        elif args.model == 'mlp':
            model = SimpleMLP(input_dim=250, num_classes=2).to(device)
        elif args.model == 'spline_kan':
            model = SplineKANClassifier(input_dim=250, num_classes=2).to(device)
        else:
            raise ValueError("Unknown model")
            
        # 3. Load Initial Weights (Pre-trained on Source)
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
        else:
            print(f"Warning: Checkpoint {args.checkpoint} not found. Using random init.")

        # 4. Measure Initial Source F1
        f1_source_initial = evaluate(model, source_loader, device)
        print(f"Initial Source F1: {f1_source_initial:.4f}")

        # 5. Adapt to Target (Few-Shot fine-tuning)
        indices = get_k_shot_indices(target_dataset, k, seed=args.seed)
        train_subset = Subset(target_dataset, indices)
        train_loader = DataLoader(train_subset, batch_size=min(16, k), shuffle=True)
        
        model = fine_tune(model, train_loader, epochs=10, lr=1e-4, device=device)

        # 6. Measure Final F1s
        f1_source_final = evaluate(model, source_loader, device)
        f1_target_final = evaluate(model, DataLoader(target_dataset, batch_size=64, shuffle=False), device)
        
        forgetting = f1_source_initial - f1_source_final
        print(f"Final Source F1: {f1_source_final:.4f} (Forgetting: {forgetting:.4f})")
        print(f"Final Target F1: {f1_target_final:.4f}")

        results.append({
            'k': k,
            'source_f1_initial': f1_source_initial,
            'source_f1_final': f1_source_final,
            'forgetting': forgetting,
            'target_f1': f1_target_final
        })

    # Save Results
    df_res = pd.DataFrame(results)
    save_path = f"experiments/forgetting_{args.model}_seed{args.seed}.csv"
    df_res.to_csv(save_path, index=False)
    print(f"\nSaved forgetting analysis to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--source_file', type=str, default='data/mitbih_processed.csv')
    parser.add_argument('--target_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    args = parser.parse_args()
    main(args)
