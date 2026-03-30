import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import argparse
import random
import csv
import os
import math
from tqdm import tqdm

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device).float(), labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent ViT training collapse
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    return running_loss / len(loader), accuracy_score(all_labels, all_preds)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        
    return running_loss / len(loader), acc, f1, auc


def main(args):
    # Reproducibility
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    
    # 1. Load Data
    print("Loading Source Domain (MIT-BIH)...")
    full_dataset = HarmonizedDataset(args.mit_file)
    
    # Train/Validation Split (85/15)
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {train_size}, Validation: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    print("Loading Target Domain (PTB-XL)...")
    target_dataset = HarmonizedDataset(args.ptb_file)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=True)
    
    # 2. Initialize Model
    if args.model == 'wavkan':
        model = WavKANClassifier(input_dim=1000, num_classes=5, 
                               hidden_dim=args.hidden_dim, 
                               wavelet_type=args.wavelet_type,
                               depth=args.depth).to(device)
    elif args.model == 'resnet':
        model = ResNet1D(in_channels=12, num_classes=5).to(device)
    elif args.model == 'vit':
        model = ViT1D(seq_len=1000, num_classes=5).to(device)
    elif args.model == 'mlp':
        model = SimpleMLP(input_dim=1000, num_classes=5).to(device)
    elif args.model == 'spline_kan':
        model = SplineKANClassifier(input_dim=1000, num_classes=5).to(device)
    else:
        raise ValueError("Unknown model")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Cosine Annealing with Warmup
    warmup_epochs = min(5, args.epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 3. Training Loop with Validation & Loss Logging
    print(f"Starting training for {args.model}...")
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'val_f1': round(val_f1, 4),
            'val_auc': round(val_auc, 4)
        })
        
        # Save best model (early stopping checkpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"experiments/{args.model}_endpoint.pth")
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")
    
    # Save training history
    history_path = f"experiments/{args.model}_history.csv"
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to {history_path}")
    
    # 4. Zero-Shot Testing (using best checkpoint)
    print("Loading best checkpoint for Zero-Shot Evaluation...")
    model.load_state_dict(torch.load(f"experiments/{args.model}_endpoint.pth", map_location=device))
    
    print("Running Zero-Shot Evaluation on Target Domain...")
    test_loss, test_acc, test_f1, test_auc = evaluate(model, target_loader, criterion, device)
    print(f"Result - Acc: {test_acc:.4f} - F1: {test_f1:.4f} - AUC: {test_auc:.4f}")
    
    # Save zero-shot results per seed for statistical analysis
    import pandas as pd
    zs_results = {
        'model': args.model,
        'seed': args.seed,
        'zero_shot_acc': round(test_acc, 4),
        'zero_shot_f1': round(test_f1, 4),
        'zero_shot_auc': round(test_auc, 4)
    }
    zs_path = f"experiments/zeroshot_{args.model}_seed{args.seed}.csv"
    pd.DataFrame([zs_results]).to_csv(zs_path, index=False)
    print(f"Zero-shot results saved to {zs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mit_file', type=str, default='data/mitbih_processed.csv')
    parser.add_argument('--ptb_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--model', type=str, required=True, choices=['wavkan', 'resnet', 'vit', 'spline_kan', 'mlp'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--wavelet_type', type=str, default='mexican_hat', choices=['mexican_hat', 'morlet'])
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)
