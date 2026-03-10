"""
DANN Training Script for Cross-Dataset ECG Generalization.

Trains a Domain-Adversarial Neural Network (Ganin et al., 2016) that learns
domain-invariant feature representations by simultaneously:
  1. Minimizing classification loss on source domain (MIT-BIH)
  2. Maximizing domain confusion via gradient reversal on target domain (PTB-XL)

Usage:
  python -m src.train_dann --epochs 50 --seed 42
"""
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
from src.models.dann import DANN


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_alpha(epoch, total_epochs):
    """Progressive GRL alpha scheduling from Ganin et al. (2016).
    
    Alpha increases from 0 to 1 following a sigmoid schedule:
      alpha = 2 / (1 + exp(-10*p)) - 1, where p = epoch/total_epochs
    """
    p = epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


def train_dann_epoch(model, source_loader, target_loader, 
                      class_criterion, domain_criterion,
                      optimizer, device, alpha, max_grad_norm=1.0):
    """Train one epoch of DANN with adversarial domain adaptation."""
    model.train()
    
    total_class_loss = 0
    total_domain_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Create infinite target iterator (cycle through target data)
    target_iter = iter(target_loader)
    
    for source_data, source_labels in tqdm(source_loader, desc="Training", leave=False):
        # Get source batch
        source_data = source_data.to(device)
        source_labels = source_labels.to(device)
        batch_size = source_data.size(0)
        
        # Get target batch (cycle if exhausted)
        try:
            target_data, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_data, _ = next(target_iter)
        target_data = target_data.to(device)
        
        # Trim to match batch sizes
        min_batch = min(batch_size, target_data.size(0))
        source_data = source_data[:min_batch]
        source_labels = source_labels[:min_batch]
        target_data = target_data[:min_batch]
        
        # Domain labels: source=0, target=1
        source_domain_labels = torch.zeros(min_batch, 1, device=device)
        target_domain_labels = torch.ones(min_batch, 1, device=device)
        
        optimizer.zero_grad()
        
        # Forward pass on source data
        class_output, source_domain_output = model(source_data, alpha=alpha)
        class_loss = class_criterion(class_output, source_labels)
        source_domain_loss = domain_criterion(source_domain_output, source_domain_labels)
        
        # Forward pass on target data (only domain discrimination)
        _, target_domain_output = model(target_data, alpha=alpha)
        target_domain_loss = domain_criterion(target_domain_output, target_domain_labels)
        
        # Combined loss
        domain_loss = (source_domain_loss + target_domain_loss) / 2
        total_loss = class_loss + domain_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Track metrics
        total_class_loss += class_loss.item() * min_batch
        total_domain_loss += domain_loss.item() * min_batch
        preds = class_output.argmax(dim=1)
        total_correct += (preds == source_labels).sum().item()
        total_samples += min_batch
    
    avg_class_loss = total_class_loss / total_samples
    avg_domain_loss = total_domain_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_class_loss, avg_domain_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model using only the label classifier."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    
    with torch.no_grad():
        for data, labels in tqdm(loader, desc="Evaluating", leave=False):
            data, labels = data.to(device), labels.to(device)
            outputs = model.predict(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    avg_loss = total_loss / len(all_labels)
    return avg_loss, acc, f1, auc


def main(args):
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    
    # 1. Load Source Domain (MIT-BIH)
    print("Loading Source Domain (MIT-BIH)...")
    full_dataset = HarmonizedDataset(args.mit_file)
    
    val_size = int(len(full_dataset) * 0.15)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Source Train: {train_size}, Source Val: {val_size}")
    
    source_train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    source_val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # 2. Load Target Domain (PTB-XL) — labels used only for evaluation, not training
    print("Loading Target Domain (PTB-XL)...")
    target_dataset = HarmonizedDataset(args.ptb_file)
    target_loader = DataLoader(
        target_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    # Separate eval loader (no shuffle)
    target_eval_loader = DataLoader(
        target_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # 3. Initialize DANN
    model = DANN(in_channels=1, num_classes=2, feature_dim=256).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DANN parameters: {param_count:,}")
    
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Cosine Annealing with Warmup
    warmup_epochs = min(5, args.epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 4. Training Loop
    print(f"Starting DANN adversarial training for {args.epochs} epochs...")
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        alpha = compute_alpha(epoch, args.epochs)
        
        class_loss, domain_loss, train_acc = train_dann_epoch(
            model, source_train_loader, target_loader,
            class_criterion, domain_criterion,
            optimizer, device, alpha
        )
        
        val_loss, val_acc, val_f1, val_auc = evaluate(
            model, source_val_loader, class_criterion, device
        )
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"CLoss: {class_loss:.4f} DLoss: {domain_loss:.4f} α: {alpha:.3f} | "
              f"Val F1: {val_f1:.4f} AUC: {val_auc:.4f} | LR: {current_lr:.6f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': round(class_loss, 4),
            'domain_loss': round(domain_loss, 4),
            'alpha': round(alpha, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'val_f1': round(val_f1, 4),
            'val_auc': round(val_auc, 4),
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "experiments/dann_endpoint.pth")
            print(f"  -> Best DANN model saved (val_loss={val_loss:.4f})")
    
    # Save training history
    history_path = f"experiments/dann_history_seed{args.seed}.csv"
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to {history_path}")
    
    # Also save as dann_history.csv for compatibility
    with open("experiments/dann_history.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    
    # Save endpoint with seed-specific name
    if os.path.exists("experiments/dann_endpoint.pth"):
        import shutil
        shutil.copy("experiments/dann_endpoint.pth", f"experiments/dann_seed{args.seed}.pth")
    
    # 5. Zero-Shot Evaluation on PTB-XL
    print("\nLoading best DANN checkpoint for Zero-Shot Evaluation...")
    model.load_state_dict(torch.load("experiments/dann_endpoint.pth", map_location=device))
    
    print("Running Zero-Shot Evaluation on Target Domain (PTB-XL)...")
    test_loss, test_acc, test_f1, test_auc = evaluate(
        model, target_eval_loader, class_criterion, device
    )
    print(f"DANN Zero-Shot Result - Acc: {test_acc:.4f} - F1: {test_f1:.4f} - AUC: {test_auc:.4f}")
    
    # Save zero-shot results
    import pandas as pd
    zs_results = {
        'model': 'dann',
        'seed': args.seed,
        'zero_shot_acc': round(test_acc, 4),
        'zero_shot_f1': round(test_f1, 4),
        'zero_shot_auc': round(test_auc, 4),
    }
    zs_path = f"experiments/zeroshot_dann_seed{args.seed}.csv"
    pd.DataFrame([zs_results]).to_csv(zs_path, index=False)
    print(f"Zero-shot results saved to {zs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DANN Training for ECG Domain Adaptation")
    parser.add_argument('--mit_file', type=str, default='data/mitbih_processed.csv')
    parser.add_argument('--ptb_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
