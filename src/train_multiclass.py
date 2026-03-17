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
    
    return running_loss / len(loader), f1_score(all_labels, all_preds, average='macro')


def evaluate_hierarchical(model, loader, device, num_classes):
    """Evaluate hierarchical classification: Level 1 (Normal vs Abnormal), Level 2 (specific types)."""
    model.eval()
    all_preds_l1 = []
    all_labels_l1 = []
    all_preds_l2 = []
    all_labels_l2 = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Level 1: Binary (0 = Normal, 1+ = Abnormal)
            l1_preds = (preds > 0).astype(int)
            l1_labels = (labels > 0).cpu().numpy().astype(int)
            all_preds_l1.extend(l1_preds)
            all_labels_l1.extend(l1_labels)
            
            # Level 2: Multi-class (only for abnormal cases)
            abnormal_mask = labels > 0
            if abnormal_mask.sum() > 0:
                abnormal_preds = preds[abnormal_mask]
                abnormal_labels = labels[abnormal_mask].cpu().numpy()
                all_preds_l2.extend(abnormal_preds)
                all_labels_l2.extend(abnormal_labels)
    
    from sklearn.metrics import f1_score
    f1_l1 = f1_score(all_labels_l1, all_preds_l1, average='macro')
    f1_l2 = f1_score(all_labels_l2, all_preds_l2, average='macro') if all_labels_l2 else 0.0
    
    return f1_l1, f1_l2


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define class mapping (example: binary to arrhythmia types)
    # This is a placeholder - adjust based on your dataset
    class_mapping = {
        0: 0,  # Normal
        1: 1,  # Abnormal (could be split into specific types if data allows)
    }
    
    if args.multi_class:
        # Example mapping for multi-class (adjust based on SCP codes or dataset)
        class_mapping = {
            0: 0,  # Normal
            1: 1,  # Atrial fibrillation
            # Add more mappings as needed
        }
        num_classes = len(set(class_mapping.values()))
    else:
        num_classes = 2

    # Load datasets
    source_dataset = HarmonizedDataset(args.source_file, multi_class=args.multi_class, class_mapping=class_mapping)
    target_dataset = HarmonizedDataset(args.target_file, multi_class=args.multi_class, class_mapping=class_mapping)

    # Split target into train/val for adaptation
    target_train_size = int(0.8 * len(target_dataset))
    target_val_size = len(target_dataset) - target_train_size
    target_train, target_val = random_split(target_dataset, [target_train_size, target_val_size])

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True)
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    if args.model == 'wavkan':
        model = WavKANClassifier(input_dim=250, num_classes=num_classes, hidden_dim=args.hidden_dim, in_channels=args.in_channels).to(device)
    elif args.model == 'spline_kan':
        model = SplineKANClassifier(input_dim=250, num_classes=num_classes, hidden_dim=args.hidden_dim, in_channels=args.in_channels).to(device)
    elif args.model == 'resnet':
        model = ResNet1D(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.model == 'vit':
        model = ViT1D(seq_len=250, num_classes=num_classes).to(device)
    elif args.model == 'simple_mlp':
        model = SimpleMLP(input_dim=250 * args.in_channels, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = 0.0
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_f1 = train_epoch(model, source_loader, criterion, optimizer, device)
        val_acc, val_f1_macro, val_f1_weighted = evaluate(model, target_val_loader, device, num_classes)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Acc: {val_acc:.4f}, Val F1 Macro: {val_f1_macro:.4f}")

        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            patience_counter = 0
            torch.save(model.state_dict(), f'experiments/multiclass_{args.model}_seed{args.seed}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    # Load best model and evaluate on full target
    model.load_state_dict(torch.load(f'experiments/multiclass_{args.model}_seed{args.seed}.pth'))
    target_full_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
    final_acc, final_f1_macro, final_f1_weighted = evaluate(model, target_full_loader, device, num_classes)
    final_f1_l1, final_f1_l2 = evaluate_hierarchical(model, target_full_loader, device, num_classes)

    # Save results
    results = {
        'model': args.model,
        'seed': args.seed,
        'multi_class': args.multi_class,
        'num_classes': num_classes,
        'final_acc': final_acc,
        'final_f1_macro': final_f1_macro,
        'final_f1_weighted': final_f1_weighted,
        'final_f1_l1': final_f1_l1,
        'final_f1_l2': final_f1_l2,
        'best_val_f1': best_f1
    }

    with open(f'experiments/multiclass_{args.model}_seed{args.seed}_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"Final Acc: {final_acc:.4f}, F1 Macro: {final_f1_macro:.4f}, F1 Weighted: {final_f1_weighted:.4f}, F1 L1: {final_f1_l1:.4f}, F1 L2: {final_f1_l2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train multi-class ECG classification')
    parser.add_argument('--source_file', type=str, required=True, help='Path to source domain CSV')
    parser.add_argument('--target_file', type=str, required=True, help='Path to target domain CSV')
    parser.add_argument('--model', type=str, choices=['wavkan', 'spline_kan', 'resnet', 'vit', 'simple_mlp'], required=True)
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for KAN models')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels (leads)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--multi_class', action='store_true', help='Enable multi-class classification')

    args = parser.parse_args()
    main(args)