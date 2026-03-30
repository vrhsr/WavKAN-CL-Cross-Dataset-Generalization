import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import argparse
import random
import csv
import os
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


def pseudo_label_loss(outputs, threshold=0.9):
    """Pseudo-labeling loss for source-free adaptation."""
    probs = torch.softmax(outputs, dim=1)
    max_probs, pseudo_labels = torch.max(probs, dim=1)
    mask = max_probs > threshold
    if mask.sum() == 0:
        return torch.tensor(0.0, device=outputs.device)
    
    # Cross-entropy with pseudo-labels
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(outputs, pseudo_labels)
    return (loss * mask).mean()


def bn_adaptation(model, loader, device, steps=1):
    """BatchNorm adaptation on target data."""
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    # Enable BN params
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True
    
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    for _ in range(steps):
        for inputs, _ in loader:
            inputs = inputs.to(device).float()
            optimizer.zero_grad()
            _ = model(inputs)  # Forward to update BN stats
            # No loss, just update BN
            optimizer.step()
            break  # One batch per step
    
    # Restore requires_grad
    for param in model.parameters():
        param.requires_grad = True


def train_epoch_source_free(model, target_loader, optimizer, device, lambda_pseudo=1.0, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(target_loader, desc="Source-free adaptation"):
        inputs, labels = inputs.to(device).float(), labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Pseudo-label loss
        pseudo_loss = pseudo_label_loss(outputs)
        
        # Entropy minimization (optional)
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        
        loss = lambda_pseudo * pseudo_loss + 0.1 * entropy  # Weight entropy lightly

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(target_loader), f1_score(all_labels, all_preds)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return f1_score(all_labels, all_preds)


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load target dataset (source-free: no source data used)
    target_dataset = HarmonizedDataset(args.target_file)

    # Split target into train/val
    target_train_size = int(0.8 * len(target_dataset))
    target_val_size = len(target_dataset) - target_train_size
    target_train, target_val = random_split(target_dataset, [target_train_size, target_val_size])

    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True)
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False)

    # Initialize model (pretrained on source, but we simulate source-free by starting from scratch or loading)
    if args.model == 'wavkan':
        model = WavKANClassifier(input_dim=1000, num_classes=5, hidden_dim=args.hidden_dim, in_channels=args.in_channels).to(device)
    elif args.model == 'spline_kan':
        model = SplineKANClassifier(input_dim=1000, num_classes=5, hidden_dim=args.hidden_dim, in_channels=args.in_channels).to(device)
    elif args.model == 'resnet':
        model = ResNet1D(in_channels=args.in_channels, num_classes=5).to(device)
    elif args.model == 'vit':
        model = ViT1D(seq_len=1000, num_classes=5).to(device)
    elif args.model == 'simple_mlp':
        model = SimpleMLP(input_dim=1000 * args.in_channels, num_classes=5).to(device)

    # Load pretrained model if provided (simulating source training)
    if args.pretrained_path:
        model.load_state_dict(torch.load(args.pretrained_path))
        print(f"Loaded pretrained model from {args.pretrained_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = 0.0
    patience = 10
    patience_counter = 0

    # Source-free adaptation
    for epoch in range(args.epochs):
        # Optional BN adaptation
        if args.bn_adapt:
            bn_adaptation(model, target_train_loader, device, steps=1)
        
        train_loss, train_f1 = train_epoch_source_free(model, target_train_loader, optimizer, device, lambda_pseudo=args.lambda_pseudo)
        val_f1 = evaluate(model, target_val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'experiments/source_free_{args.model}_seed{args.seed}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    # Load best model and evaluate on full target
    model.load_state_dict(torch.load(f'experiments/source_free_{args.model}_seed{args.seed}.pth'))
    target_full_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
    final_f1 = evaluate(model, target_full_loader, device)

    # Save results
    results = {
        'model': args.model,
        'seed': args.seed,
        'lambda_pseudo': args.lambda_pseudo,
        'bn_adapt': args.bn_adapt,
        'final_f1': final_f1,
        'best_val_f1': best_f1
    }

    with open(f'experiments/source_free_{args.model}_seed{args.seed}_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"Final F1 on target: {final_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Source-free domain adaptation')
    parser.add_argument('--target_file', type=str, required=True, help='Path to target domain CSV')
    parser.add_argument('--model', type=str, choices=['wavkan', 'spline_kan', 'resnet', 'vit', 'simple_mlp'], required=True)
    parser.add_argument('--pretrained_path', type=str, help='Path to pretrained model on source')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lambda_pseudo', type=float, default=1.0, help='Weight for pseudo-label loss')
    parser.add_argument('--bn_adapt', action='store_true', help='Enable BatchNorm adaptation')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)