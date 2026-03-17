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


def coral_loss(source_features, target_features):
    """
    CORAL loss: Align covariance matrices between source and target domains.
    """
    def covariance(x):
        n = x.size(0)
        mean_x = x.mean(dim=0, keepdim=True)
        x_centered = x - mean_x
        cov = (x_centered.t() @ x_centered) / (n - 1)
        return cov

    source_cov = covariance(source_features)
    target_cov = covariance(target_features)

    # Frobenius norm of difference
    loss = torch.norm(source_cov - target_cov, p='fro') ** 2
    return loss


def mmd_loss(source_features, target_features, kernel='rbf', sigma=1.0):
    """
    MMD loss: Maximum Mean Discrepancy between source and target.
    """
    def rbf_kernel(x, y, sigma):
        x = x.unsqueeze(1)  # (n, 1, d)
        y = y.unsqueeze(0)  # (1, m, d)
        dist = torch.sum((x - y) ** 2, dim=2)
        return torch.exp(-dist / (2 * sigma ** 2))

    n = source_features.size(0)
    m = target_features.size(0)
    
    xx = rbf_kernel(source_features, source_features, sigma).sum() / (n * n)
    yy = rbf_kernel(target_features, target_features, sigma).sum() / (m * m)
    xy = rbf_kernel(source_features, target_features, sigma).sum() / (n * m)
    
    return xx + yy - 2 * xy


class CORALModel(nn.Module):
    def __init__(self, base_model, feature_dim=128):
        super(CORALModel, self).__init__()
        self.base_model = base_model
        # Add a feature extractor layer if needed
        if hasattr(base_model, 'feature_extractor'):
            self.feature_extractor = base_model.feature_extractor
        else:
            # For models without explicit feature extractor, use the model directly
            self.feature_extractor = base_model
        self.classifier = nn.Linear(feature_dim, 2)  # Assuming binary classification

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def extract_features(self, x):
        return self.feature_extractor(x)


def train_epoch_coral(model, source_loader, target_loader, criterion, optimizer, device, lambda_coral=1.0, lambda_mmd=0.0, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Get iterators
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for i in range(len(source_loader)):
        try:
            source_inputs, source_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_inputs, source_labels = next(source_iter)

        try:
            target_inputs, _ = next(target_iter)  # Target labels not used in unsupervised DA
        except StopIteration:
            target_iter = iter(target_loader)
            target_inputs, _ = next(target_iter)

        source_inputs, source_labels = source_inputs.to(device).float(), source_labels.to(device).long()
        target_inputs = target_inputs.to(device).float()

        optimizer.zero_grad()

        # Forward pass
        source_features = model.extract_features(source_inputs)
        target_features = model.extract_features(target_inputs)
        source_outputs = model.classifier(source_features)

        # Classification loss
        cls_loss = criterion(source_outputs, source_labels)

        # Domain alignment losses
        coral = coral_loss(source_features, target_features)
        mmd = mmd_loss(source_features, target_features) if lambda_mmd > 0 else 0

        # Total loss
        loss = cls_loss + lambda_coral * coral + lambda_mmd * mmd

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(source_outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(source_labels.cpu().numpy())

    return running_loss / len(source_loader), f1_score(all_labels, all_preds)


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

    # Load datasets
    source_dataset = HarmonizedDataset(args.source_file)
    target_dataset = HarmonizedDataset(args.target_file)

    # Split target into train/val for adaptation
    target_train_size = int(0.8 * len(target_dataset))
    target_val_size = len(target_dataset) - target_train_size
    target_train, target_val = random_split(target_dataset, [target_train_size, target_val_size])

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True)
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    if args.model == 'wavkan':
        base_model = WavKANClassifier(input_dim=250, num_classes=2, hidden_dim=args.hidden_dim)
    elif args.model == 'spline_kan':
        base_model = SplineKANClassifier(input_dim=250, num_classes=2, hidden_dim=args.hidden_dim)
    elif args.model == 'resnet':
        base_model = ResNet1D(in_channels=1, num_classes=2)
    elif args.model == 'vit':
        base_model = ViT1D(seq_len=250, num_classes=2)
    elif args.model == 'simple_mlp':
        base_model = SimpleMLP(input_dim=250, num_classes=2)

    # Wrap in CORAL model
    model = CORALModel(base_model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = 0.0
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_f1 = train_epoch_coral(model, source_loader, target_train_loader, criterion, optimizer, device, lambda_coral=args.lambda_coral, lambda_mmd=args.lambda_mmd)
        val_f1 = evaluate(model, target_val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'experiments/coral_{args.model}_seed{args.seed}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    # Load best model and evaluate on full target
    model.load_state_dict(torch.load(f'experiments/coral_{args.model}_seed{args.seed}.pth'))
    target_full_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
    final_f1 = evaluate(model, target_full_loader, device)

    # Save results
    results = {
        'model': args.model,
        'seed': args.seed,
        'lambda_coral': args.lambda_coral,
        'lambda_mmd': args.lambda_mmd,
        'final_f1': final_f1,
        'best_val_f1': best_f1
    }

    with open(f'experiments/coral_{args.model}_seed{args.seed}_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"Final F1 on target: {final_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with CORAL/MMD domain adaptation')
    parser.add_argument('--source_file', type=str, required=True, help='Path to source domain CSV')
    parser.add_argument('--target_file', type=str, required=True, help='Path to target domain CSV')
    parser.add_argument('--model', type=str, choices=['wavkan', 'spline_kan', 'resnet', 'vit', 'simple_mlp'], required=True)
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for KAN models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lambda_coral', type=float, default=1.0, help='Weight for CORAL loss')
    parser.add_argument('--lambda_mmd', type=float, default=0.0, help='Weight for MMD loss')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)