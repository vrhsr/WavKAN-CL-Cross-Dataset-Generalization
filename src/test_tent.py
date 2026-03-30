import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import numpy as np
from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from sklearn.metrics import f1_score


def entropy_loss(logits):
    """Entropy minimization loss for TENT (Multi-label)."""
    probs = torch.sigmoid(logits)
    entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
    return entropy.mean()


def tent_adapt(model, loader, optimizer, device, steps=1, lr=1e-3):
    """Perform TENT adaptation on a batch."""
    model.train()
    for _ in range(steps):
        for inputs, _ in loader:  # Unlabeled, so ignore labels
            inputs = inputs.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = entropy_loss(outputs)
            loss.backward()
            optimizer.step()
            break  # Adapt on one batch per step
    model.eval()


def evaluate_tent(model, test_loader, device, adapt_steps=1, adapt_lr=1e-3):
    """Evaluate with TENT adaptation."""
    # Create optimizer for adaptation (only BN/affine params typically)
    tent_params = []
    for name, param in model.named_parameters():
        if 'bn' in name or 'norm' in name or 'bias' in name:
            tent_params.append(param)
    if not tent_params:
        # If no BN layers, adapt all params
        tent_params = list(model.parameters())

    optimizer = optim.SGD(tent_params, lr=adapt_lr, momentum=0.9)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()

            # Adapt on this batch
            adapt_loader = DataLoader([(inputs, labels)], batch_size=len(inputs), shuffle=False)
            tent_adapt(model, adapt_loader, optimizer, device, steps=adapt_steps, lr=adapt_lr)

            # Now predict
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return f1_score(all_labels, all_preds, average='macro', zero_division=0)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    if args.model == 'wavkan':
        model = WavKANClassifier(input_dim=1000, num_classes=5, hidden_dim=args.hidden_dim).to(device)
    elif args.model == 'spline_kan':
        model = SplineKANClassifier(input_dim=1000, num_classes=5, hidden_dim=args.hidden_dim).to(device)
    elif args.model == 'resnet':
        model = ResNet1D(in_channels=12, num_classes=5).to(device)
    elif args.model == 'vit':
        model = ViT1D(seq_len=1000, num_classes=5).to(device)
    elif args.model == 'simple_mlp':
        base_model = SimpleMLP(input_dim=1000, num_classes=5).to(device)

    # Load pretrained model
    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")

    # Load target dataset
    target_dataset = HarmonizedDataset(args.target_file)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate without adaptation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in target_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    no_adapt_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"F1 without TENT: {no_adapt_f1:.4f}")

    # Evaluate with TENT
    tent_f1 = evaluate_tent(model, target_loader, device, adapt_steps=args.adapt_steps, adapt_lr=args.adapt_lr)
    print(f"F1 with TENT: {tent_f1:.4f}")

    # Save results
    results = {
        'model': args.model,
        'adapt_steps': args.adapt_steps,
        'adapt_lr': args.adapt_lr,
        'no_adapt_f1': no_adapt_f1,
        'tent_f1': tent_f1
    }

    import csv
    with open(f'experiments/tent_{args.model}_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate with TENT test-time adaptation')
    parser.add_argument('--target_file', type=str, required=True, help='Path to target domain CSV')
    parser.add_argument('--model', type=str, choices=['wavkan', 'spline_kan', 'resnet', 'vit', 'simple_mlp'], required=True)
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--adapt_steps', type=int, default=1, help='Number of adaptation steps per batch')
    parser.add_argument('--adapt_lr', type=float, default=1e-3, help='Learning rate for TENT adaptation')

    args = parser.parse_args()
    main(args)