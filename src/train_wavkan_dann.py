import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import argparse
import os
import json
import wandb
import math
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.spline_kan import SplineKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP, InceptionTime, XResNet1D
from src.models.wavkan_dann import HybridDANN
from src.losses import MultilabelFocalLoss
from src.utils import set_seed, get_device, log_gpu_memory, compute_pos_weight

def compute_alpha(epoch, total_epochs):
    p = epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


def train_dann_epoch(model, source_loader, target_loader, 
                      class_criterion, domain_criterion,
                      optimizer, scaler, device, alpha, max_grad_norm=1.0):
    model.train()
    
    total_class_loss = 0
    total_domain_loss = 0
    total_samples = 0
    
    target_iter = iter(target_loader)
    
    for source_data, source_labels in tqdm(source_loader, desc="Training DANN", leave=False):
        source_data, source_labels = source_data.to(device).float(), source_labels.to(device).float()
        batch_size = source_data.size(0)
        
        try:
            target_data, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_data, _ = next(target_iter)
            
        target_data = target_data.to(device).float()
        
        min_batch = min(batch_size, target_data.size(0))
        source_data = source_data[:min_batch]
        source_labels = source_labels[:min_batch]
        target_data = target_data[:min_batch]
        
        # Domain Labels: Source=0, Target=1
        source_domain_labels = torch.zeros(min_batch, 1, device=device)
        target_domain_labels = torch.ones(min_batch, 1, device=device)
        
        optimizer.zero_grad()
        
        with autocast():
            # Source Forward
            class_output, source_domain_output = model(source_data, alpha=alpha)
            class_loss = class_criterion(class_output, source_labels)
            source_domain_loss = domain_criterion(source_domain_output, source_domain_labels)
            
            # Target Forward (Domain only)
            _, target_domain_output = model(target_data, alpha=alpha)
            target_domain_loss = domain_criterion(target_domain_output, target_domain_labels)
            
            domain_loss = (source_domain_loss + target_domain_loss) / 2
            total_loss = class_loss + domain_loss
            
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_class_loss += class_loss.item() * min_batch
        total_domain_loss += domain_loss.item() * min_batch
        total_samples += min_batch
        
    return total_class_loss / total_samples, total_domain_loss / total_samples


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            
            with autocast():
                outputs = model.predict(inputs)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auroc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    except ValueError:
        auroc_macro = 0.0
        
    return f1_macro, auroc_macro


def get_feature_dim(model_name, hidden_dim):
    if model_name in ['wavkan', 'spline_kan']: return hidden_dim
    elif model_name == 'resnet': return 512
    elif model_name == 'vit': return 128
    elif model_name == 'simple_mlp': return hidden_dim
    elif model_name == 'inception': return 128  # based on depth/channels
    elif model_name == 'xresnet': return 1024
    return hidden_dim

def main(args):
    set_seed(args.seed)
    device = get_device()
    num_classes = 5

    wandb.init(
        project="wavkan-cl-dann",
        config=vars(args),
        name=f"dann_{args.model}_{args.seed}",
        tags=["dann", args.model, "adaptation"]
    )

    source_label_path = args.source_file.replace('signals', 'labels') if '.npy' in args.source_file else None
    target_label_path = args.target_file.replace('signals', 'labels') if '.npy' in args.target_file else None

    source_dataset = HarmonizedDataset(args.source_file, label_path=source_label_path)
    target_dataset = HarmonizedDataset(args.target_file, label_path=target_label_path)

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    target_eval_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    if args.model == 'wavkan':
        backbone = WavKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=args.hidden_dim, in_channels=args.in_channels)
    elif args.model == 'spline_kan':
        backbone = SplineKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=args.hidden_dim, in_channels=args.in_channels)
    elif args.model == 'resnet':
        backbone = ResNet1D(in_channels=args.in_channels, num_classes=num_classes, seq_len=1000)
    elif args.model == 'vit':
        backbone = ViT1D(seq_len=1000, num_classes=num_classes, in_channels=args.in_channels)
    elif args.model == 'simple_mlp':
        backbone = SimpleMLP(input_dim=1000, num_classes=num_classes, in_channels=args.in_channels)
    elif args.model == 'inception':
        backbone = InceptionTime(in_channels=args.in_channels, num_classes=num_classes)
    elif args.model == 'xresnet':
        backbone = XResNet1D(in_channels=args.in_channels, num_classes=num_classes)

    feature_dim = get_feature_dim(args.model, args.hidden_dim)
    model = HybridDANN(backbone, args.model, feature_dim=feature_dim).to(device)

    if hasattr(source_dataset, 'y') and source_dataset.y is not None:
        y_tensor = torch.tensor(source_dataset.y, dtype=torch.float32)
        pos_weight = compute_pos_weight(y_tensor, device)
    else:
        pos_weight = None

    class_criterion = MultilabelFocalLoss(gamma=2.0, pos_weight=pos_weight)
    domain_criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_val_f1 = 0.0

    checkpoint_path = f'experiments/checkpoints/dann_{args.model}_seed{args.seed}_best.pt'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(args.epochs):
        alpha = compute_alpha(epoch, args.epochs)
        
        c_loss, d_loss = train_dann_epoch(
            model, source_loader, target_loader, class_criterion, domain_criterion,
            optimizer, scaler, device, alpha
        )
        
        # Zero-shot evaluation on target domain at every epoch
        val_f1_macro, val_auroc_macro = evaluate(model, target_eval_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} - CLoss: {c_loss:.4f}, DLoss: {d_loss:.4f}, alpha: {alpha:.3f} | Target F1: {val_f1_macro:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "class_loss": c_loss,
            "domain_loss": d_loss,
            "alpha": alpha,
            "val_f1_macro": val_f1_macro,
            "val_auroc_macro": val_auroc_macro,
            "lr": optimizer.param_groups[0]['lr']
        })

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

    # Load best zero-shot model
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    final_f1_macro, final_auroc_macro = evaluate(model, target_eval_loader, device)

    results = {
        'config': vars(args),
        'metrics': {
            'final_f1_macro': final_f1_macro,
            'final_auroc_macro': final_auroc_macro,
            'best_val_f1': best_val_f1
        },
        'status': 'completed'
    }

    run_path = f'experiments/runs/dann_{args.model}_seed{args.seed}.json'
    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    with open(run_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Final Zero-Shot DANN F1 Macro: {final_f1_macro:.4f}, AUROC Macro: {final_auroc_macro:.4f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN Hybrid for multi-label cross-domain ECG classification')
    parser.add_argument('--source_file', type=str, required=True)
    parser.add_argument('--target_file', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['wavkan', 'spline_kan', 'resnet', 'vit', 'simple_mlp'], required=True)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
