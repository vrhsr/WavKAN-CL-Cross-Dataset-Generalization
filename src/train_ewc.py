import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import argparse
import os
import json
import wandb
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import math

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.spline_kan import SplineKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP, InceptionTime, XResNet1D
from src.losses import MultilabelFocalLoss
from src.utils import set_seed, get_device, log_gpu_memory

def compute_fisher(model, dataloader, criterion, device, num_samples=2000):
    """
    Computes the diagonal Fisher Information Matrix for EWC,
    constrained to `num_samples` to prevent memory/time blowups.
    """
    model.eval()
    fisher_dict = {}
    optparam_dict = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            optparam_dict[name] = param.data.clone()
            fisher_dict[name] = torch.zeros_like(param.data)
            
    samples_processed = 0
    
    print(f"Computing Fisher Information (limit: {num_samples} samples)...")
    for inputs, labels in tqdm(dataloader, desc="Fisher trace", leave=False):
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        
        # Batch size handling
        bs = inputs.size(0)
        
        # Compute gradients sample-by-sample for correct Fisher diagonal
        # To speed up, we can approximate by calculating batch gradients if strictly necessary,
        # but true empirical Fisher is expected gradient squared limit
        for i in range(bs):
            if samples_processed >= num_samples:
                break
                
            model.zero_grad()
            out = model(inputs[i:i+1])
            loss = criterion(out, labels[i:i+1])
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name].data += param.grad.data ** 2
                    
            samples_processed += 1
            
        if samples_processed >= num_samples:
            break
            
    # Average the Fisher Information
    for name in fisher_dict:
        fisher_dict[name] /= samples_processed
        
    return fisher_dict, optparam_dict

def train_ewc_epoch(model, loader, criterion, optimizer, scaler, fisher_dict, optparam_dict, ewc_lambda, device, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(loader, desc="EWC Finetuning", leave=False):
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            task_loss = criterion(outputs, labels)
            
            # EWC Penalty
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and name in fisher_dict:
                    fisher = fisher_dict[name]
                    optparam = optparam_dict[name]
                    ewc_loss += (fisher * (param - optparam).pow(2)).sum()
                    
            loss = task_loss + (ewc_lambda / 2) * ewc_loss
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return running_loss / len(loader), f1_macro

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            
            with autocast():
                outputs = model(inputs)
            
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

def main(args):
    set_seed(args.seed)
    device = get_device()
    num_classes = 5

    wandb.init(
        project="wavkan-cl-ewc",
        config=vars(args),
        name=f"ewc_{args.model}_{args.seed}"
    )

    source_label_path = args.source_file.replace('signals', 'labels') if '.npy' in args.source_file else None
    target_label_path = args.target_file.replace('signals', 'labels') if '.npy' in args.target_file else None

    source_dataset = HarmonizedDataset(args.source_file, label_path=source_label_path)
    target_dataset = HarmonizedDataset(args.target_file, label_path=target_label_path)

    target_train_size = int(0.8 * len(target_dataset))
    target_val_size = len(target_dataset) - target_train_size
    target_train, target_val = random_split(target_dataset, [target_train_size, target_val_size])

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    target_train_loader = DataLoader(target_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    target_val_loader = DataLoader(target_val, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    if args.model == 'wavkan':
        model = WavKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=args.hidden_dim, in_channels=args.in_channels).to(device)
    elif args.model == 'spline_kan':
        model = SplineKANClassifier(input_dim=1000, num_classes=num_classes, hidden_dim=args.hidden_dim, in_channels=args.in_channels).to(device)
    elif args.model == 'resnet':
        model = ResNet1D(in_channels=args.in_channels, num_classes=num_classes, seq_len=1000).to(device)
    elif args.model == 'vit':
        model = ViT1D(seq_len=1000, num_classes=num_classes, in_channels=args.in_channels).to(device)
    elif args.model == 'simple_mlp':
        model = SimpleMLP(input_dim=1000, num_classes=num_classes, in_channels=args.in_channels).to(device)
    elif args.model == 'inception':
        model = InceptionTime(in_channels=args.in_channels, num_classes=num_classes).to(device)
    elif args.model == 'xresnet':
        model = XResNet1D(in_channels=args.in_channels, num_classes=num_classes).to(device)

    # Load Source pre-trained weights
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
        print(f"Loaded source checkpoint from {args.checkpoint}")
    else:
        print("WARNING: EWC requires a source pre-trained model. Checkpoint not found. Proceeding with random weights.")

    criterion = MultilabelFocalLoss(gamma=2.0)
    
    # Measure initial source performance (zero-shot target could also be measured)
    initial_source_f1, _ = evaluate(model, DataLoader(source_dataset, batch_size=args.batch_size), device)
    initial_target_f1, _ = evaluate(model, target_val_loader, device)

    print(f"Initial Source F1: {initial_source_f1:.4f} | Initial Target Val F1: {initial_target_f1:.4f}")

    # Compute Fisher Information (Limiting to 2000 samples)
    fisher_dict, optparam_dict = compute_fisher(model, source_loader, criterion, device, num_samples=2000)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_val_f1 = 0.0
    checkpoint_path = f'experiments/checkpoints/ewc_{args.model}_seed{args.seed}_best.pt'

    for epoch in range(args.epochs):
        train_loss, train_f1 = train_ewc_epoch(
            model, target_train_loader, criterion, optimizer, scaler, 
            fisher_dict, optparam_dict, args.ewc_lambda, device
        )
        
        val_f1_macro, val_auroc_macro = evaluate(model, target_val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Target F1: {train_f1:.4f}, Val F1: {val_f1_macro:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "val_f1_macro": val_f1_macro,
            "val_auroc_macro": val_auroc_macro,
            "lr": optimizer.param_groups[0]['lr']
        })

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

    # Evaluate best model on Full Target and Full Source (Forgetting check)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    
    target_full_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
    final_target_f1, final_target_auroc = evaluate(model, target_full_loader, device)
    
    source_full_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False)
    final_source_f1, _ = evaluate(model, source_full_loader, device)
    
    forgetting = initial_source_f1 - final_source_f1

    results = {
        'config': vars(args),
        'metrics': {
            'initial_source_f1': initial_source_f1,
            'initial_target_val_f1': initial_target_f1,
            'final_target_f1': final_target_f1,
            'final_target_auroc': final_target_auroc,
            'final_source_f1': final_source_f1,
            'forgetting': forgetting
        },
        'status': 'completed'
    }

    run_path = f'experiments/runs/ewc_{args.model}_seed{args.seed}.json'
    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    with open(run_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"EWC Complete! | Target F1: {final_target_f1:.4f} | Forgetting: {forgetting:.4f} (Source F1: {initial_source_f1:.4f} -> {final_source_f1:.4f})")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuous Learning with EWC for ECG')
    parser.add_argument('--source_file', type=str, required=True)
    parser.add_argument('--target_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='Pre-trained source model path')
    parser.add_argument('--model', type=str, choices=['wavkan', 'spline_kan', 'resnet', 'vit', 'simple_mlp', 'inception', 'xresnet'], required=True)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ewc_lambda', type=float, default=5000.0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
