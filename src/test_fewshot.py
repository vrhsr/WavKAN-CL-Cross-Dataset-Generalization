import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import argparse
import pandas as pd
import numpy as np
from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from src.models.dann import DANN
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def get_k_shot_indices(dataset, k, seed=42):
    """
    Multilabel Stratified sampling: selects precisely k samples maximizing class balance.
    """
    labels = np.array([dataset[i][1].numpy() for i in range(len(dataset))])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, train_size=min(k, len(dataset)-1), random_state=seed)
    train_idx, _ = next(msss.split(np.zeros(len(labels)), labels))
    return train_idx

def fine_tune(model, train_loader, epochs=10, lr=1e-4, device='cpu', linear_probe=False):
    """
    Fine-tunes the model on the small support set.
    If linear_probe=True, freezes all layers except the classifier head.
    """
    if linear_probe:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier head only
        if hasattr(model, 'label_classifier'):
            for param in model.label_classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'mlp_head'):
            for param in model.mlp_head.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            optimizer.zero_grad()
            if isinstance(model, DANN):
                outputs = model.predict(inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            if isinstance(model, DANN):
                outputs = model.predict(inputs)
            else:
                outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    # Target dataset (PTB-XL)
    full_dataset = HarmonizedDataset(args.ptb_file)
    
    # K-Shot settings
    k_shots = [10, 50, 100, 500] 
    results = {}
    
    print(f"\n--- Starting Few-Shot Adaptation: {args.model} ---")
    
    for k in k_shots:
        print(f"Running {k}-shot adaptation...")
        
        # 1.a. Re-initialize model with Pre-trained Weights (Reset for each k)
        if args.model == 'wavkan':
            model = WavKANClassifier(input_dim=1000, num_classes=5,
                                   hidden_dim=args.hidden_dim,
                                   in_channels=12).to(device)
        elif args.model == 'resnet':
            model = ResNet1D(in_channels=12, num_classes=5, seq_len=1000).to(device)
        elif args.model == 'vit':
            model = ViT1D(seq_len=1000, num_classes=5, in_channels=12).to(device)
        elif args.model == 'mlp':
            model = SimpleMLP(input_dim=1000, num_classes=5, in_channels=12).to(device)
        elif args.model == 'spline_kan':
            model = SplineKANClassifier(input_dim=1000, num_classes=5, hidden_dim=args.hidden_dim, in_channels=12).to(device)
            
        # Load Zero-Shot Weights (Starting point)
        # Load Weights (Zero-Shot or SSL)
        if args.pretrained_path:
            model_path = args.pretrained_path
        else:
            model_path = f"experiments/{args.model}_endpoint.pth"
            
        print(f"Loading weights from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=device)
            # Handle SSL weights which might have 'projection_head' distinct from classifier
            # But here we load state_dict. STRICT=False because SSL head might match or not?
            # SSL model has same architecture, so strict loading should work 
            # UNLESS classifier head shapes mismatch (2 classes vs distinct).
            # WavKAN SSL has 2 output classes in classifier?
            # train_ssl.py: model = WavKANClassifier(..., num_classes=5)
            # So architecture is identical.
            model.load_state_dict(state_dict, strict=False)
        except FileNotFoundError:
            print(f"Weights file {model_path} not found. Skipping.")
            return

        # 1.b. Create Support Set (Few-Shot Train)
        indices = get_k_shot_indices(full_dataset, k)
        train_subset = Subset(full_dataset, indices)
        train_loader = DataLoader(train_subset, batch_size=min(16, k), shuffle=True)
        
        # 1.c. Create Query Set (exclude training samples to prevent leakage)
        all_indices = set(range(len(full_dataset)))
        test_indices = list(all_indices - set(indices))
        test_subset = Subset(full_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
        
        # 2. Fine-tune (or linear probe)
        probe_mode = 'linear_probe' if args.linear_probe else 'fine_tune'
        model = fine_tune(model, train_loader, epochs=10, lr=1e-4, device=device, 
                         linear_probe=args.linear_probe)
        
        # 3. Evaluate
        f1 = evaluate(model, test_loader, device)
        results[f"{k}-shot"] = f1
        print(f"-> F1: {f1:.4f}")

    # Save Results
    df_res = pd.DataFrame([results])
    df_res.index = [args.model]
    save_path = f"experiments/fewshot_{args.model}.csv"
    df_res.to_csv(save_path)
    print(f"Saved results to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--model', type=str, required=True, choices=['wavkan', 'resnet', 'vit', 'spline_kan', 'mlp', 'dann'])
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pre-trained weights')
    parser.add_argument('--linear_probe', action='store_true', help='Freeze encoder, train classifier only')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for WavKAN')
    parser.add_argument('--wavelet_type', type=str, default='mexican_hat', choices=['mexican_hat', 'morlet'])
    parser.add_argument('--depth', type=int, default=3, help='Network depth for WavKAN')
    args = parser.parse_args()
    main(args)
