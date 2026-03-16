import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

from src.dataset import SSLAugmentedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D
from src.models.spline_kan import SplineKANClassifier
from src.losses import NTXentLoss

def train_ssl(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"SSL Training on {device}")
    
    # 1. Dataset (PTB-XL, ignore labels)
    dataset = SSLAugmentedDataset(args.data_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    # 2. Model
    # Initialize normally. Forward(contrastive=True) returns projection.
    if args.model == 'wavkan':
        model = WavKANClassifier(input_dim=250, num_classes=2).to(device)
    elif args.model == 'spline_kan':
        model = SplineKANClassifier(input_dim=250, num_classes=2).to(device)
    else:
        raise ValueError("Unknown model for SSL")
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = NTXentLoss(temperature=0.5, device=device)
    
    # 3. Training Loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for (view1, view2) in loop:
            view1 = view1.to(device).float()
            view2 = view2.to(device).float()
            
            optimizer.zero_grad()
            
            # Forward (Projection)
            z1 = model(view1, contrastive=True)
            z2 = model(view2, contrastive=True)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
        # Save Checkpoint at specific epochs
        checkpoints = [100, 300, 500, 1000]
        if (epoch + 1) in checkpoints and (epoch + 1) <= args.epochs:
            if not os.path.exists('experiments/ssl'):
                os.makedirs('experiments/ssl')
            torch.save(model.state_dict(), f"experiments/ssl/{args.model}_epoch{epoch+1}.pth")
            print(f"Saved checkpoint at epoch {epoch+1}.")
        
        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"experiments/ssl/{args.model}_pretrained.pth")
            print("Best model updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--model', type=str, default='wavkan', choices=['wavkan', 'spline_kan'])
    parser.add_argument('--epochs', type=int, default=50) # Long training for SSL
    parser.add_argument('--batch_size', type=int, default=256) # Large batch size for SimCLR
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train_ssl(args)
