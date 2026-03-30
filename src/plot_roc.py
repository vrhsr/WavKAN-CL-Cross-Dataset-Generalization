import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader, Subset
import os

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ResNet1D, ViT1D, SimpleMLP
from src.models.spline_kan import SplineKANClassifier
from src.models.dann import DANN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'experiments/plots'

def load_model(name, device):
    """Re-instantiate and load model weights."""
    if name == 'wavkan':
        model = WavKANClassifier(input_dim=1000, num_classes=5, hidden_dim=128)
    elif name == 'spline_kan':
        model = SplineKANClassifier(input_dim=1000, num_classes=5)
    elif name == 'resnet':
        model = ResNet1D(in_channels=12, num_classes=5)
    elif name == 'vit':
        model = ViT1D(seq_len=1000, num_classes=5)
    elif name == 'mlp':
        model = SimpleMLP(input_dim=1000, num_classes=5)
    elif name == 'dann':
        model = DANN(in_channels=12, num_classes=5, feature_dim=256)
    else:
        raise ValueError(f"Unknown model: {name}")
    
    path = f"experiments/{name}_endpoint.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        return model
    else:
        print(f"Warning: Checkpoint for {name} not found at {path}")
        return None

def get_predictions(model, loader, device):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device).float()
            
            if hasattr(model, 'predict'):
                outputs = model.predict(inputs)
            else:
                outputs = model(inputs)
                
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
            # Limit the size for faster ROC generation while keeping representation
            if len(all_labels) > 5000:
                break
                
    return np.array(all_labels), np.array(all_probs)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading PTB-XL Target Dataset for ROC evaluation...")
    target_dataset = HarmonizedDataset('data/ptbxl_processed.csv')
    
    # Use a large enough subset to get smooth curves
    np.random.seed(42)
    indices = np.random.choice(len(target_dataset), min(5000, len(target_dataset)), replace=False)
    subset = Subset(target_dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    
    models = ['wavkan', 'spline_kan', 'resnet', 'vit', 'dann']
    model_labels = {
        'wavkan': 'WavKAN',
        'spline_kan': 'Spline-KAN',
        'resnet': 'ResNet-1D',
        'vit': 'ViT-1D',
        'dann': 'DANN (Adaptation)'
    }
    
    results = {}
    
    for m_name in models:
        print(f"Processing {m_name}...")
        model = load_model(m_name, DEVICE)
        if model is None:
            continue
            
        y_true, y_probs = get_predictions(model, loader, DEVICE)
        results[m_name] = (y_true, y_probs)
    
    if not results:
        print("No models found. Exiting.")
        return
        
    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    for m_name, (y_true, y_probs) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_labels[m_name]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) on Target Domain', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300)
    print(f"Saved ROC Curve to {OUTPUT_DIR}/roc_curve.png")
    
    # Plot Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for m_name, (y_true, y_probs) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        plt.plot(recall, precision, lw=2, label=f'{model_labels[m_name]} (AP = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title('Precision-Recall Curve on Target Domain', fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curve.png'), dpi=300)
    print(f"Saved PR Curve to {OUTPUT_DIR}/pr_curve.png")

if __name__ == "__main__":
    main()
