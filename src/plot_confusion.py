import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import os

from src.dataset import HarmonizedDataset
from src.models.wavkan import WavKANClassifier
from src.models.baselines import ViT1D

def plot_confusion_matrices():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    print("Loading PTB-XL Zero-Shot Test Set...")
    dataset = HarmonizedDataset('data/ptbxl_processed.csv')
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # 2. Extract Predictions
    results = {}
    
    models = {
        'WavKAN': WavKANClassifier(input_dim=1000, num_classes=5, hidden_dim=128),
        'ViT-1D': ViT1D(seq_len=1000, num_classes=5)
    }
    
    # Load weights
    mapped_names = {'WavKAN': 'wavkan', 'ViT-1D': 'vit'}
    
    for pretty_name, model in models.items():
        weight_path = f"experiments/{mapped_names[pretty_name]}_endpoint.pth"
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
            model.to(device)
            model.eval()
            print(f"Loaded weights for {pretty_name}")
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device).float(), labels.to(device).long()
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            results[pretty_name] = (np.array(all_labels), np.array(all_preds))
        else:
            print(f"WARNING: No weights found for {pretty_name} at {weight_path}")
            
    if not results:
        print("No models evaluated.")
        return
        
    # 3. Plot Confusion Matrices side-by-side
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
        
    for ax, (model_name, (labels, preds)) in zip(axes, results.items()):
        cm = confusion_matrix(labels, preds)
        
        # Calculate percentages
        cm_perc = cm / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotation strings mixing count and percent
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    annot[i, j] = f"{c}\n({p*100:.1f}%)"
                else:
                    annot[i, j] = f"{c}\n({p*100:.1f}%)"
        
        sns.heatmap(cm_perc, annot=annot, fmt='', cmap='Blues', ax=ax, 
                    xticklabels=['Normal (0)', 'Abnormal (1)'], 
                    yticklabels=['Normal (0)', 'Abnormal (1)'],
                    vmin=0, vmax=1, annot_kws={"size": 12, "weight": "bold"})
                    
        ax.set_title(f"{model_name} Zero-Shot Confusion", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    os.makedirs('paper/plots', exist_ok=True)
    save_path = 'paper/plots/confusion_matrices.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices saved to {save_path}")

if __name__ == "__main__":
    plot_confusion_matrices()
