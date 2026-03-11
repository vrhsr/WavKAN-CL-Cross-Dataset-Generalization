import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def plot_fewshot_curves():
    # Data extracted directly from Table II of the manuscript
    data = {
        'Shots': [10, 50, 100, 500],
        'ResNet-1D': [0.328, 0.486, 0.572, 0.729],
        'ViT-1D': [0.566, 0.531, 0.575, 0.688],
        'DANN': [0.358, 0.425, 0.509, 0.767],
        'Spline-KAN': [0.424, 0.652, 0.674, 0.770],
        'WavKAN': [0.319, 0.473, 0.547, 0.706],
        'SimpleMLP': [0.343, 0.250, 0.198, 0.404]
    }
    
    # Colors matching the manuscript narrative
    colors = {
        'ResNet-1D': '#95a5a6',     # Grey
        'ViT-1D': '#e74c3c',        # Red
        'DANN': '#9b59b6',          # Purple
        'Spline-KAN': '#2ecc71',    # Green (Champion of adaptation)
        'WavKAN': '#3498db',        # Blue (Champion of efficiency)
        'SimpleMLP': '#34495e'      # Dark grey
    }
    
    # Line styles
    styles = {
        'ResNet-1D': '--',
        'ViT-1D': '-',
        'DANN': ':',
        'Spline-KAN': '-',
        'WavKAN': '-',
        'SimpleMLP': '-.'
    }
    
    # Markers
    markers = {
        'ResNet-1D': 'v',
        'ViT-1D': 's',
        'DANN': 'x',
        'Spline-KAN': '*',
        'WavKAN': 'o',
        'SimpleMLP': 'd'
    }

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    for model in data.keys():
        if model != 'Shots':
            lw = 3 if model in ['Spline-KAN', 'WavKAN', 'ViT-1D'] else 2
            alpha = 1.0 if model in ['Spline-KAN', 'WavKAN', 'ViT-1D'] else 0.7
            plt.plot(data['Shots'], data[model], label=model, 
                     color=colors[model], linestyle=styles[model], 
                     marker=markers[model], markersize=10, linewidth=lw, alpha=alpha)

    plt.xscale('log')
    plt.xticks(data['Shots'], data['Shots'], fontsize=12)
    plt.yticks(np.arange(0.1, 0.9, 0.1), fontsize=12)
    
    plt.xlabel('Target Domain Labeled Samples (k-shots, Log Scale)', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('F1 Score on PTB-XL', fontsize=14, fontweight='bold', labelpad=10)
    plt.title('Few-Shot Adaptation Trajectories', fontsize=18, fontweight='bold', pad=15)
    
    plt.legend(fontsize=12, loc='lower right', frameon=True, fancybox=True, framealpha=0.9, shadow=True)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save the figure
    os.makedirs('paper/plots', exist_ok=True)
    save_path = 'paper/plots/fewshot_curves.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Few-shot curve generated and saved to {save_path}")

if __name__ == "__main__":
    plot_fewshot_curves()
