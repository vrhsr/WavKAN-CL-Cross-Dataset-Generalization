import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    # Load complexity data (Params and MACs)
    comp_df = pd.read_csv('experiments/model_complexity.csv')
    
    # Load few-shot summary data (500-shot F1)
    # The columns are: model, 10-shot_mean, ..., 500-shot_mean, 500-shot_std, 500-shot
    fs_df = pd.read_csv('experiments/results_fewshot_summary.csv')
    
    # Rename model names in fs_df to match comp_df if necessary
    name_map = {
        'wavkan': 'WavKAN',
        'resnet': 'ResNet-1D',
        'vit': 'ViT-1D',
        'spline_kan': 'Spline-KAN',
        'mlp': 'SimpleMLP',
        'dann': 'DANN'
    }
    
    # The first column in results_fewshot_summary is unnamed, typically 'Unnamed: 0'
    first_col = fs_df.columns[0]
    fs_df['model_mapped'] = fs_df[first_col].map(name_map)
    
    # Merge datasets
    merged = pd.merge(comp_df, fs_df, left_on='name', right_on='model_mapped', how='inner')
    
    if merged.empty:
        print("Merge failed.")
        return
        
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    # Scatter plot
    # x = params, y = 500-shot F1
    colors = sns.color_palette("Set2", len(merged))
    
    for i, row in merged.iterrows():
        plt.scatter(row['params']/1e6, row['500-shot_mean'], 
                    s=300, color=colors[i], edgecolor='k', alpha=0.9, label=row['name'])

        
        # Add text labels slightly offset
        plt.text(row['params']/1e6, row['500-shot_mean'] - 0.015, row['name'], 
                 horizontalalignment='center', fontsize=12, fontweight='bold')
                 
    # We want to identify the Pareto frontier (max F1 for given params or min params for given F1)
    pts = merged[['params', '500-shot_mean']].values
    pts[:, 0] = pts[:, 0] / 1e6 # Params in Millions
    # Sort by params ascending
    order = np.argsort(pts[:, 0])
    pts_sorted = pts[order]
    
    pareto_pts = []
    max_f1 = -1
    for pt in pts_sorted:
        if pt[1] >= max_f1:
            pareto_pts.append(pt)
            max_f1 = pt[1]
            
    pareto_pts = np.array(pareto_pts)
    
    plt.plot(pareto_pts[:, 0], pareto_pts[:, 1], 'k--', alpha=0.5, zorder=0, label='Pareto Frontier')
    
    plt.xscale('log')
    plt.xlabel('Number of Parameters (Millions) [Log Scale]', fontweight='bold')
    plt.ylabel('500-Shot Cross-Dataset F1', fontweight='bold')
    plt.title('Parameter Efficiency vs. Adaptation F1', fontweight='bold', pad=15)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs('paper/plots', exist_ok=True)
    plt.savefig('paper/plots/pareto_efficiency.png', dpi=300, bbox_inches='tight')
    print("Saved pareto_efficiency.png")

if __name__ == "__main__":
    main()
