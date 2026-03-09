import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def main():
    results_dir = 'experiments'
    plots_dir = 'experiments/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get all ablation histories
    files = glob.glob(f'{results_dir}/ablation_*_history_seed*.csv')
    if not files:
        print("No files found!")
        return

    # Dictionary to hold data: {experiment_name: {epoch: [list of f1s]}}
    data = {}
    
    for f in files:
        base = os.path.basename(f)
        # e.g., ablation_wavelet_mexican_hat_history_seed42.csv
        parts = base.replace('_history', '').replace('.csv', '').split('_seed')
        exp_name = parts[0]
        
        df = pd.read_csv(f)
        
        # Determine the target length to pad (50 epochs)
        target_len = 50
        
        # Pad with NaNs if the array is shorter than 50
        val_f1 = np.pad(df['val_f1'].values.astype(float), (0, max(0, target_len - len(df))), constant_values=np.nan)[:target_len]
        val_loss = np.pad(df['val_loss'].values.astype(float), (0, max(0, target_len - len(df))), constant_values=np.nan)[:target_len]
        train_loss = np.pad(df['train_loss'].values.astype(float), (0, max(0, target_len - len(df))), constant_values=np.nan)[:target_len]
        
        if exp_name not in data:
            data[exp_name] = {'val_f1': [], 'val_loss': [], 'train_loss': []}
            
        data[exp_name]['val_f1'].append(val_f1)
        data[exp_name]['val_loss'].append(val_loss)
        data[exp_name]['train_loss'].append(train_loss)

    # Plot Val F1 Comparison
    plt.figure(figsize=(10, 6))
    for exp_name, metrics in data.items():
        arr = np.array(metrics['val_f1'])
        
        # compute mean ignoring NaNs
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_f1 = np.nanmean(arr, axis=0)
            std_f1 = np.nanstd(arr, axis=0)
            
        epochs = np.arange(1, len(mean_f1) + 1)
        
        plt.plot(epochs, mean_f1, label=exp_name, linewidth=2)
        plt.fill_between(epochs, mean_f1 - std_f1, mean_f1 + std_f1, alpha=0.2)
        
        # Print final max score to console for quick review
        best_f1 = np.nanmax(mean_f1)
        print(f"[{exp_name}] Best Mean F1 across 5 seeds: {best_f1:.4f}")

    plt.title('Ablation Study: Validation F1 Score Over Time (Averaged over 5 Seeds)')
    plt.xlabel('Epochs')
    plt.ylabel('Validation F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/ablation_f1_comparison.png', dpi=300)
    print(f"Saved {plots_dir}/ablation_f1_comparison.png")

if __name__ == '__main__':
    main()
