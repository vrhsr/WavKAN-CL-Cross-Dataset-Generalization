import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

RESULTS_DIR = 'experiments'
PLOT_DIR = 'experiments/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

def plot_robustness():
    """
    Aggregates robustness_{model}.csv files and plots SNR vs F1.
    """
    pattern = os.path.join(RESULTS_DIR, "robustness_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("No robustness results found.")
        return

    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        dfs.append(df)
    
    if not dfs:
        return

    full_df = pd.concat(dfs)
    
    # Transpose for plotting: Index=Model, Cols=SNR
    # We want X-axis = SNR, Y-axis = F1, Hue = Model
    
    # Melt
    df_melt = full_df.reset_index().melt(id_vars='index', var_name='SNR', value_name='F1 Score')
    df_melt.rename(columns={'index': 'Model'}, inplace=True)
    
    # Reorder SNR levels manually if needed to ensure decreasing quality order or increasing?
    # Labels are "Clean", "20dB", "15dB"...
    # Let's sort them.
    order = ['Clean', '20dB', '15dB', '10dB', '5dB', '0dB']
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melt, x='SNR', y='F1 Score', hue='Model', marker='o', linewidth=2.5, sort=False)
    # Enforce order
    plt.gca().set_xticks(range(len(order)))
    plt.gca().set_xticklabels(order)
    
    plt.title("Model Robustness to Noise (Zero-Shot)")
    plt.ylabel("F1 Score")
    plt.xlabel("Signal-to-Noise Ratio (Lower is worse)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/robustness_curve.png")
    print("Saved robustness_curve.png")

def plot_fewshot():
    """
    Aggregates fewshot_{model}.csv and plots Samples vs F1.
    """
    pattern = os.path.join(RESULTS_DIR, "fewshot_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("No few-shot results found.")
        return

    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        dfs.append(df)
        
    full_df = pd.concat(dfs)
    
    df_melt = full_df.reset_index().melt(id_vars='index', var_name='Shots', value_name='F1 Score')
    df_melt.rename(columns={'index': 'Model'}, inplace=True)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melt, x='Shots', y='F1 Score', hue='Model', marker='o', linewidth=2.5)
    
    plt.title("Few-Shot Adaptation Efficiency")
    plt.ylabel("Target F1 Score")
    plt.xlabel("Number of Target Labels Used")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/few_shot_curve.png")
    print("Saved few_shot_curve.png")

if __name__ == "__main__":
    print("Generating Plots...")
    try:
        plot_robustness()
    except Exception as e:
        print(f"Error plotting robustness: {e}")
        
    try:
        plot_fewshot()
    except Exception as e:
        print(f"Error plotting fewshow: {e}")
