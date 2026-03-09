import pandas as pd
import glob
import os

# Robustness "Clean" is exactly the zero-shot F1 on PTB-XL
models_to_extract = ['resnet', 'vit', 'mlp']

print("Extracting zero-shot F1 from robustness data...")
for model in models_to_extract:
    rob_files = glob.glob(f"experiments/robustness_{model}_seed*.csv")
    for filepath in rob_files:
        seed = filepath.split('seed')[-1].replace('.csv', '')
        
        # Read the robustness CSV
        try:
            df = pd.read_csv(filepath, index_col=0)
            if 'Clean' in df.columns:
                zs_f1 = df['Clean'].iloc[0]
                
                # Create the zero-shot CSV
                zs_df = pd.DataFrame([{
                    'model': model,
                    'seed': int(seed),
                    'zero_shot_acc': 0.0, # Not used in final paper, only F1
                    'zero_shot_f1': zs_f1,
                    'zero_shot_auc': 0.0
                }])
                zs_path = f"experiments/zeroshot_{model}_seed{seed}.csv"
                zs_df.to_csv(zs_path, index=False)
                print(f"Extracted {model} seed {seed}: F1={zs_f1:.4f} -> {zs_path}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

print("Extraction complete.")
