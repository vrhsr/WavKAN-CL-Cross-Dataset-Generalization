import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Concatenate multiple ECG datasets for LODO training.")
    parser.add_argument('--signals', nargs='+', required=True, help="List of signal .npy files")
    parser.add_argument('--labels', nargs='+', required=True, help="List of label .npy files")
    parser.add_argument('--out_sig', type=str, required=True, help="Output signal .npy")
    parser.add_argument('--out_lab', type=str, required=True, help="Output label .npy")
    args = parser.parse_args()
    
    if len(args.signals) != len(args.labels):
        raise ValueError("Must provide same number of signal and label files")
        
    all_sig, all_lab = [], []
    for s_path, l_path in zip(args.signals, args.labels):
        print(f"Loading {s_path}...")
        all_sig.append(np.load(s_path))
        all_lab.append(np.load(l_path))
        
    merged_sig = np.concatenate(all_sig, axis=0)
    merged_lab = np.concatenate(all_lab, axis=0)
    
    print(f"Merged signals shape: {merged_sig.shape}")
    print(f"Merged labels shape: {merged_lab.shape}")
    
    os.makedirs(os.path.dirname(args.out_sig), exist_ok=True)
    np.save(args.out_sig, merged_sig)
    np.save(args.out_lab, merged_lab)
    print(f"Saved to {args.out_sig} and {args.out_lab}")

if __name__ == "__main__":
    main()
