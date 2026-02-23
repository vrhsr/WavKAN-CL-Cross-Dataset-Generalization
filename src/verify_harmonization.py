import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, ks_2samp
from scipy.signal import welch

MIT_FILE = 'data/mitbih_processed.csv'
PTB_FILE = 'data/ptbxl_processed.csv'
PLOT_DIR = 'experiments/verification'
SAMPLE_RATE = 100

def get_signal_columns(df):
    """
    Robustly identifies signal columns.
    Assumes signal columns are numeric strings '0', '1', ... or just default integer columns.
    Excludes known metadata columns.
    """
    exclude = ['label', 'patient_id', 'Unnamed: 0']
    cols = [c for c in df.columns if c not in exclude]
    # Verify these are numeric
    # If they are strings like '0', '1', it works.
    return cols

def waveform_similarity(a, b, name):
    """
    Computes Pearson Correlation between two mean waveforms.
    """
    r, p = pearsonr(a, b)
    print(f"{name} | Pearson r = {r:.4f}, p = {p:.2e}")
    return r, p

def amplitude_ks_test(df_mit, df_ptb, sig_cols, name):
    """
    Computes Kolmogorov-Smirnov test on amplitude distributions.
    Comparing a subset (flattened) to avoid memory issues if dataset is huge.
    """
    # Take a random sample of 1000 beats from each to speed up KS test
    mit_sample = df_mit[sig_cols].sample(n=min(1000, len(df_mit)), random_state=42).values.flatten()
    ptb_sample = df_ptb[sig_cols].sample(n=min(1000, len(df_ptb)), random_state=42).values.flatten()
    
    ks, p = ks_2samp(mit_sample, ptb_sample)
    print(f"{name} | Amplitude KS-stat = {ks:.4f}, p = {p:.2e}")
    return ks, p

def plot_psd(df_mit, df_ptb, sig_cols):
    """
    Plots Power Spectral Density comparison.
    """
    print("Computing PSD...")
    # Compute mean PSD for first 1000 beats
    n_beats = 1000
    
    mit_signals = df_mit[sig_cols].values[:n_beats]
    ptb_signals = df_ptb[sig_cols].values[:n_beats]
    
    f, mit_psd = welch(mit_signals, fs=SAMPLE_RATE, nperseg=128, axis=1)
    _, ptb_psd = welch(ptb_signals, fs=SAMPLE_RATE, nperseg=128, axis=1)
    
    mean_mit_psd = np.mean(mit_psd, axis=0)
    mean_ptb_psd = np.mean(ptb_psd, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(f, mean_mit_psd, label="MIT-BIH (Source)")
    plt.semilogy(f, mean_ptb_psd, label="PTB-XL (Target)", linestyle="--")
    plt.title("Power Spectral Density Comparison (Log Scale)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    save_path = f"{PLOT_DIR}/psd_comparison.png"
    plt.savefig(save_path)
    print(f"Saved PSD comparison plot to {save_path}")

def plot_mean_waveform(df_mit, df_ptb, sig_cols):
    """
    Plots mean waveforms for Normal and Abnormal classes.
    Includes Pearson correlation in the plot title.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Class 0 (Normal)
    mit_norm = df_mit[df_mit['label'] == 0][sig_cols].mean(axis=0).values
    ptb_norm = df_ptb[df_ptb['label'] == 0][sig_cols].mean(axis=0).values
    
    # Class 1 (Abnormal)
    mit_abn = df_mit[df_mit['label'] == 1][sig_cols].mean(axis=0).values
    ptb_abn = df_ptb[df_ptb['label'] == 1][sig_cols].mean(axis=0).values
    
    # Statistics
    r_norm, _ = waveform_similarity(mit_norm, ptb_norm, "Normal Mean Waveform")
    r_abn, _ = waveform_similarity(mit_abn, ptb_abn, "Abnormal Mean Waveform")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Normal Beats (r={r_norm:.3f})")
    plt.plot(mit_norm, label='MIT-BIH')
    plt.plot(ptb_norm, label='PTB-XL', linestyle='--')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title(f"Abnormal Beats (r={r_abn:.3f})")
    plt.plot(mit_abn, label='MIT-BIH')
    plt.plot(ptb_abn, label='PTB-XL', linestyle='--')
    plt.legend()
    plt.grid(True)
    
    save_path = f"{PLOT_DIR}/mean_waveforms.png"
    plt.savefig(save_path)
    print(f"Saved Mean Waveform plot to {save_path}")
    
    return mit_norm, ptb_norm, mit_abn, ptb_abn

def main():
    if not os.path.exists(MIT_FILE) or not os.path.exists(PTB_FILE):
        print("Data files not found. Waiting for processing to complete...")
        return

    print("Loading Data...")
    df_mit = pd.read_csv(MIT_FILE)
    df_ptb = pd.read_csv(PTB_FILE)
    
    sig_cols = get_signal_columns(df_mit)
    print(f"Identified {len(sig_cols)} signal columns.")
    
    print("\n--- 1. Statistical Verification (KS-Test) ---")
    amplitude_ks_test(df_mit, df_ptb, sig_cols, "Global Amplitude")
    
    print("\n--- 2. Waveform Verification (Pearson) ---")
    plot_mean_waveform(df_mit, df_ptb, sig_cols)
    
    print("\n--- 3. Frequency Domain Verification (PSD) ---")
    plot_psd(df_mit, df_ptb, sig_cols)
    
    print("\nVerification Complete. Check plots in experiments/verification/")

if __name__ == "__main__":
    main()
