"""
CPSC 2018 Dataset Preprocessor for Cross-Dataset ECG Generalization.

Downloads and preprocesses the CPSC 2018 (ICBEB) dataset from PhysioNet
into the same harmonized format as MIT-BIH and PTB-XL:
  - Single-lead (Lead II), 250-sample segments
  - Binary classification: Normal (0) vs Abnormal (1)
  
CPSC 2018 categories mapped to binary:
  Normal Sinus Rhythm (NSR) -> 0 (Normal)
  All other rhythms (AF, IAVB, LBBB, RBBB, PAC, PVC, STD, STE) -> 1 (Abnormal)

Prerequisites:
  pip install wfdb scipy

Usage:
  python -m src.emit_cpsc2018
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
import glob
from tqdm import tqdm

DATA_DIR = "data/cpsc2018_raw"
OUTPUT_PATH = "data/cpsc2018_processed.csv"
TARGET_LENGTH = 250  # Match MIT-BIH/PTB-XL segment length
TARGET_FS = 100      # Target sampling frequency (FIXED to match others)


def download_cpsc2018():
    """Download CPSC 2018 dataset from PhysioNet."""
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb library not installed. Run: pip install wfdb")
        return False
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Downloading CPSC 2018 dataset from PhysioNet...")
    print("This may take several minutes (~500MB)...")
    
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        wfdb.dl_database('cpsc2018', dl_dir=DATA_DIR)
        print(f"Download complete. Files saved to {DATA_DIR}/")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def load_reference_labels(data_dir):
    """Load the reference labels from REFERENCE.csv."""
    ref_path = os.path.join(data_dir, "REFERENCE.csv")
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference file not found at {ref_path}")
        return None
    
    df = pd.read_csv(ref_path, header=None, names=['record', 'label'])
    return df


def apply_bandpass(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='both')
    return filtfilt(b, a, signal)


def process_record(record_path, target_fs=TARGET_FS, target_len=TARGET_LENGTH):
    """Process a single CPSC 2018 record."""
    try:
        import wfdb
    except ImportError:
        return []
    
    try:
        record = wfdb.rdrecord(record_path)
    except Exception:
        return []
    
    signal = record.p_signal
    fs = record.fs
    
    # Extract Lead II
    lead_names = [name.strip().upper() for name in record.sig_name]
    if 'II' in lead_names:
        lead_idx = lead_names.index('II')
    elif 'MLII' in lead_names:
        lead_idx = lead_names.index('MLII')
    else:
        lead_idx = 1
    
    lead_signal = signal[:, lead_idx]
    lead_signal = lead_signal[~np.isnan(lead_signal)]
    if len(lead_signal) == 0:
        return []
    
    # Final bandpass filter before resampling (to avoid aliasing)
    lead_signal = apply_bandpass(lead_signal, fs)
    
    # Resample to 100Hz
    if fs != target_fs:
        num_samples = int(len(lead_signal) * target_fs / fs)
        lead_signal = resample(lead_signal, num_samples)
    
    # Normalize
    if np.std(lead_signal) > 0:
        lead_signal = (lead_signal - np.mean(lead_signal)) / np.std(lead_signal)
    
    segments = []
    for start in range(0, len(lead_signal) - target_len + 1, target_len):
        segment = lead_signal[start:start + target_len]
        segments.append(segment)
    
    return segments


def map_to_binary(label_code):
    """Map CPSC 2018 label codes to binary: Normal (0) vs Abnormal (1)."""
    return 0 if label_code == 1 else 1


def main():
    """Main preprocessing pipeline for CPSC 2018."""
    print("=" * 60)
    print("  CPSC 2018 DATASET PREPROCESSING (FIXED)")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) < 10:
        success = download_cpsc2018()
        if not success: return
    
    ref_df = load_reference_labels(DATA_DIR)
    if ref_df is None: return
    
    print("\nProcessing records...")
    all_segments = []
    all_labels = []
    all_patients = []
    
    for _, row in tqdm(ref_df.iterrows(), total=len(ref_df)):
        record_name = row['record']
        label = map_to_binary(row['label'])
        
        record_path = os.path.join(DATA_DIR, record_name)
        segments = process_record(record_path)
        
        for seg in segments:
            all_segments.append(seg)
            all_labels.append(label)
            # Use record name base as patient_id (e.g. A0001)
            # Convert to numeric if possible or hash
            pid = int(record_name[1:]) if record_name[0] == 'A' else 9999
            all_patients.append(pid)
    
    if not all_segments:
        print("ERROR: No segments extracted.")
        return
    
    data_matrix = np.array(all_segments)
    df = pd.DataFrame(data_matrix)
    df['label'] = all_labels
    df['patient_id'] = all_patients
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Class balance: Normal={df['label'].value_counts(normalize=True)[0]:.1%}")
    print("\nDone! You can now use this dataset with:")
    print("  python -m src.train --model wavkan --mit_file data/cpsc2018_processed.csv")


if __name__ == "__main__":
    main()
