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
from scipy.signal import resample
import glob

DATA_DIR = "data/cpsc2018_raw"
OUTPUT_PATH = "data/cpsc2018_processed.csv"
TARGET_LENGTH = 250  # Match MIT-BIH/PTB-XL segment length
TARGET_FS = 250      # Target sampling frequency


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
        wfdb.dl_database('cpsc2018', dl_dir=DATA_DIR)
        print(f"Download complete. Files saved to {DATA_DIR}/")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print("  https://physionet.org/content/cpsc2018/1.0.0/")
        print(f"  Extract to: {DATA_DIR}/")
        return False


def load_reference_labels(data_dir):
    """Load the reference labels from REFERENCE.csv."""
    ref_path = os.path.join(data_dir, "REFERENCE.csv")
    if not os.path.exists(ref_path):
        # Try alternative name
        ref_path = os.path.join(data_dir, "REFERENCE.csv")
    
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference file not found at {ref_path}")
        return None
    
    df = pd.read_csv(ref_path, header=None, names=['record', 'label'])
    return df


def process_record(record_path, target_fs=TARGET_FS, target_len=TARGET_LENGTH):
    """Process a single CPSC 2018 record.
    
    Extracts Lead II, resamples to target_fs, and segments into
    fixed-length windows of target_len samples.
    """
    try:
        import wfdb
    except ImportError:
        return []
    
    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        print(f"  Warning: Could not read {record_path}: {e}")
        return []
    
    signal = record.p_signal  # (samples, leads)
    fs = record.fs
    
    # Extract Lead II (usually index 1, but check channel names)
    lead_names = [name.strip().upper() for name in record.sig_name]
    if 'II' in lead_names:
        lead_idx = lead_names.index('II')
    elif 'MLII' in lead_names:
        lead_idx = lead_names.index('MLII')
    else:
        lead_idx = 1  # Default to second lead
    
    lead_signal = signal[:, lead_idx]
    
    # Remove NaN values
    lead_signal = lead_signal[~np.isnan(lead_signal)]
    if len(lead_signal) == 0:
        return []
    
    # Resample to target frequency
    if fs != target_fs:
        num_samples = int(len(lead_signal) * target_fs / fs)
        lead_signal = resample(lead_signal, num_samples)
    
    # Normalize (z-score)
    if np.std(lead_signal) > 0:
        lead_signal = (lead_signal - np.mean(lead_signal)) / np.std(lead_signal)
    
    # Segment into fixed-length windows
    segments = []
    for start in range(0, len(lead_signal) - target_len + 1, target_len):
        segment = lead_signal[start:start + target_len]
        segments.append(segment)
    
    return segments


def map_to_binary(label_code):
    """Map CPSC 2018 label codes to binary: Normal (0) vs Abnormal (1).
    
    CPSC 2018 codes:
      1 = Normal Sinus Rhythm (NSR) -> 0
      2 = AF (Atrial Fibrillation) -> 1
      3 = I-AVB (First-degree AV Block) -> 1
      4 = LBBB (Left Bundle Branch Block) -> 1
      5 = RBBB (Right Bundle Branch Block) -> 1
      6 = PAC (Premature Atrial Contraction) -> 1
      7 = PVC (Premature Ventricular Contraction) -> 1
      8 = STD (ST-segment Depression) -> 1
      9 = STE (ST-segment Elevation) -> 1
    """
    return 0 if label_code == 1 else 1


def main():
    """Main preprocessing pipeline for CPSC 2018."""
    print("=" * 60)
    print("  CPSC 2018 DATASET PREPROCESSING")
    print("=" * 60)
    
    # Step 1: Check if raw data exists, download if not
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) < 10:
        print("\nRaw data not found. Attempting download...")
        success = download_cpsc2018()
        if not success:
            return
    
    # Step 2: Load reference labels
    print("\nLoading reference labels...")
    ref_df = load_reference_labels(DATA_DIR)
    if ref_df is None:
        return
    
    print(f"Found {len(ref_df)} records")
    label_counts = ref_df['label'].value_counts().sort_index()
    print("Label distribution:")
    for label, count in label_counts.items():
        binary = "Normal" if label == 1 else "Abnormal"
        print(f"  Code {label} ({binary}): {count} records")
    
    # Step 3: Process all records
    print("\nProcessing records...")
    all_segments = []
    all_labels = []
    
    for _, row in ref_df.iterrows():
        record_name = row['record']
        label = map_to_binary(row['label'])
        
        record_path = os.path.join(DATA_DIR, record_name)
        segments = process_record(record_path)
        
        for seg in segments:
            all_segments.append(seg)
            all_labels.append(label)
    
    if not all_segments:
        print("ERROR: No segments extracted. Check the data directory.")
        return
    
    # Step 4: Create DataFrame and save
    print(f"\nTotal segments: {len(all_segments)}")
    print(f"Normal: {all_labels.count(0)}, Abnormal: {all_labels.count(1)}")
    
    data_matrix = np.array(all_segments)
    label_array = np.array(all_labels)
    
    # Same format as MIT-BIH/PTB-XL: columns 0-249 are features, column 250 is label
    df = pd.DataFrame(data_matrix)
    df[TARGET_LENGTH] = label_array
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, header=False)
    
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: ({len(df)}, {df.shape[1]})")
    print(f"Class balance: Normal={label_array.sum()==0:.1%} / Abnormal={label_array.mean():.1%}")
    print("\nDone! You can now use this dataset with:")
    print("  python -m src.train --model wavkan --mit_file data/cpsc2018_processed.csv")


if __name__ == "__main__":
    main()
