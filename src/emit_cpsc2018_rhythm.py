"""
CPSC 2018 Dataset Preprocessor (RHYTHM-HARMONIZED).

Preprocessing script for CPSC 2018 that aligns with the rhythm-only ontology
used for MIT-BIH and PTB-XL:
  - Lead: Lead II
  - Sampling Rate: 100 Hz
  - Filter: 0.5 - 40 Hz Butterworth Bandpass
  - Segment Length: 250 samples
  - Label Mapping:
      Normal Sinus Rhythm (1) -> 0 (Normal)
      AF (2), I-AVB (3), LBBB (4), RBBB (5), PAC (6), PVC (7) -> 1 (Abnormal)
      Exclude: STD (8), STE (9) -> Morphological, not rhythm
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
from tqdm import tqdm

DATA_DIR = "data/cpsc2018_raw"
OUTPUT_PATH = "data/cpsc2018_rhythm_processed.csv"
TARGET_LENGTH = 250
TARGET_FS = 100

def apply_bandpass(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='both')
    return filtfilt(b, a, signal)

def process_record(record_path, target_fs=TARGET_FS, target_len=TARGET_LENGTH):
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
    
    # Preliminary bandpass filter
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

def map_rhythm_label(label_code):
    """
    1: Normal -> 0
    2, 3, 4, 5, 6, 7: Rhythm Abnorm -> 1
    8, 9: Morphological -> None (Exclude)
    """
    if label_code == 1:
        return 0
    elif 2 <= label_code <= 7:
        return 1
    else:
        return None  # Exclude STD, STE

def main():
    print("=" * 60)
    print("  CPSC 2018 RHYTHM-HARMONIZED PREPROCESSING")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Raw data directory {DATA_DIR} not found.")
        return
    
    ref_path = os.path.join(DATA_DIR, "REFERENCE.csv")
    ref_df = pd.read_csv(ref_path, header=None, names=['record', 'label'])
    
    print("\nProcessing and Filtering (Rhythm-only)...")
    all_segments = []
    all_labels = []
    all_patients = []
    
    excluded_count = 0
    
    for _, row in tqdm(ref_df.iterrows(), total=len(ref_df)):
        record_name = row['record']
        label = map_rhythm_label(row['label'])
        
        if label is None:
            excluded_count += 1
            continue
            
        record_path = os.path.join(DATA_DIR, record_name)
        segments = process_record(record_path)
        
        for seg in segments:
            all_segments.append(seg)
            all_labels.append(label)
            pid = int(record_name[1:]) if record_name[0] == 'A' else 9999
            all_patients.append(pid)
    
    print(f"\nExcluded {excluded_count} records (ST abnormalities).")
    
    if not all_segments:
        print("ERROR: No segments extracted.")
        return
    
    data_matrix = np.array(all_segments)
    df = pd.DataFrame(data_matrix)
    df['label'] = all_labels
    df['patient_id'] = all_patients
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\nSaved rhythm-harmonized dataset to {OUTPUT_PATH}")
    print(f"Total Segments: {len(df)}")
    print(f"Normal: {df['label'].value_counts()[0]} ({df['label'].value_counts(normalize=True)[0]:.1%})")
    print(f"Abnormal: {df['label'].value_counts()[1]} ({df['label'].value_counts(normalize=True)[1]:.1%})")

if __name__ == "__main__":
    main()
