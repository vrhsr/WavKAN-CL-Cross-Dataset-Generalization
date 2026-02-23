import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
from tqdm import tqdm
import os

# CONFIGURATION (LOCKED)
SAMPLE_RATE = 100      # Target Hz
WINDOW_SEC = 2.5       # Window length
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC) # 250 samples
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 360      # MIT-BIH native
LEAD_IDX = 0           # Lead II is usually index 0 in MIT-BIH
DATA_DIR = 'data/mitbih'
OUTPUT_FILE = 'data/mitbih_processed.csv'

# AAMI MAPPING
# N: Normal, L, R, e, j
# S: A, a, J, S
# V: V, E
# F: F
# Q: /, f, Q
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # SVE (Abnormal)
    'V': 1, 'E': 1,                          # VE (Abnormal)
    'F': 1,                                  # Fusion (Abnormal)
    '/': 1, 'f': 1, 'Q': 1                   # Unknown/Paced (Abnormal)
}

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-6:
        return np.zeros_like(signal)
    return (signal - mean) / std

def process_record(record_name):
    # Load Signal
    record_path = f"{DATA_DIR}/{record_name}"
    signals, fields = wfdb.rdsamp(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    # Extract Lead II
    # MIT-BIH usually has Lead II at index 0
    signal = signals[:, 0] 

    # Filter (0.5 - 40 Hz)
    signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)

    processed_beats = []
    labels = []
    patient_ids = []

    # Process Beats
    for i, (sample_idx, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
        if symbol not in AAMI_MAP:
            continue
            
        # Define Window (Centered)
        # We want 2.5s at 100Hz = 250 samples
        # At 360Hz, 2.5s = 900 samples. 
        # So we take 450 before and 450 after the R-peak
        window_raw = int(WINDOW_SEC * SOURCE_RATE) # 900
        half_window = window_raw // 2
        
        start = sample_idx - half_window
        end = sample_idx + half_window
        
        if start < 0 or end > len(signal):
            continue
            
        beat_raw = signal[start:end]
        
        # Resample (900 -> 250)
        beat_resampled = resample(beat_raw, WINDOW_SAMPLES)
        
        # Z-Score (Per Beat)
        beat_norm = z_score_normalize(beat_resampled)
        
        processed_beats.append(beat_norm)
        labels.append(AAMI_MAP[symbol])
        patient_ids.append(record_name)

    return processed_beats, labels, patient_ids

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Downloading MIT-BIH to {DATA_DIR}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        wfdb.dl_database('mitdb', DATA_DIR)

    records = [f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.dat')]
    records = sorted(list(set(records))) # Unique records
    
    print(f"Found {len(records)} records. Processing...")

    all_beats = []
    all_labels = []
    all_patients = []

    for rec in tqdm(records):
        try:
            beats, lbls, pids = process_record(rec)
            all_beats.extend(beats)
            all_labels.extend(lbls)
            all_patients.extend(pids)
        except Exception as e:
            print(f"Error processing {rec}: {e}")

    # Create DataFrame
    # Columns: label, patient_id, [sample_0, ..., sample_249]
    df = pd.DataFrame(np.array(all_beats))
    df['label'] = all_labels
    df['patient_id'] = all_patients
    
    # Save
    print(f"Saving {len(df)} beats to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
