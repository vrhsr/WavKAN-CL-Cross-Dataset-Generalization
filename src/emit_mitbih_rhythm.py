"""
Rhythm-Harmonized MIT-BIH Preprocessor.

Produces data/mitbih_rhythm_processed.csv using ONLY beat types that have
rhythm-based equivalents in PTB-XL (AF, bundle branch blocks, PVCs, PACs).

Changes from emit_mitbih.py:
  - Removes Fusion (F) and Paced/Unknown (/, f, Q) categories
  - These have no clear PTB-XL rhythm equivalent
  - Keeps N, L, R, e, j (Normal) and A, a, J, S, V, E (Rhythm Abnormal)
"""
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
from tqdm import tqdm
import os

# CONFIGURATION (LOCKED - same as emit_mitbih.py)
SAMPLE_RATE = 100
WINDOW_SEC = 2.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 250
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 360
DATA_DIR = 'data/mitbih'
OUTPUT_FILE = 'data/mitbih_rhythm_processed.csv'

# RHYTHM-ONLY AAMI MAPPING
# Normal beats
# Rhythm abnormalities only (SVE + VE) — these have PTB-XL equivalents:
#   A, a, J, S → Supraventricular Ectopy (maps to PAC/SVTAC in PTB-XL)
#   V, E → Ventricular Ectopy (maps to PVC in PTB-XL)
# EXCLUDED: F (Fusion), /, f, Q (Paced/Unknown) — no clear PTB-XL rhythm match
AAMI_RHYTHM_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,            # SVE (Abnormal Rhythm)
    'V': 1, 'E': 1,                              # VE (Abnormal Rhythm)
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
    record_path = f"{DATA_DIR}/{record_name}"
    signals, fields = wfdb.rdsamp(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = signals[:, 0]  # Lead II
    signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)

    processed_beats = []
    labels = []
    patient_ids = []

    for i, (sample_idx, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
        if symbol not in AAMI_RHYTHM_MAP:
            continue

        window_raw = int(WINDOW_SEC * SOURCE_RATE)
        half_window = window_raw // 2

        start = sample_idx - half_window
        end = sample_idx + half_window

        if start < 0 or end > len(signal):
            continue

        beat_raw = signal[start:end]
        beat_resampled = resample(beat_raw, WINDOW_SAMPLES)
        beat_norm = z_score_normalize(beat_resampled)

        processed_beats.append(beat_norm)
        labels.append(AAMI_RHYTHM_MAP[symbol])
        patient_ids.append(record_name)

    return processed_beats, labels, patient_ids


def main():
    if not os.path.exists(DATA_DIR):
        print(f"Downloading MIT-BIH to {DATA_DIR}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        wfdb.dl_database('mitdb', DATA_DIR)

    records = [f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.dat')]
    records = sorted(list(set(records)))

    print(f"Found {len(records)} records. Processing with RHYTHM-ONLY labels...")

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

    df = pd.DataFrame(np.array(all_beats))
    df['label'] = all_labels
    df['patient_id'] = all_patients

    # Print class distribution
    n_normal = sum(1 for l in all_labels if l == 0)
    n_abnormal = sum(1 for l in all_labels if l == 1)
    print(f"\n=== RHYTHM-HARMONIZED MIT-BIH ===")
    print(f"Total beats: {len(df)}")
    print(f"Normal: {n_normal} ({100*n_normal/len(df):.1f}%)")
    print(f"Abnormal (rhythm): {n_abnormal} ({100*n_abnormal/len(df):.1f}%)")

    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
