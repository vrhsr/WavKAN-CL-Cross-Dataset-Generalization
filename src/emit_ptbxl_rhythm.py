"""
Rhythm-Harmonized PTB-XL Preprocessor.

Produces data/ptbxl_rhythm_processed.csv using ONLY rhythm-based SCP codes
that have analogues in MIT-BIH beat annotations.

Changes from emit_ptbxl.py:
  - Uses SCP rhythm codes instead of diagnostic_class
  - Normal: NORM (Normal Sinus Rhythm) → 0
  - Abnormal (rhythm-only): AFIB, AFLT, SVTAC, PSVT, 1AVB, 2AVB, 3AVB,
    LBBB, RBBB, CLBBB, CRBBB, PVC, BIGU, TRIGU, PAC, PACE → 1
  - EXCLUDES: MI, STTC, HYP (structural/morphological — no MIT-BIH equivalent)

This eliminates the label ontology confound where "Abnormal" meant different
things in MIT-BIH (rhythm disorders) vs PTB-XL (structural conditions).
"""
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
import ast
from tqdm import tqdm
import os

# CONFIGURATION (LOCKED - same as emit_ptbxl.py)
SAMPLE_RATE = 100
WINDOW_SEC = 2.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 250
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 500
LEAD_IDX = 1  # Lead II
DATA_DIR = 'data/ptb-xl-1.0.3'
OUTPUT_FILE = 'data/ptbxl_rhythm_processed.csv'

# RHYTHM-ONLY SCP CODES
# These are the PTB-XL rhythm/conduction codes that have clear MIT-BIH equivalents
NORMAL_CODES = {'NORM'}

ABNORMAL_RHYTHM_CODES = {
    # Atrial arrhythmias (MIT-BIH: A, a, J, S → SVE)
    'AFIB',   # Atrial Fibrillation
    'AFLT',   # Atrial Flutter
    'SVTAC',  # Supraventricular Tachycardia
    'PSVT',   # Paroxysmal SVT
    'PAC',    # Premature Atrial Contraction
    'SARRH',  # Sinus Arrhythmia
    'SBRAD',  # Sinus Bradycardia
    'STACH',  # Sinus Tachycardia
    # Ventricular arrhythmias (MIT-BIH: V, E → VE)
    'PVC',    # Premature Ventricular Contraction
    'BIGU',   # Bigeminy
    'TRIGU',  # Trigeminy
    # Conduction disturbances (MIT-BIH: L, R are bundle branch variants)
    'LBBB',   # Left Bundle Branch Block
    'RBBB',   # Right Bundle Branch Block
    'CLBBB',  # Complete LBBB
    'CRBBB',  # Complete RBBB
    'LAFB',   # Left Anterior Fascicular Block
    'LPFB',   # Left Posterior Fascicular Block
    '1AVB',   # First-degree AV Block
    '2AVB',   # Second-degree AV Block
    '3AVB',   # Third-degree AV Block
    'WPW',    # Wolff-Parkinson-White
}


def map_rhythm_label(scp_codes):
    """Map SCP codes to binary using rhythm-only criteria.
    
    Returns:
        0 if Normal (NORM present, no rhythm abnormalities)
        1 if any rhythm abnormality code is present
       -1 if neither (exclude from dataset)
    """
    code_keys = set(scp_codes.keys())
    
    # Check for rhythm abnormalities first (takes priority)
    if code_keys & ABNORMAL_RHYTHM_CODES:
        return 1
    
    # Check for Normal
    if code_keys & NORMAL_CODES:
        return 0
    
    # Neither rhythm abnormality nor normal → exclude
    # (This excludes pure MI, STTC, HYP records with no rhythm component)
    return -1


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


def download_ptbxl_zip():
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    zip_path = os.path.join("data", "ptbxl.zip")
    extract_to = "data"

    print(f"Downloading Full PTB-XL from {url}...")
    import requests

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(zip_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    print("Extracting...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    extracted_dirs = [d for d in os.listdir(extract_to) if 'ptb-xl' in d and os.path.isdir(os.path.join(extract_to, d))]
    for d in extracted_dirs:
        if d != 'ptb-xl-1.0.3':
            if 'large-publicly' in d:
                src = os.path.join(extract_to, d)
                dst = os.path.join(extract_to, 'ptb-xl-1.0.3')
                if not os.path.exists(dst):
                    os.rename(src, dst)
                break

    os.remove(zip_path)


def process_ptbxl_rhythm():
    # 1. Download if missing
    db_csv_path = os.path.join(DATA_DIR, 'ptbxl_database.csv')
    if not os.path.exists(db_csv_path):
        download_ptbxl_zip()

    print(f"Loading PTB-XL Metadata from {DATA_DIR}...")
    Y = pd.read_csv(db_csv_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    processed_beats = []
    labels = []
    patient_ids = []
    
    excluded_structural = 0
    included_normal = 0
    included_abnormal = 0

    for ecg_id, row in tqdm(Y.iterrows(), total=len(Y)):
        # 1. Get rhythm-only label
        lbl = map_rhythm_label(row.scp_codes)
        if lbl == -1:
            excluded_structural += 1
            continue

        if lbl == 0:
            included_normal += 1
        else:
            included_abnormal += 1

        # 2. Path
        record_path = os.path.join(DATA_DIR, row.filename_hr)

        # 3. Load Signal
        try:
            signals, fields = wfdb.rdsamp(record_path)
        except Exception as e:
            continue

        # 4. Extract Lead II
        signal = signals[:, LEAD_IDX]

        # 5. Filter + Segment
        try:
            signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)
            qrs_inds = wfdb.processing.gqrs_detect(signal, fs=SOURCE_RATE)

            for r_peak in qrs_inds:
                window_raw = int(WINDOW_SEC * SOURCE_RATE)
                half_window = window_raw // 2

                start = r_peak - half_window
                end = r_peak + half_window

                if start < 0 or end > len(signal):
                    continue

                beat_raw = signal[start:end]
                beat_resampled = resample(beat_raw, WINDOW_SAMPLES)
                beat_norm = z_score_normalize(beat_resampled)

                processed_beats.append(beat_norm)
                labels.append(lbl)
                patient_ids.append(row.patient_id)
        except Exception:
            continue

    # Print statistics
    print(f"\n=== RHYTHM-HARMONIZED PTB-XL ===")
    print(f"Records included (Normal): {included_normal}")
    print(f"Records included (Abnormal Rhythm): {included_abnormal}")
    print(f"Records excluded (structural-only): {excluded_structural}")
    
    # Save
    df = pd.DataFrame(np.array(processed_beats))
    df['label'] = labels
    df['patient_id'] = patient_ids

    n_normal = sum(1 for l in labels if l == 0)
    n_abnormal = sum(1 for l in labels if l == 1)
    print(f"\nTotal beats: {len(df)}")
    print(f"Normal: {n_normal} ({100*n_normal/len(df):.1f}%)")
    print(f"Abnormal (rhythm): {n_abnormal} ({100*n_abnormal/len(df):.1f}%)")

    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    process_ptbxl_rhythm()
