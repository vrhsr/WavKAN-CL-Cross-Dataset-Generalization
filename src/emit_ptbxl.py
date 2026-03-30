import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
import ast
from tqdm import tqdm
import os
import zipfile
import requests
import argparse

# CONFIGURATION
SAMPLE_RATE = 100      # Target Hz
WINDOW_SEC = 10.0      # Window length (10 seconds for full record)
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC) # 1000 samples
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 500      # PTB-XL High Res
DATA_DIR = 'data/ptb-xl-1.0.3'
OUTPUT_DIR = 'data'
NUM_LEADS = 12

# 5 Superclasses in order: NORM, MI, STTC, CD, HYP
SUPERCLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def map_scp_to_superclass(scp_codes, agg_df):
    """Map SCP codes to a multi-hot vector of the 5 superclasses."""
    multi_hot = np.zeros(5, dtype=np.float32)
    for key in scp_codes.keys():
        if key in agg_df.index:
            cls = agg_df.loc[key].diagnostic_class
            if str(cls) != 'nan' and cls in SUPERCLASS_NAMES:
                idx = SUPERCLASS_NAMES.index(cls)
                multi_hot[idx] = 1.0
    return multi_hot

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
    
    os.makedirs(extract_to, exist_ok=True)
    
    print(f"Downloading Full PTB-XL from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(zip_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    print("Extracting...")
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

def process_ptbxl(single_lead=False):
    db_csv_path = os.path.join(DATA_DIR, 'ptbxl_database.csv')
    if not os.path.exists(db_csv_path):
        download_ptbxl_zip()
    
    print(f"Loading PTB-XL Metadata from {DATA_DIR}...")
    Y = pd.read_csv(db_csv_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    agg_df = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    all_signals = []
    all_labels = []
    
    metadata_records = []
    count = 0
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    target_leads = 1 if single_lead else NUM_LEADS
    
    for ecg_id, row in tqdm(Y.iterrows(), total=len(Y), desc=f"Processing PTB-XL ({target_leads}-lead)"):
        count += 1
        
        # 1. Multi-hot label
        label_vector = map_scp_to_superclass(row.scp_codes, agg_df)
        if label_vector.sum() == 0:
            continue # Skip if no matching superclass
            
        record_path = os.path.join(DATA_DIR, row.filename_hr)
        
        try:
            signals, fields = wfdb.rdsamp(record_path)
        except Exception as e:
            if count < 10: print(f"Error loading {record_path}: {e}")
            continue
            
        # 4. Extract Leads and resample
        resampled_leads = np.zeros((target_leads, WINDOW_SAMPLES), dtype=np.float32)
        
        try:
            for lead_idx in range(target_leads):
                source_idx = 1 if single_lead else lead_idx # Lead II is index 1 for single lead
                signal = signals[:, source_idx]
                signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)
                
                # Resample cleanly using original length -> target length
                beat_resampled = resample(signal, WINDOW_SAMPLES)
                resampled_leads[lead_idx, :] = z_score_normalize(beat_resampled)
                
            all_signals.append(resampled_leads)
            all_labels.append(label_vector)
            
            # Save metadata
            metadata_records.append({
                'patient_id': row.patient_id,
                'age': row.age,
                'sex': row.sex,
                'strat_fold': row.strat_fold,
                'diagnostic_class': str(label_vector.tolist())
            })
            
        except Exception as e:
            continue

    signals_arr = np.stack(all_signals)
    labels_arr = np.stack(all_labels)
    metadata_df = pd.DataFrame(metadata_records)

    print(f"Output shapes: Signals {signals_arr.shape}, Labels {labels_arr.shape}")
    
    np.save(os.path.join(OUTPUT_DIR, 'ptbxl_signals.npy'), signals_arr)
    np.save(os.path.join(OUTPUT_DIR, 'ptbxl_labels.npy'), labels_arr)
    metadata_df.to_csv(os.path.join(OUTPUT_DIR, 'ptbxl_metadata.csv'), index=False)
    
    print(f"PTB-XL {target_leads}-lead processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-lead', action='store_true', help='Extract only Lead II')
    args = parser.parse_args()
    process_ptbxl(args.single_lead)
