import os
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
from tqdm import tqdm
import requests
import tarfile
from sklearn.model_selection import GroupShuffleSplit
import wfdb
import glob
from src.label_mapping import map_snomed_to_superclass

# CONFIGURATION
SAMPLE_RATE = 100
WINDOW_SEC = 10.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
NUM_LEADS = 12

DATA_DIR = "data/chapman-shaoxing"
OUTPUT_DIR = "data"

def download_chapman():
    url = "https://storage.googleapis.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_Zheng.tar.gz"
    tar_path = "data/chapman.tar.gz"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(tar_path):
        print(f"Downloading Chapman-Shaoxing Dataset from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(tar_path, 'wb') as file, tqdm(total=total_size, unit='iB', unit_scale=True) as bar:
             for data in response.iter_content(block_size):
                 bar.update(len(data))
                 file.write(data)
                 
    print("Extracting Chapman tar.gz...")
    with tarfile.open(tar_path, "r:gz") as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        for member in tar.getmembers():
            member_path = os.path.join(DATA_DIR, member.name)
            if not is_within_directory(DATA_DIR, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
        tar.extractall(path=DATA_DIR)

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, data)

def z_score_normalize(signal):
    std = np.std(signal)
    if std < 1e-6: return np.zeros_like(signal)
    return (signal - np.mean(signal)) / std

def get_dx_from_record(record):
    for comment in record.comments:
        if comment.startswith('Dx:'):
            return comment.split('Dx:')[1].strip()
    return ""

def main():
    if not os.path.exists(DATA_DIR) or len(glob.glob(f"{DATA_DIR}/**/*.hea", recursive=True)) == 0:
        download_chapman()
        
    hea_files = glob.glob(f"{DATA_DIR}/**/*.hea", recursive=True)
    record_paths = [f[:-4] for f in hea_files]
    
    print(f"Found {len(record_paths)} records. Processing...")
    
    all_signals = []
    all_labels = []
    metadata = []
    
    for path in tqdm(record_paths):
        try:
            record = wfdb.rdrecord(path)
            label_vec = map_snomed_to_superclass(dx_str)
            
            if label_vec is None or sum(label_vec) == 0:
                continue
                
            signals = record.p_signal
            fs = record.fs
            
            resampled_leads = np.zeros((NUM_LEADS, WINDOW_SAMPLES), dtype=np.float32)
            
            for lead_idx in range(min(NUM_LEADS, signals.shape[1])):
                sig = signals[:, lead_idx]
                sig = sig[~np.isnan(sig)]
                if len(sig) == 0: continue
                
                sig = bandpass_filter(sig, FILTER_LOW, FILTER_HIGH, fs)
                
                # Resample cleanly using original length -> target length
                beat_resampled = resample(sig, WINDOW_SAMPLES)
                resampled_leads[lead_idx, :] = z_score_normalize(beat_resampled)
                
            all_signals.append(resampled_leads)
            all_labels.append(label_vec)
            
            patient_id = os.path.basename(path)
            metadata.append({
                'patient_id': patient_id,
                'diagnostic_class': str(label_vec)
            })
            
        except Exception as e:
            continue
            
    if len(all_signals) == 0:
        print("Error: No signals extracted.")
        return
        
    signals_arr = np.stack(all_signals)
    labels_arr = np.stack(all_labels)
    metadata_df = pd.DataFrame(metadata)
    
    # Assign stratified folds at patient level (split 80/10/10)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(gss.split(metadata_df, groups=metadata_df['patient_id']))
    
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(metadata_df.iloc[temp_idx], groups=metadata_df.iloc[temp_idx]['patient_id']))
    
    metadata_df['strat_fold'] = 1 # default train
    metadata_df.loc[metadata_df.index[temp_idx[val_idx]], 'strat_fold'] = 9 # val
    metadata_df.loc[metadata_df.index[temp_idx[test_idx]], 'strat_fold'] = 10 # test
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'chapman_signals.npy'), signals_arr)
    np.save(os.path.join(OUTPUT_DIR, 'chapman_labels.npy'), labels_arr)
    metadata_df.to_csv(os.path.join(OUTPUT_DIR, 'chapman_metadata.csv'), index=False)
    
    print(f"Chapman 12-lead processing complete! Shape: {signals_arr.shape}")

if __name__ == "__main__":
    main()
