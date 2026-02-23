import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
import ast
from tqdm import tqdm
import os

# CONFIGURATION (LOCKED - MUST MATCH MIT-BIH)
SAMPLE_RATE = 100      # Target Hz
WINDOW_SEC = 2.5       # Window length
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC) # 250 samples
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 500      # PTB-XL High Res
LEAD_IDX = 1           # Lead II in standard 12-lead order (I, II, III...) is Index 1
DATA_DIR = 'data/ptb-xl-1.0.3'
OUTPUT_FILE = 'data/ptbxl_processed.csv'

# LABEL MAPPING (BINARY)
# NORM -> 0
# MI, STTC, CD, HYP -> 1
def map_label(scp_codes, agg_df):
    results = []
    for key in scp_codes.keys():
        if key in agg_df.index:
            cls = agg_df.loc[key].diagnostic_class
            if str(cls) != 'nan':
                results.append(cls)
    
    if 'NORM' in results:
        return 0
    if any(c in ['MI', 'STTC', 'CD', 'HYP'] for c in results):
        return 1
    return -1 # Exclude

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
    
    # Check if we already have the unzipped folder
    # Note: PhysioNet zips often extract to a long name folder
    # We will rename it to 'ptb-xl-1.0.3' if needed, or just map DATA_DIR
    
    print(f"Downloading Full PTB-XL from {url}...")
    import requests
    
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
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
        
    # Handle folder renaming
    # The zip usually makes "ptb-xl-a-large...-1.0.3"
    # We want "ptb-xl-1.0.3"
    extracted_dirs = [d for d in os.listdir(extract_to) if 'ptb-xl' in d and os.path.isdir(os.path.join(extract_to, d))]
    for d in extracted_dirs:
        if d != 'ptb-xl-1.0.3':
            # Identify the long name
            if 'large-publicly' in d:
                src = os.path.join(extract_to, d)
                dst = os.path.join(extract_to, 'ptb-xl-1.0.3')
                if os.path.exists(dst):
                    print(f"Target dir {dst} already exists. Merging/Skipping rename.")
                else:
                    os.rename(src, dst)
                break
    
    # Cleanup
    os.remove(zip_path)

def process_ptbxl():
    # 1. Download if missing
    db_csv_path = os.path.join(DATA_DIR, 'ptbxl_database.csv')
    if not os.path.exists(db_csv_path):
        download_ptbxl_zip()
    
    print(f"Loading PTB-XL Metadata from {DATA_DIR}...")
    Y = pd.read_csv(db_csv_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    agg_df = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    processed_beats = []
    labels = []
    patient_ids = []
    
    count = 0
    # Iterate ALL records
    for ecg_id, row in tqdm(Y.iterrows(), total=len(Y)):
        count += 1
        
        # 1. Get Label first
        lbl = map_label(row.scp_codes, agg_df)
        if lbl == -1:
            continue
            
        # 2. Path
        record_path = os.path.join(DATA_DIR, row.filename_hr)
        
        # 3. Load Signal
        try:
            signals, fields = wfdb.rdsamp(record_path)
        except Exception as e:
            # If file missing (maybe zip failed?), skip
            if count < 10: print(f"Error loading {record_path}: {e}")
            continue
            
        # 4. Extract Lead II
        signal = signals[:, LEAD_IDX] # Index 1
        
        # 5. Filter
        try:
            signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)
            
            # 6. Peak Detection
            qrs_inds = wfdb.processing.gqrs_detect(signal, fs=SOURCE_RATE)
            
            # 7. Segment
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
        except Exception as e:
            # Skip record if processing fails
            continue
    
    # Save
    df = pd.DataFrame(np.array(processed_beats))
    df['label'] = labels
    df['patient_id'] = patient_ids
    
    print(f"Saving {len(df)} beats to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    process_ptbxl()
