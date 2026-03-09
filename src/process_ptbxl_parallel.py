import wfdb
from wfdb import processing
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
import ast
from tqdm import tqdm
import os
import multiprocessing
from functools import partial

# CONFIGURATION
SAMPLE_RATE = 100
WINDOW_SEC = 2.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 500
LEAD_IDX = 1
DATA_DIR = 'data/ptb-xl-1.0.3'
OUTPUT_FILE = 'data/ptbxl_processed.csv'

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
    return -1

# Global for workers to avoid pickling overhead
global_agg_dict = None

def get_agg_dict():
    global global_agg_dict
    if global_agg_dict is None:
        agg_df = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        global_agg_dict = agg_df['diagnostic_class'].to_dict()
    return global_agg_dict

def process_record(args):
    """
    Worker function to process a single record.
    args: (ecg_id, filename_hr, scp_codes)
    """
    index, filename_hr, scp_codes = args
    
    # 1. Map Label
    agg_dict = get_agg_dict()
    
    results = []
    for key in scp_codes.keys():
        if key in agg_dict:
            cls = agg_dict[key]
            if str(cls) != 'nan':
                results.append(cls)
                
    lbl = -1
    if 'NORM' in results:
        lbl = 0
    elif any(c in ['MI', 'STTC', 'CD', 'HYP'] for c in results):
        lbl = 1
    
    if lbl == -1:
        return []

    # 2. Load Signal
    record_path = os.path.join(DATA_DIR, filename_hr)
    try:
        signals, fields = wfdb.rdsamp(record_path)
    except Exception as e:
        # Only print for the first few errors to avoid flooding
        if int(index) < 50: 
            print(f"Error loading {record_path}: {e}")
        return []

    # 3. Process
    signal = signals[:, LEAD_IDX]
    signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)
    
    try:
        qrs_inds = processing.gqrs_detect(signal, fs=SOURCE_RATE)
    except Exception as e:
        if int(index) < 50:
            print(f"Error QRS detect {record_path}: {e}")
        return []

    beats = []
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
        
        beats.append({
            'signal': beat_norm, 
            'label': lbl, 
            'patient_id': index # checking if patient_id is passed or we just use index
        })
        
    return beats

def main():
    print(f"Starting Parallel Processing on {os.cpu_count()} cores...")
    
    db_csv_path = os.path.join(DATA_DIR, 'ptbxl_database.csv')
    if not os.path.exists(db_csv_path):
        print("Database not found. Please download ptb-xl first.")
        return

    # Load Metadata
    Y = pd.read_csv(db_csv_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    agg_df = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    agg_dict = agg_df['diagnostic_class'].to_dict()
    
    # Prepare arguments
    tasks = []
    for ecg_id, row in Y.iterrows():
        tasks.append((row.patient_id, row.filename_hr, row.scp_codes))
        
    print(f"Distributing {len(tasks)} tasks...")
    
    processed_beats = []
    labels = []
    patient_ids = []
    
    num_processes = min(8, os.cpu_count() or 1)
    print(f"Using {num_processes} processes to avoid memory overload...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Chunksize helps with overhead
        results = list(tqdm(pool.imap(process_record, tasks, chunksize=50), total=len(tasks)))
        
    # Flatten results
    print("Aggregating results...")
    for res in results:
        for beat in res:
            processed_beats.append(beat['signal'])
            labels.append(beat['label'])
            patient_ids.append(beat['patient_id'])
            
    # Save
    df = pd.DataFrame(np.array(processed_beats))
    df['label'] = labels
    df['patient_id'] = patient_ids
    
    print(f"Saving {len(df)} beats to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
