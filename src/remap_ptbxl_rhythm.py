"""
Fast PTB-XL Rhythm Remapping.

Instead of re-downloading and re-processing the full 1.8GB PTB-XL dataset,
this script:
1. Reads the PTB-XL metadata (ptbxl_database.csv + scp_statements.csv)
2. Builds a patient_id → rhythm_label mapping
3. Reads the existing ptbxl_processed.csv
4. Remaps labels and filters to rhythm-only records
5. Saves as ptbxl_rhythm_processed.csv

This avoids hours of signal processing while achieving the same result.
"""
import pandas as pd
import ast
import os
import numpy as np

# Paths
METADATA_CSV = 'data/ptbxl_database.csv'
SCP_CSV = 'data/scp_statements.csv'
INPUT_CSV = 'data/ptbxl_processed.csv'
OUTPUT_CSV = 'data/ptbxl_rhythm_processed.csv'

# Rhythm-only SCP codes (same as emit_ptbxl_rhythm.py)
NORMAL_CODES = {'NORM'}
ABNORMAL_RHYTHM_CODES = {
    'AFIB', 'AFLT', 'SVTAC', 'PSVT', 'PAC', 'SARRH', 'SBRAD', 'STACH',
    'PVC', 'BIGU', 'TRIGU',
    'LBBB', 'RBBB', 'CLBBB', 'CRBBB', 'LAFB', 'LPFB',
    '1AVB', '2AVB', '3AVB', 'WPW',
}


def map_rhythm_label(scp_codes):
    """Map SCP codes to binary using rhythm-only criteria."""
    code_keys = set(scp_codes.keys())
    if code_keys & ABNORMAL_RHYTHM_CODES:
        return 1
    if code_keys & NORMAL_CODES:
        return 0
    return -1  # Exclude (structural-only)


def main():
    print("=" * 60)
    print("  PTB-XL RHYTHM LABEL REMAPPING (Fast Mode)")
    print("=" * 60)

    # Step 1: Load metadata
    print("\nLoading PTB-XL metadata...")
    meta = pd.read_csv(METADATA_CSV, index_col='ecg_id')
    meta['scp_codes'] = meta['scp_codes'].apply(ast.literal_eval)
    print(f"  Total records in metadata: {len(meta)}")

    # Step 2: Build patient_id → rhythm_label mapping
    # A patient can have multiple ECG records with different diagnoses.
    # We need per-record mapping, but our processed CSV only has patient_id.
    # Strategy: for each patient_id, collect ALL their rhythm labels.
    # If ANY record has a rhythm abnormality → patient is abnormal.
    # If ALL records are NORM only → patient is normal.
    # If NO records match rhythm criteria → exclude patient entirely.
    
    patient_labels = {}
    excluded_records = 0
    
    for ecg_id, row in meta.iterrows():
        pid = row['patient_id']
        lbl = map_rhythm_label(row['scp_codes'])
        
        if lbl == -1:
            excluded_records += 1
            continue
        
        if pid not in patient_labels:
            patient_labels[pid] = lbl
        else:
            # If any record is abnormal, patient is abnormal
            if lbl == 1:
                patient_labels[pid] = 1

    n_normal = sum(1 for v in patient_labels.values() if v == 0)
    n_abnormal = sum(1 for v in patient_labels.values() if v == 1)
    print(f"  Rhythm-mapped patients: {len(patient_labels)}")
    print(f"    Normal: {n_normal}")
    print(f"    Abnormal (rhythm): {n_abnormal}")
    print(f"  Excluded records (structural-only): {excluded_records}")

    # Step 3: Load existing processed CSV
    print(f"\nLoading existing processed data from {INPUT_CSV}...")
    # Use chunked reading for memory efficiency
    chunks = []
    kept = 0
    dropped = 0
    
    for chunk in pd.read_csv(INPUT_CSV, chunksize=50000):
        # Filter: keep only rows whose patient_id is in our rhythm mapping
        mask = chunk['patient_id'].isin(patient_labels)
        filtered = chunk[mask].copy()
        
        # Remap labels
        filtered['label'] = filtered['patient_id'].map(patient_labels)
        
        chunks.append(filtered)
        kept += len(filtered)
        dropped += (~mask).sum()
    
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"  Kept: {kept} beats")
    print(f"  Dropped: {dropped} beats (structural-only patients)")

    # Step 4: Print final statistics
    n_normal_beats = (df['label'] == 0).sum()
    n_abnormal_beats = (df['label'] == 1).sum()
    print(f"\n=== RHYTHM-HARMONIZED PTB-XL ===")
    print(f"Total beats: {len(df)}")
    print(f"Normal: {n_normal_beats} ({100*n_normal_beats/len(df):.1f}%)")
    print(f"Abnormal (rhythm): {n_abnormal_beats} ({100*n_abnormal_beats/len(df):.1f}%)")

    # Step 5: Save
    print(f"\nSaving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
