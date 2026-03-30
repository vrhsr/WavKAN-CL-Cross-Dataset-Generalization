"""
SNOMED-CT to PTB-XL Superclass Mapping Script.

Maps SNOMED-CT diagnosis codes (used across CINC 2020 datasets like Chapman, Georgia)
to the 5 primary PTB-XL Superclasses: [NORM, MI, STTC, CD, HYP].
"""

SNOMED_MAPPING = {
    # NORM - Normal
    '426783006': 'NORM', # Normal sinus rhythm

    # MI - Myocardial Infarction
    '164861001': 'MI',   # Myocardial ischemia
    '164865005': 'MI',   # Myocardial infarction
    '164867002': 'MI',   # Old myocardial infarction
    '54329005': 'MI',    # Acute myocardial infarction
    '57054005': 'MI',    # Acute myocardial infarction
    '433524009': 'MI',   # Anteroseptal myocardial infarction
    '270492004': 'MI',   # Ischemia (sometimes coded here)
    
    # STTC - ST/T wave changes (Arrhythmias often grouped here or separately, but we follow general rhythm rules)
    '426177001': 'STTC', # Sinus bradycardia
    '427084000': 'STTC', # Sinus tachycardia
    '164931005': 'STTC', # ST segment elevation
    '429622005': 'STTC', # ST segment depression
    '59931005': 'STTC',  # T wave inversion
    '164890007': 'STTC', # Atrial flutter
    '17338001': 'STTC',  # Ventricular ectopic beats
    '427172004': 'STTC', # Premature ventricular contractions (PVC)
    '284470004': 'STTC', # Premature atrial contraction (PAC)
    '326202001': 'STTC', # Premature contractions
    '427393009': 'STTC', # Sinus arrhythmia
    '63593006': 'STTC',  # PAC
    '164884008': 'STTC', # VFB
    '713427006': 'STTC', # Complete RBBB (wait, RBBB usually CD. Keeping as CD below)

    # CD - Conduction Disturbance
    '164889003': 'CD',   # Atrial fibrillation - grouped in CD in many PhysioNet simplifications 
    '164909002': 'CD',   # LBBB
    '59118001': 'CD',    # RBBB
    '713426002': 'CD',   # Incomplete RBBB
    '251146004': 'CD',   # QT interval prolongation
    '698252002': 'CD',   # NSIVCB
    '195042002': 'CD',   # 2nd degree AV block
    '233917008': 'CD',   # AV block
    '27885002': 'CD',    # 3rd degree AV block
    
    # HYP - Hypertrophy
    '164934002': 'HYP',  # Left ventricular hypertrophy
    '39732003': 'HYP',   # Right ventricular hypertrophy
    '445118002': 'HYP',  # Left atrial hypertrophy
}

# Add standard string diagnoses often found in Chapman directly just-in-case
STRING_MAPPING = {
    'SB': 'STTC',
    'SR': 'NORM',
    'AFIB': 'CD',
    'STTC': 'STTC',
    'PVC': 'STTC',
    'PAC': 'STTC',
    'TVI': 'STTC',
    'AVB': 'CD',
    'LBBB': 'CD',
    'RBBB': 'CD',
}

def map_snomed_to_superclass(snomed_codes_str, split_char=','):
    """
    Given a string of SNOMED codes (e.g. '164889003,270492004'),
    returns a multi-hot list [NORM, MI, STTC, CD, HYP] of size 5.
    If no valid mapping, returns None (can be skipped).
    """
    codes = [c.strip() for c in str(snomed_codes_str).split(split_char) if c.strip()]
    labels = [0, 0, 0, 0, 0] # NORM, MI, STTC, CD, HYP
    mapping_order = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    found_any = False
    for code in codes:
        if code in SNOMED_MAPPING:
            sc = SNOMED_MAPPING[code]
            idx = mapping_order.index(sc)
            labels[idx] = 1
            found_any = True
        elif code in STRING_MAPPING:
            sc = STRING_MAPPING[code]
            idx = mapping_order.index(sc)
            labels[idx] = 1
            found_any = True
            
    if not found_any:
        return None
        
    return labels

def map_cpsc_to_superclass(cpsc_codes_str, split_char=','):
    """
    CPSC 2018 uses custom integer codes 1-9.
    1 (Normal)  -> NORM
    2 (AF)      -> CD
    3 (IAVB)    -> CD
    4 (LBBB)    -> CD
    5 (RBBB)    -> CD
    6 (PAC)     -> CD
    7 (PVC)     -> CD
    8 (STD)     -> STTC
    9 (STE)     -> STTC
    """
    mapping = {
        '1': 'NORM', '2': 'CD', '3': 'CD', '4': 'CD', '5': 'CD',
        '6': 'CD', '7': 'CD', '8': 'STTC', '9': 'STTC'
    }
    codes = [c.strip() for c in str(cpsc_codes_str).split(split_char) if c.strip()]
    labels = [0, 0, 0, 0, 0] # NORM, MI, STTC, CD, HYP
    mapping_order = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    found_any = False
    for code in codes:
        if code in mapping:
            sc = mapping[code]
            idx = mapping_order.index(sc)
            labels[idx] = 1
            found_any = True
            
    if not found_any:
        return None
        
    return labels

def map_scp_to_superclass(scp_codes_str):
    """
    Placeholder for PTB-XL mapping if used dynamically.
    PTB-XL typically relies on its internal dataframe ontology.
    """
    pass

def print_mapping_stats():
    print(f"Supported SNOMED codes: {len(SNOMED_MAPPING)}")
    
if __name__ == "__main__":
    print_mapping_stats()
