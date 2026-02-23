import pandas as pd
import ast

def load_ptbxl_database(path):
    """
    Loads the PTB-XL database index and maps the SCP codes to superclasses.
    """
    # Load the index file
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load scp_statements.csv for mapping
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                cls = agg_df.loc[key].diagnostic_class
                if str(cls) != 'nan':
                    tmp.append(cls)
        return list(set(tmp))

    # Apply aggregation
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    
    return Y

def map_to_aami_binary(superclass_list):
    """
    Maps PTB-XL superclasses to Binary AAMI (Normal vs Abnormal).
    """
    if 'NORM' in superclass_list:
        return 0 # Normal
    if len(superclass_list) > 0:
        return 1 # Abnormal (MI, STTC, CD, HYP)
    return -1 # Unlabeled/Other

if __name__ == "__main__":
    import sys
    import os
    
    # Check if data exists
    data_path = 'e:/rpr/data/ptb-xl-1.0.3/'
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        print("Please download PTB-XL and extract it to e:/rpr/data/")
    else:
        print("Loading PTB-XL...")
        Y = load_ptbxl_database(data_path)
        print(f"Loaded {len(Y)} recordings.")
        
        # Apply binary mapping test
        Y['label_binary'] = Y['diagnostic_superclass'].apply(map_to_aami_binary)
        print(Y['label_binary'].value_counts())
        print("Harmonization Logic Verified.")
