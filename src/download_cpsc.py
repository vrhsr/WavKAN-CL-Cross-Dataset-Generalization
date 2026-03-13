import os
import re
import requests
from tqdm import tqdm

BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/"
SAVE_DIR = "E:/rpr/data/cpsc2018_raw/"

os.makedirs(SAVE_DIR, exist_ok=True)
session = requests.Session()

for folder in ['g1','g2','g3','g4','g5','g6','g7']:
    folder_url = BASE_URL + folder + "/"
    folder_path = os.path.join(SAVE_DIR, folder)
    os.makedirs(folder_path, exist_ok=True)

    print(f"\nScanning {folder}...")
    r = session.get(folder_url)

    if r.status_code != 200:
        print(f"Failed: {r.status_code}")
        continue

    files = re.findall(r'href="(A\d+\.(?:mat|hea))"', r.text)
    print(f"Found {len(files)} files")

    for fname in tqdm(files, desc=folder, unit="file"):
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            continue
        fr = session.get(folder_url + fname, stream=True)
        if fr.status_code == 200:
            with open(fpath, 'wb') as f:
                for chunk in fr.iter_content(8192):
                    f.write(chunk)

print("\nDone!")