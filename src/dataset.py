import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class HarmonizedDataset(Dataset):
    def __init__(self, csv_file, noise_snr_db=None):
        """
        Loads the harmonized ECG data from CSV.
        Expected format: 250 signal columns (0..249), 'label', 'patient_id'
        
        Args:
            csv_file (str): Path to processed CSV
            noise_snr_db (float, optional): If set, adds Gaussian noise with this SNR to signals.
        """
        print(f"Loading dataset from {csv_file}...")
        self.noise_snr_db = noise_snr_db
        self.df = pd.read_csv(csv_file)
        
        # Identify signal columns (0..249)
        # We assume columns that are numeric strings are the signal
        self.signal_cols = [c for c in self.df.columns if str(c).isdigit()]
        
        self.X = self.df[self.signal_cols].values.astype(np.float32)
        self.y = self.df['label'].values.astype(np.int64)
        
        print(f"Loaded {len(self.df)} samples. Shape: {self.X.shape}")
        if self.noise_snr_db is not None:
            print(f"Configured for Noise Injection: SNR={self.noise_snr_db}dB")

    def add_noise(self, signal):
        """
        Adds Gaussian noise to a signal for a given SNR.
        SNR_dB = 10 * log10(P_signal / P_noise)
        """
        if self.noise_snr_db is None:
            return signal
            
        # Calculate signal power
        P_signal = np.mean(signal**2)
        if P_signal == 0:
            return signal
            
        # P_noise = P_signal / 10^(SNR/10)
        P_noise = P_signal / (10 ** (self.noise_snr_db / 10))
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(P_noise), signal.shape)
        return signal + noise

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return (signal, label)
        # Signal shape: (1, 250) for Conv1D compatibility
        signal_raw = self.X[idx]
        
        # Add noise if configured
        if self.noise_snr_db is not None:
            signal_raw = self.add_noise(signal_raw)
            # Re-float32 cast in case noise made it float64
            signal_raw = signal_raw.astype(np.float32)
            
        signal = torch.tensor(signal_raw).unsqueeze(0) # (1, Length)
        
        label = torch.tensor(self.y[idx])
        
        return signal, label

class SSLAugmentedDataset(Dataset):
    def __init__(self, csv_file):
        """
        Dataset for Self-Supervised Learning (SimCLR).
        Returns two augmented views of the same signal.
        """
        self.df = pd.read_csv(csv_file)
        self.signal_cols = [c for c in self.df.columns if str(c).isdigit()]
        self.X = self.df[self.signal_cols].values.astype(np.float32)
        print(f"SSL Dataset Loaded: {len(self.X)} samples.")

    def augment(self, signal):
        """
        Apply random augmentations:
        1. Gaussian Noise
        2. Amplitude Scaling
        3. Random Masking
        """
        sig = signal.copy()
        
        # 1. Amplitude Scale (0.5 to 1.5)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.5, 1.5)
            sig = sig * scale
            
        # 2. Add Noise (SNR ~ 10-20dB)
        if np.random.rand() > 0.5:
            noise_amp = np.random.uniform(0.01, 0.05) * np.max(np.abs(sig))
            noise = np.random.normal(0, noise_amp, sig.shape)
            sig = sig + noise
            
        # 3. Random Masking (Zero out 10% of signal)
        if np.random.rand() > 0.5:
            mask_len = int(len(sig) * 0.1)
            start = np.random.randint(0, len(sig) - mask_len)
            sig[start:start+mask_len] = 0.0
            
        return torch.tensor(sig).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal_raw = self.X[idx]
        
        # Generate two views
        view1 = self.augment(signal_raw).unsqueeze(0)
        view2 = self.augment(signal_raw).unsqueeze(0)
        
        return view1, view2
