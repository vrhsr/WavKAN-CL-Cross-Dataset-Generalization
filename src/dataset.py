import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class HarmonizedDataset(Dataset):
    def __init__(self, csv_file, noise_snr_db=None, corruption_type='awgn', corruption_kwargs=None, multi_lead=False, leads=['II'], multi_class=False, class_mapping=None):
        """
        Loads the harmonized ECG data from CSV.
        Expected format: 250 signal columns (0..249), 'label', 'patient_id'
        For multi-lead: assumes columns named like 'I_0', 'I_1', ..., 'II_0', etc.
        For multi-class: applies class_mapping to convert labels.
        
        Args:
            csv_file (str): Path to processed CSV
            noise_snr_db (float, optional): If set, adds additive noise with this SNR to signals.
            corruption_type (str): Corruption mode used for robustness testing.
            corruption_kwargs (dict, optional): Optional parameters for corruption behavior.
            multi_lead (bool): If True, load multiple leads.
            leads (list): List of lead names to load (e.g., ['I', 'II', 'III']).
            multi_class (bool): If True, apply multi-class mapping.
            class_mapping (dict): Mapping from original labels to multi-class labels.
        """
        print(f"Loading dataset from {csv_file}...")
        self.noise_snr_db = noise_snr_db
        self.corruption_type = corruption_type
        self.corruption_kwargs = corruption_kwargs or {}
        self.multi_lead = multi_lead
        self.leads = leads
        self.multi_class = multi_class
        self.class_mapping = class_mapping or {}
        
        df = pd.read_csv(csv_file)
        
        if multi_lead:
            # Assume columns like 'I_0' to 'I_249', 'II_0' to 'II_249', etc.
            self.X = []
            for lead in leads:
                lead_cols = [f"{lead}_{i}" for i in range(250)]
                if not all(col in df.columns for col in lead_cols):
                    raise ValueError(f"Lead {lead} columns not found in CSV")
                lead_data = df[lead_cols].values
                self.X.append(lead_data)
            self.X = np.stack(self.X, axis=1)  # Shape: (n_samples, n_leads, 250)
        else:
            # Single lead
            self.signal_cols = [c for c in df.columns if str(c).isdigit()]
            self.X = df[self.signal_cols].values  # Shape: (n_samples, 250)
        
        self.y = df['label'].values
        
        if multi_class and class_mapping:
            self.y = np.array([class_mapping.get(label, label) for label in self.y])
            print(f"Applied multi-class mapping. Unique classes: {np.unique(self.y)}")
        
        print(f"Loaded {len(df)} samples. Shape: {self.X.shape}")
        if self.noise_snr_db is not None:
            print(f"Configured for Noise Injection: SNR={self.noise_snr_db}dB")

    def _snr_noise_std(self, signal):
        """Return noise std based on configured SNR."""
        if self.noise_snr_db is None:
            return 0.0

        p_signal = np.mean(signal**2)
        if p_signal <= 0:
            return 0.0
        p_noise = p_signal / (10 ** (self.noise_snr_db / 10))
        return np.sqrt(max(p_noise, 1e-12))

    def add_noise(self, signal):
        """
        Adds Gaussian noise to a signal for a given SNR.
        SNR_dB = 10 * log10(P_signal / P_noise)
        """
        if self.noise_snr_db is None:
            return signal
            
        noise_std = self._snr_noise_std(signal)
        if noise_std == 0:
            return signal
        noise = np.random.normal(0, noise_std, signal.shape)
        return signal + noise

    def apply_corruption(self, signal):
        """Apply realistic ECG corruption modes for robustness analysis."""
        corruption = (self.corruption_type or 'awgn').lower()

        if self.noise_snr_db is None:
            return signal

        x = signal.astype(np.float32).copy()
        n = len(x)
        t = np.arange(n, dtype=np.float32)
        fs = float(self.corruption_kwargs.get('fs', 100.0))

        if corruption == 'awgn':
            return self.add_noise(x)

        noise_std = self._snr_noise_std(x)
        if noise_std == 0:
            return x

        if corruption == 'baseline_wander':
            freq_hz = float(self.corruption_kwargs.get('freq_hz', 0.33))
            phase = np.random.uniform(0, 2 * np.pi)
            wander = np.sin(2 * np.pi * freq_hz * t / fs + phase)
            return x + noise_std * wander.astype(np.float32)

        if corruption == 'powerline':
            freq_hz = float(self.corruption_kwargs.get('freq_hz', 50.0))
            phase = np.random.uniform(0, 2 * np.pi)
            hum = np.sin(2 * np.pi * freq_hz * t / fs + phase)
            return x + noise_std * hum.astype(np.float32)

        if corruption == 'muscle':
            # EMG-like high-frequency noise
            emg = np.random.normal(0, noise_std, n).astype(np.float32)
            kernel = np.array([1.0, -2.0, 1.0], dtype=np.float32)  # simple high-pass shape
            emg = np.convolve(emg, kernel, mode='same')
            return x + emg

        if corruption == 'motion':
            # Piecewise amplitude jumps + drift
            y = x.copy()
            n_segments = int(self.corruption_kwargs.get('segments', 3))
            for _ in range(n_segments):
                start = np.random.randint(0, max(1, n - 20))
                end = min(n, start + np.random.randint(10, 40))
                jump = np.random.normal(0, noise_std * 3)
                y[start:end] += jump
            return y + np.random.normal(0, noise_std * 0.3, n).astype(np.float32)

        if corruption == 'lead_dropout':
            y = x.copy()
            drop_len = int(self.corruption_kwargs.get('drop_len', max(5, n // 10)))
            start = np.random.randint(0, max(1, n - drop_len))
            y[start:start + drop_len] = 0.0
            return y

        if corruption == 'sampling_jitter':
            # Slight time-warp to simulate sampling mismatch
            jitter = np.random.normal(0, float(self.corruption_kwargs.get('jitter_std', 0.4)), n)
            idx = np.clip(np.arange(n) + jitter, 0, n - 1)
            return np.interp(np.arange(n), idx, x).astype(np.float32)

        return self.add_noise(x)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return (signal, label)
        signal_raw = self.X[idx]
        
        # Add noise if configured
        if self.noise_snr_db is not None:
            if self.multi_lead:
                # Apply corruption to each lead
                corrupted = []
                for lead_idx in range(signal_raw.shape[0]):
                    corrupted.append(self.apply_corruption(signal_raw[lead_idx]))
                signal_raw = np.stack(corrupted, axis=0)
            else:
                signal_raw = self.apply_corruption(signal_raw)
            # Re-float32 cast in case noise made it float64
            signal_raw = signal_raw.astype(np.float32)
        
        if self.multi_lead:
            signal = torch.tensor(signal_raw)  # Shape: (n_leads, 250)
        else:
            signal = torch.tensor(signal_raw).unsqueeze(0)  # (1, 250)
        
        label_value = int(self.y[idx])
        if self.noise_snr_db is not None and self.corruption_type == 'label_flip':
            flip_prob = float(self.corruption_kwargs.get('flip_prob', 0.1))
            if np.random.rand() < flip_prob:
                label_value = 1 - label_value
        label = torch.tensor(label_value)
        
        return signal, label

class SSLAugmentedDataset(Dataset):
    def __init__(self, csv_file):
        """
        Dataset for Self-Supervised Learning (SimCLR).
        Returns two augmented views of the same signal.
        """
        df = pd.read_csv(csv_file)
        self.signal_cols = [c for c in df.columns if str(c).isdigit()]
        self.X = df[self.signal_cols].values.astype(np.float32)
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
