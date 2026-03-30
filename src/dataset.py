import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class ECGTrainTransform:
    def __call__(self, x):
        # x shape: (channels, seq_len)
        if random.random() < 0.5:
            x = x + torch.randn_like(x) * 0.05  # Gaussian noise
        if random.random() < 0.3:
            scale = 0.8 + random.random() * 0.4  # [0.8, 1.2]
            x = x * scale                        # Amplitude scaling
        if random.random() < 0.2:
            lead_idx = random.randint(0, x.shape[0] - 1)
            x[lead_idx] = 0                      # Lead dropout
        return x

class HarmonizedDataset(Dataset):
    def __init__(self, data_path, label_path=None, noise_snr_db=None, corruption_type='awgn', corruption_kwargs=None, multi_lead=False, leads=['II'], multi_class=False, class_mapping=None, transform=None):
        """
        Loads the harmonized ECG data from CSV or NPY.
        """
        print(f"Loading dataset from {data_path}...")
        self.noise_snr_db = noise_snr_db
        self.corruption_type = corruption_type
        self.corruption_kwargs = corruption_kwargs or {}
<<<<<<< HEAD
        self.multi_lead = multi_lead
        self.leads = leads
        self.multi_class = multi_class
        self.class_mapping = class_mapping or {}
        self.transform = transform
        
        if data_path.endswith('.npy'):
            self.X = np.load(data_path)
            self.y = np.load(label_path) if label_path else np.zeros(len(self.X))
            self.multi_lead = self.X.ndim == 3 # (N, leads, len)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            if multi_lead:
                self.X = []
                for lead in leads:
                    lead_cols = [f"{lead}_{i}" for i in range(250)]
                    if not all(col in df.columns for col in lead_cols):
                        raise ValueError(f"Lead {lead} columns not found in CSV")
                    lead_data = df[lead_cols].values
                    self.X.append(lead_data)
                self.X = np.stack(self.X, axis=1)
            else:
                self.signal_cols = [c for c in df.columns if str(c).isdigit()]
                self.X = df[self.signal_cols].values
            self.y = df['label'].values
            
            if multi_class and class_mapping:
                self.y = np.array([class_mapping.get(label, label) for label in self.y])
        else:
            raise ValueError(f"Unsupported format: {data_path}")
            
        print(f"Loaded {len(self.X)} samples. Shape: {self.X.shape}")
=======
        # Optimize memory by specifying float32 immediately
        # Read just columns first to build dtype dict without loading data
        cols = pd.read_csv(csv_file, nrows=0).columns
        self.signal_cols = [c for c in cols if str(c).isdigit()]
        dtype_dict = {c: np.float32 for c in self.signal_cols}
        dtype_dict['label'] = np.int16
        if 'patient_id' in cols:
            dtype_dict['patient_id'] = np.int32
            
        df = pd.read_csv(csv_file, dtype=dtype_dict)
        
        self.y = df['label'].values
        
        if multi_class and class_mapping:
            self.y = np.array([class_mapping.get(label, label) for label in self.y])
            print(f"Applied multi-class mapping. Unique classes: {np.unique(self.y)}")
        
        print(f"Loaded {len(df)} samples. Shape: {self.X.shape}")
>>>>>>> 31960eb21812e61d0c2c98429f36f02f1ec30048
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
        signal_raw = self.X[idx]
        
        if self.noise_snr_db is not None:
<<<<<<< HEAD
            if signal_raw.ndim == 2:
                corrupted = [self.apply_corruption(signal_raw[i]) for i in range(signal_raw.shape[0])]
                signal_raw = np.stack(corrupted, axis=0)
            else:
                signal_raw = self.apply_corruption(signal_raw)
=======
            signal_raw = self.apply_corruption(signal_raw)
            # Re-float32 cast in case noise made it float64
>>>>>>> 31960eb21812e61d0c2c98429f36f02f1ec30048
            signal_raw = signal_raw.astype(np.float32)
        
        if signal_raw.ndim == 2:
            signal = torch.tensor(signal_raw, dtype=torch.float32)
        else:
            signal = torch.tensor(signal_raw, dtype=torch.float32).unsqueeze(0)
            
        # 1. Per-sample, per-lead normalization
        signal = (signal - signal.mean(dim=-1, keepdim=True)) / (signal.std(dim=-1, keepdim=True) + 1e-8)
        
        # 2. Supervised transforms
        if self.transform is not None:
            signal = self.transform(signal)
            
        if isinstance(self.y[idx], np.ndarray):
            label = torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            label_value = int(self.y[idx])
            if self.noise_snr_db is not None and self.corruption_type == 'label_flip':
                flip_prob = float(self.corruption_kwargs.get('flip_prob', 0.1))
                if np.random.rand() < flip_prob:
                    label_value = 1 - label_value
            label = torch.tensor(label_value, dtype=torch.long)
            
        return signal, label

class SSLAugmentedDataset(Dataset):
    def __init__(self, data_file):
        """
        Dataset for Self-Supervised Learning (SimCLR).
        Returns two augmented views of the same 12-lead signal.
        """
        if data_file.endswith('.npy'):
            self.X = np.load(data_file).astype(np.float32)
        else:
            df = pd.read_csv(data_file)
            signal_cols = [c for c in df.columns if str(c).isdigit()]
            self.X = df[signal_cols].values.astype(np.float32)
            if self.X.ndim == 2:
                self.X = np.expand_dims(self.X, axis=1)
        print(f"SSL Dataset Loaded: {len(self.X)} samples.")

    def temporal_crop(self, x, crop_ratio=0.8):
        T = x.shape[1]
        crop_len = int(T * crop_ratio)
        start = np.random.randint(0, max(1, T - crop_len))
        cropped = x[:, start:start + crop_len]
        padded = np.pad(cropped, ((0, 0), (0, T - crop_len)), mode='constant')
        return padded

    def augment(self, signal):
        """
        Apply 12-lead aware random augmentations.
        Input shape: (C, L)
        """
        sig = signal.copy()
        
        # 1. Amplitude Scale (0.5 to 1.5)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.5, 1.5, size=(sig.shape[0], 1))
            sig = sig * scale
            
        # 2. Add Noise (SNR ~ 10-20dB)
        if np.random.rand() > 0.5:
            noise_amp = np.random.uniform(0.01, 0.05) * np.max(np.abs(sig))
            sig = sig + np.random.normal(0, noise_amp, sig.shape)
            
        # 3. Lead Dropout (zero out random leads, p=0.15)
        if np.random.rand() > 0.5:
            p = 0.15
            mask = np.random.rand(sig.shape[0]) > p
            sig = sig * mask[:, np.newaxis]
            
        # 4. Temporal Crop (80% crop + pad)
        if np.random.rand() > 0.5:
            sig = self.temporal_crop(sig, crop_ratio=0.8)
            
        return torch.tensor(sig).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal_raw = self.X[idx]
        if signal_raw.ndim == 1:
            signal_raw = signal_raw[np.newaxis, :]
            
        view1 = self.augment(signal_raw)
        view2 = self.augment(signal_raw)
        
        return view1, view2
