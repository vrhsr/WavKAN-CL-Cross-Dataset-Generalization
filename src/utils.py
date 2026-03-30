import torch
import numpy as np
import random
import os
from datetime import datetime

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def get_device() -> torch.device:
    """Get best available device and print initial memory stats."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def log_gpu_memory(tag: str = "") -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[GPU {tag}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Total: {total:.2f}GB")

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, metrics: dict, config: dict, path: str) -> None:
    """Standardized checkpoint saving."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at epoch {epoch}")

def compute_pos_weight(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """labels: (N, 5) multi-hot tensor"""
    pos_count = labels.sum(dim=0)          # count of positives per class
    neg_count = len(labels) - pos_count    # count of negatives per class
    pos_weight = neg_count / (pos_count + 1e-8)  # higher weight for rare classes
    return pos_weight.to(device)
