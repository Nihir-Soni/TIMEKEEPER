import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RestorationErrorDataset(Dataset):
    def __init__(self, data_dir, image_size=256, is_train=True, split_ratio=0.8, seed=42):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Discover all samples
        all_samples = sorted([p for p in self.data_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")])
        
        # Split logic (deterministic)
        np.random.seed(seed)
        np.random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * split_ratio)
        if is_train:
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
            
        # If the dataset is too small for split, fallback to using everything to prevent crash during test
        if len(self.samples) == 0 and len(all_samples) > 0:
            self.samples = all_samples
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        
        damaged_path = sample_dir / "damaged.png"
        restored_path = sample_dir / "restored.png"
        error_path = sample_dir / "actual_error.npy"
        
        damaged = cv2.imread(str(damaged_path))
        restored = cv2.imread(str(restored_path))
        error_map = np.load(str(error_path))
        
        # Resize to fixed dimension for batching
        d_resized = cv2.resize(damaged, (self.image_size, self.image_size))
        r_resized = cv2.resize(restored, (self.image_size, self.image_size))
        e_resized = cv2.resize(error_map, (self.image_size, self.image_size))
        
        # Convert BGR to RGB
        d_resized = cv2.cvtColor(d_resized, cv2.COLOR_BGR2RGB)
        r_resized = cv2.cvtColor(r_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize inputs [0, 1]
        d_tensor = torch.from_numpy(d_resized).float().permute(2, 0, 1) / 255.0
        r_tensor = torch.from_numpy(r_resized).float().permute(2, 0, 1) / 255.0
        
        # Normalize error map to [0, 1] based on max possible diff (255)
        # Note: target is E(x,y) / 255.0
        e_tensor = torch.from_numpy(e_resized).float().unsqueeze(0) / 255.0
        
        # Input is concatenation of damaged and restored
        input_tensor = torch.cat([d_tensor, r_tensor], dim=0)
        
        return input_tensor, e_tensor
