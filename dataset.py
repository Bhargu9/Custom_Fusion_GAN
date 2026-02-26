# dataset.py
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import glob
import random

DATA_MIN = 0.0
DATA_MAX = 255.0

class COGPatchedDataset(Dataset):
    """
    This dataset now generates multi-scale LR patches (10m, 20m, 30m)
    by downsampling the HR patch on the fly.
    """
    def __init__(self, cog_dir, patch_size=256):
        super().__init__()
        self.patch_size = patch_size
        path_pattern = os.path.join(cog_dir, '**', '*.tif*') 
        all_files = sorted(glob.glob(path_pattern, recursive=True))
        self.hr_files = [
            f for f in all_files 
            if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
        ]

        print(f"Found {len(all_files)} files, filtered to {len(self.hr_files)} valid COG images in {cog_dir}.")
        
        if not self.hr_files:
            raise FileNotFoundError(f"No valid .tif or .tiff files found in {cog_dir}.")

    def __getitem__(self, index):
        hr_file_path = self.hr_files[index]
        with rasterio.open(hr_file_path) as src:
            height, width = src.height, src.width
            rand_x = random.randint(0, width - self.patch_size)
            rand_y = random.randint(0, height - self.patch_size)
            window = Window(rand_x, rand_y, self.patch_size, self.patch_size)
            hr_patch = src.read([1, 2, 3], window=window).astype(np.float32)

        hr_patch = (hr_patch - DATA_MIN) / (DATA_MAX - DATA_MIN)
        hr_patch = (hr_patch * 2.0) - 1.0

        if random.random() > 0.5:
            hr_patch = np.ascontiguousarray(np.flip(hr_patch, axis=2))
        if random.random() > 0.5:
            hr_patch = np.ascontiguousarray(np.flip(hr_patch, axis=1))

        hr_tensor = torch.from_numpy(hr_patch)

        lr_10m_tensor = F.interpolate(
            hr_tensor.unsqueeze(0), 
            scale_factor=0.5,  
            mode='area'
        ).squeeze(0)

        lr_20m_tensor = F.interpolate(
            hr_tensor.unsqueeze(0), 
            scale_factor=0.25, 
            mode='area'
        ).squeeze(0)
        
        lr_30m_tensor = F.interpolate(
            hr_tensor.unsqueeze(0), 
            scale_factor=0.125, 
            mode='area'
        ).squeeze(0)

        return hr_tensor, lr_10m_tensor, lr_20m_tensor, lr_30m_tensor
    
    def __len__(self):
        return len(self.hr_files)
