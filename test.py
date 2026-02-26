import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm


# CONFIGURATION

BASE_DIR = '/home/bhargavp22co/Custom'
WEIGHTS_PATH = os.path.join(BASE_DIR, 'saved_models/Old/generator_best.pth')
TEST_IMG_DIR = '/home/bhargavp22co/Ahmedabad/Testing/Tile_COG'
SAVE_DIR = '/home/bhargavp22co/Ahmedabad/Testing/Results_Fusion_Patched'

PATCH_SIZE = 256 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. LEGACY ARCHITECTURE (For loading Old Weights)
class LegacyMultiScaleFusionGenerator(nn.Module):
    def __init__(self, in_channels=3, num_features=64, n_rrdb_blocks=16, num_heads=8):
        super(LegacyMultiScaleFusionGenerator, self).__init__()
        sys.path.append(BASE_DIR)
        from model import ResidualInResidualDenseBlock, UpsampleBlock, SpatialMHSA
        
        self.head_10m = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.head_20m = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.head_30m = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        rrdb_body = [ResidualInResidualDenseBlock(num_features) for _ in range(n_rrdb_blocks)]
        self.RRDB_body_10m = nn.Sequential(*rrdb_body)
        self.RRDB_body_20m = nn.Sequential(*rrdb_body)
        self.RRDB_body_30m = nn.Sequential(*rrdb_body)
        
        self.conv_after_rrdb_10m = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_after_rrdb_20m = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_after_rrdb_30m = nn.Conv2d(num_features, num_features, 3, 1, 1)

        self.downsample_10m_x05 = nn.Sequential(nn.Conv2d(num_features, num_features, 3, 2, 1), nn.PReLU())
        self.upsample_30m_x2 = UpsampleBlock(num_features, scale_factor=2)

        fusion_in_channels = num_features * 3
        self.fusion_conv_1x1 = nn.Conv2d(fusion_in_channels, num_features, 1, 1, 0)
        self.mhsa_block = SpatialMHSA(num_features, num_heads=num_heads)
        self.fusion_conv_3x3 = nn.Conv2d(num_features, num_features, 3, 1, 1)

        self.final_upsample = nn.Sequential(
            UpsampleBlock(num_features, scale_factor=2),
            UpsampleBlock(num_features, scale_factor=2),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_features, in_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x10, x20, x30):
        h10 = self.head_10m(x10); h20 = self.head_20m(x20); h30 = self.head_30m(x30)
        f10 = self.RRDB_body_10m(h10) + h10
        f20 = self.RRDB_body_20m(h20) + h20
        f30 = self.RRDB_body_30m(h30) + h30
        f10 = self.conv_after_rrdb_10m(f10)
        f20 = self.conv_after_rrdb_20m(f20)
        f30 = self.conv_after_rrdb_30m(f30)
        f10_down = self.downsample_10m_x05(f10)
        f30_up = self.upsample_30m_x2(f30)
        f_cat = torch.cat([f10_down, f20, f30_up], dim=1)
        f_fused = self.fusion_conv_1x1(f_cat)
        f_attended = self.mhsa_block(f_fused)
        f_refined = self.fusion_conv_3x3(f_attended)
        return self.final_upsample(f_refined)

# 2. PATCH PROCESSING FUNCTION

def process_patch_fusion(model, hr_patch_tensor, device):
    """
    Takes an HR patch, generates 10m/20m/30m inputs via downsampling,
    and returns the SR output.
    """
    hr_norm = (hr_patch_tensor * 2.0) - 1.0
    
    lr_10m = F.interpolate(hr_norm, scale_factor=0.5, mode='area')
    lr_20m = F.interpolate(hr_norm, scale_factor=0.25, mode='area')
    lr_30m = F.interpolate(hr_norm, scale_factor=0.125, mode='area')
    
    sr_patch = model(lr_10m, lr_20m, lr_30m)
    sr_patch = torch.clamp((sr_patch + 1) / 2.0, 0.0, 1.0)
    return sr_patch

# 3. MAIN TESTING LOOP (TILED)

def main():
    print(f"Loading Legacy Model from {WEIGHTS_PATH}...")
    model = LegacyMultiScaleFusionGenerator(in_channels=3, num_features=64, n_rrdb_blocks=16, num_heads=8).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    # Metrics
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    test_files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '**', '*.tif*'), recursive=True))
    test_files = [f for f in test_files if f.lower().endswith(('.tif', '.tiff'))]

    results_data = []
    print(f"Processing {len(test_files)} images with PATCH_SIZE={PATCH_SIZE}...")

    for img_path in tqdm(test_files):
        filename = os.path.basename(img_path)
        
        with rasterio.open(img_path) as src:
            # Read full HR image
            hr_numpy = src.read([1, 2, 3]).astype(np.float32)
            profile = src.profile
        
        # Normalize HR to [0, 1] for metrics baseline
        hr_tensor_01 = torch.from_numpy(hr_numpy).float().to(DEVICE) / 255.0
        hr_tensor_01 = hr_tensor_01.unsqueeze(0) # [1, C, H, W]
        
        b, c, h, w = hr_tensor_01.shape
        
        #  1. Calculate Padding to fit PATCH_SIZE 
        pad_h = (PATCH_SIZE - (h % PATCH_SIZE)) % PATCH_SIZE
        pad_w = (PATCH_SIZE - (w % PATCH_SIZE)) % PATCH_SIZE
        
        if pad_h > 0 or pad_w > 0:
            # Pad HR image 
            hr_padded = F.pad(hr_tensor_01, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            hr_padded = hr_tensor_01
        
        _, _, h_pad, w_pad = hr_padded.shape
        
        #  2. Create Output Canvas 
        output_canvas = torch.zeros_like(hr_padded)
        
        #  3. Sliding Window Loop 
        with torch.no_grad():
            for y in range(0, h_pad, PATCH_SIZE):
                for x in range(0, w_pad, PATCH_SIZE):
                    # Extract patch
                    hr_patch = hr_padded[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    
                    # Process patch 
                    sr_patch = process_patch_fusion(model, hr_patch, DEVICE)
                    output_canvas[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] = sr_patch

        # 4. Crop back to original size 
        sr_final = output_canvas[:, :, :h, :w]
        
        # 5. Calculate Metrics (Full Image) 
        val_psnr = psnr_fn(sr_final, hr_tensor_01).item()
        val_ssim = ssim_fn(sr_final, hr_tensor_01).item()

        results_data.append({"Image": filename, "PSNR": val_psnr, "SSIM": val_ssim})

        # 6. Save Result 
        save_path = os.path.join(SAVE_DIR, f"SR_{filename}")
        out_img_uint8 = (sr_final.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
        
        profile.update(dtype=rasterio.uint8, count=3, compress='lzw')
        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(out_img_uint8)

    df = pd.DataFrame(results_data)
    print("\n" + "="*40)
    print(f"Average PSNR: {df['PSNR'].mean():.4f}")
    print(f"Average SSIM: {df['SSIM'].mean():.4f}")
    print("="*40)
    df.to_csv(os.path.join(SAVE_DIR, 'metrics.csv'), index=False)

if __name__ == "__main__":
    main()
