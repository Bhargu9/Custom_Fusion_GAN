import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


from model import MultiScaleFusionGenerator, Discriminator
from dataset import COGPatchedDataset
from utils import save_labeled_image_grid

def main():
    parser = argparse.ArgumentParser(description="Advanced Multi-Scale SRGAN Training Script")
    
    parser.add_argument("--cog_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_rrdb_blocks", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--n_pretrain_epochs", type=int, default=50)
    parser.add_argument("--n_gan_epochs", type=int, default=200)
    
    parser.add_argument("--lr_pretrain", type=float, default=5e-5)
    parser.add_argument("--lr_gan", type=float, default=5e-5)
    
    parser.add_argument("--patience_pretrain", type=int, default=10)
    parser.add_argument("--patience_gan", type=int, default=20)
    
    parser.add_argument("--lambda_adv", type=float, default=1e-3)
    parser.add_argument("--lambda_pixel", type=float, default=1.0)
    
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("images/validation_grids_pretrain", exist_ok=True)
    os.makedirs("images/validation_grids_gan", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    pretrain_model_path = "saved_models/generator_pretrained.pth"
    gan_model_path_G = "saved_models/generator_gan.pth"
    gan_model_path_D = "saved_models/discriminator_gan.pth"

    generator = MultiScaleFusionGenerator(
        in_channels=3, num_features=64, 
        n_rrdb_blocks=args.n_rrdb_blocks, num_heads=8
    ).to(device)

    discriminator = Discriminator(input_shape=(3, args.patch_size, args.patch_size)).to(device)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    dataset = COGPatchedDataset(args.cog_dir, patch_size=args.patch_size)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    criterion_pixel = nn.L1Loss().to(device)


    # Stage 1: Pre-Training
    
    if os.path.exists(pretrain_model_path):
        print(f"Found existing pre-trained model at {pretrain_model_path}")
        print(" SKIPPING STAGE 1 (PRE-TRAINING)")
    else:
        print("STARTING STAGE 1: GENERATOR PRE-TRAINING (L1)")
        optimizer_G_pre = torch.optim.Adam(generator.parameters(), lr=args.lr_pretrain, betas=(0.9, 0.999))
        scheduler_G_pre = ReduceLROnPlateau(optimizer_G_pre, 'max', patience=5, factor=0.5, verbose=True)
        
        best_psnr_pre = 0.0
        best_ssim_pre = 0.0
        patience_counter_pre = 0

        for epoch in range(args.n_pretrain_epochs):
            generator.train()
            for i, (hr, lr_10m, lr_20m, lr_30m) in enumerate(train_loader):
                hr = hr.to(device); lr_10m = lr_10m.to(device)
                lr_20m = lr_20m.to(device); lr_30m = lr_30m.to(device)

                optimizer_G_pre.zero_grad()
                gen_hr = generator(lr_10m, lr_20m, lr_30m)
                loss_pixel = criterion_pixel(gen_hr, hr)
                loss_pixel.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_grad_norm)
                optimizer_G_pre.step()

                if i % 20 == 0:
                    print(f"[PRE-TRAIN Epoch {epoch}/{args.n_pretrain_epochs}] [Batch {i}/{len(train_loader)}] [G L1 loss: {loss_pixel.item():.4f}]")

            generator.eval()
            val_psnr = 0.0
            val_ssim = 0.0
            
            val_hr, val_lr_10m, val_lr_20m, val_lr_30m = next(iter(val_loader))
            val_hr = val_hr.to(device); val_lr_10m = val_lr_10m.to(device)
            val_lr_20m = val_lr_20m.to(device); val_lr_30m = val_lr_30m.to(device)

            with torch.no_grad():
                gen_hr_val = generator(val_lr_10m, val_lr_20m, val_lr_30m)
                gen_hr_val_norm = torch.clamp((gen_hr_val + 1) / 2.0, 0.0, 1.0)
                val_hr_norm = torch.clamp((val_hr + 1) / 2.0, 0.0, 1.0)
                
                val_psnr = psnr(gen_hr_val_norm, val_hr_norm).item()
                val_ssim = ssim(gen_hr_val_norm, val_hr_norm).item()
            
            scheduler_G_pre.step(val_psnr)
            print(f"\n--- PRE-TRAIN Validation Epoch {epoch} --- Avg. PSNR: {val_psnr:.4f} dB | Avg. SSIM: {val_ssim:.4f}\n")

            if val_psnr > best_psnr_pre or val_ssim > best_ssim_pre:
                if val_psnr > best_psnr_pre:
                    best_psnr_pre = val_psnr
                    print(f"New best pre-train PSNR: {best_psnr_pre:.4f} dB")
                if val_ssim > best_ssim_pre:
                    best_ssim_pre = val_ssim
                    print(f"New best pre-train SSIM: {best_ssim_pre:.4f}")

                patience_counter_pre = 0
                torch.save(generator.state_dict(), pretrain_model_path)
                print("Saving best pre-train model.")
            else:
                patience_counter_pre += 1
                print(f"No improvement in PSNR or SSIM. Patience: {patience_counter_pre}/{args.patience_pretrain}")
                if patience_counter_pre >= args.patience_pretrain:
                    print(f"Pre-training early stopping triggered after {args.patience_pretrain} epochs.")
                    break
        
        print(f"--- STAGE 1 (PRE-TRAINING) FINISHED ---")
        print(f"Best pre-trained model saved to {pretrain_model_path}")

    # STAGE 2: GAN TRAINING
    print("STARTING STAGE 2: GAN TRAINING")
    
    try:
        print(f"Loading best pre-trained model from {pretrain_model_path}")
        generator.load_state_dict(torch.load(pretrain_model_path))
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        print("CRITICAL: Could not load pre-trained weights. Exiting.")
        return 

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_gan, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_gan, betas=(0.9, 0.999))
    scheduler_G_gan = ReduceLROnPlateau(optimizer_G, 'max', patience=5, factor=0.5, verbose=True)
    criterion_adv = nn.BCEWithLogitsLoss().to(device)

    best_psnr_gan = 0.0
    best_ssim_gan = 0.0
    patience_counter_gan = 0

    for epoch in range(args.n_gan_epochs):
        generator.train()
        discriminator.train()

        for i, (hr, lr_10m, lr_20m, lr_30m) in enumerate(train_loader):
            hr = hr.to(device); lr_10m = lr_10m.to(device)
            lr_20m = lr_20m.to(device); lr_30m = lr_30m.to(device)
            
            valid = torch.ones(hr.size(0), 1, device=device, dtype=torch.float)
            fake = torch.zeros(hr.size(0), 1, device=device, dtype=torch.float)

            optimizer_G.zero_grad()
            gen_hr = generator(lr_10m, lr_20m, lr_30m)
            
            loss_pixel = criterion_pixel(gen_hr, hr)
            pred_fake = discriminator(gen_hr)
            loss_adv = criterion_adv(pred_fake, valid)
            loss_G = loss_pixel * args.lambda_pixel + loss_adv * args.lambda_adv
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_grad_norm)
            optimizer_G.step()

            optimizer_D.zero_grad()
            pred_real = discriminator(hr)
            loss_real = criterion_adv(pred_real, valid)
            pred_fake = discriminator(gen_hr.detach())
            loss_fake = criterion_adv(pred_fake, fake)
            loss_D = (loss_real + loss_fake) / 2
            
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_grad_norm)
            optimizer_D.step()

            if i % 20 == 0:
                print(f"[GAN Epoch {epoch}/{args.n_gan_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f} (pixel: {loss_pixel.item():.4f}, adv: {loss_adv.item():.4f})]")

        generator.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        
        val_hr, val_lr_10m, val_lr_20m, val_lr_30m = next(iter(val_loader))
        val_hr = val_hr.to(device); val_lr_10m = val_lr_10m.to(device)
        val_lr_20m = val_lr_20m.to(device); val_lr_30m = val_lr_30m.to(device)

        with torch.no_grad():
            gen_hr_val = generator(val_lr_10m, val_lr_20m, val_lr_30m)
            gen_hr_val_norm = torch.clamp((gen_hr_val + 1) / 2.0, 0.0, 1.0)
            val_hr_norm = torch.clamp((val_hr + 1) / 2.0, 0.0, 1.0)
            
            val_psnr = psnr(gen_hr_val_norm, val_hr_norm).item()
            val_ssim = ssim(gen_hr_val_norm, val_hr_norm).item()

            if epoch % 5 == 0:
                h, w = args.patch_size, args.patch_size
                val_lr_10m_viz = F.interpolate(val_lr_10m, size=(h,w), mode='nearest')
                val_lr_20m_viz = F.interpolate(val_lr_20m, size=(h,w), mode='nearest')
                val_lr_30m_viz = F.interpolate(val_lr_30m, size=(h,w), mode='nearest') 
                val_lr_10m_norm = (val_lr_10m_viz + 1) / 2.0
                val_lr_20m_norm = (val_lr_20m_viz + 1) / 2.0
                val_lr_30m_norm = (val_lr_30m_viz + 1) / 2.0 
                image_dict = {
                    "HR (Ground Truth)": val_hr_norm.squeeze(0),
                    "SR (GAN)": gen_hr_val_norm.squeeze(0),
                    "LR (10m)": val_lr_10m_norm.squeeze(0),
                    "LR (20m)": val_lr_20m_norm.squeeze(0),
                    "LR (30m)": val_lr_30m_norm.squeeze(0)
                }
                save_labeled_image_grid(image_dict, f"images/validation_grids_gan/epoch_{epoch}.png")
        
        scheduler_G_gan.step(val_psnr)
        print(f"\n--- GAN Validation Epoch {epoch} --- Avg. PSNR: {val_psnr:.4f} dB | Avg. SSIM: {val_ssim:.4f}\n")

        if val_psnr > best_psnr_gan or val_ssim > best_ssim_gan:
            if val_psnr > best_psnr_gan:
                best_psnr_gan = val_psnr
                print(f"New best GAN PSNR: {best_psnr_gan:.4f} dB")
            if val_ssim > best_ssim_gan:
                best_ssim_gan = val_ssim
                print(f"New best GAN SSIM: {best_ssim_gan:.4f}")

            patience_counter_gan = 0
            torch.save(generator.state_dict(), gan_model_path_G)
            torch.save(discriminator.state_dict(), gan_model_path_D)
            print("Saving best GAN model.")
        else:
            patience_counter_gan += 1
            print(f"No improvement in PSNR or SSIM. Patience: {patience_counter_gan}/{args.patience_gan}")
            if patience_counter_gan >= args.patience_gan:
                print(f"GAN training early stopping triggered after {args.patience_gan} epochs.")
                break
    
    print("--- STAGE 2 (GAN TRAINING) FINISHED ---")

if __name__ == "__main__":
    main()
