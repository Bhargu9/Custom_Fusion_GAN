# utils.py
import torch
import rasterio
import numpy as np
import h5py

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

def save_tensor_as_tif(tensor, original_path, save_path):
    """
    Saves a PyTorch tensor as a GeoTIFF file, preserving the georeferencing
    information from an original TIFF file.
    """
    try:
        tensor = (tensor + 1) / 2.0
        tensor = tensor.mul(255).byte()
        
        output_image = tensor.cpu().numpy()

        with rasterio.open(original_path) as src:
            profile = src.profile
            profile.update(
                dtype=rasterio.uint8,
                count=output_image.shape[0],
                height=output_image.shape[1],
                width=output_image.shape[2],
                compress='lzw'
            )
        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(output_image)
    except Exception as e:
        print(f"Error saving tensor to {save_path}: {e}")

def save_weights_as_h5(model, save_path):
    """
    Saves the weights of a PyTorch model to an HDF5 file.
    """
    try:
        with h5py.File(save_path, 'w') as hf:
            for name, param in model.named_parameters():
                hf.create_dataset(name, data=param.cpu().numpy())
        print(f"Model weights saved to {save_path}")
    except Exception as e:
        print(f"Error saving weights to {save_path}: {e}")

def save_labeled_image_grid(image_dict, save_path):
    """
    Creates and saves a grid of images with text labels above them.
    
    Args:
        image_dict (dict): A dictionary where keys are labels (str) and
                           values are image tensors (C, H, W) in [-1, 1] range.
        save_path (str): Path to save the final grid image.
    """
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    labeled_images = []
    max_height = 0
    max_width = 0

    for label, tensor in image_dict.items():
        tensor_norm_0_1 = (tensor.cpu().clamp(-1, 1) + 1.0) / 2.0
        
        pil_img = TF.to_pil_image(tensor_norm_0_1)
        max_height = max(max_height, pil_img.height)
        max_width = max(max_width, pil_img.width)

        labeled_img = Image.new('RGB', (pil_img.width, pil_img.height + 40), (0, 0, 0))
        labeled_img.paste(pil_img, (0, 40))

        draw = ImageDraw.Draw(labeled_img)
        draw.text((10, 5), label, font=font, fill=(255, 255, 255))
        
        labeled_images.append(labeled_img)

    total_width = sum(img.width for img in labeled_images)
    grid_height = max(img.height for img in labeled_images)
    
    grid_image = Image.new('RGB', (total_width, grid_height))
    current_x = 0
    for img in labeled_images:
        grid_image.paste(img, (current_x, 0))
        current_x += img.width

    grid_image.save(save_path)
    print(f"Saved validation grid to {save_path}")
