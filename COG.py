# convert_to_cog.py
import os
import glob
import subprocess
from tqdm import tqdm

def convert_directory_to_cog(input_dir, output_dir):
    """
    Finds all TIFF files in a directory and its subdirectories,
    and converts them to Cloud-Optimized GeoTIFFs (COGs).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_convert = sorted(glob.glob(os.path.join(input_dir, '**', '*.tif'), recursive=True))
    files_to_convert.extend(sorted(glob.glob(os.path.join(input_dir, '**', '*.tiff'), recursive=True)))

    print(f"Found {len(files_to_convert)} files to convert to COG format.")

    for input_path in tqdm(files_to_convert, desc="Converting to COG"):
        relative_path = os.path.relpath(os.path.dirname(input_path), input_dir)
        output_sub_dir = os.path.join(output_dir, relative_path)
        os.makedirs(output_sub_dir, exist_ok=True)
        
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_sub_dir, filename)

        command = [
            'rio', 'cogeo', 'create', 
            input_path,
            output_path,
            '-p', 'lzw'
        ]
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {input_path}.")
            print(f"Command failed with exit code {e.returncode}")
            print(f"Stderr: {e.stderr}")

if __name__ == "__main__":
    input_directory = '/home/bhargavp22co/Ahmedabad/Testing/Tiles'
    output_directory = '/home/bhargavp22co/Ahmedabad/Testing/Tile_COG'
    convert_directory_to_cog(input_directory, output_directory)
    print("\nConversion to COG format complete.")
