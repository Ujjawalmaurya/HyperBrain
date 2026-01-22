import os
import requests
import zipfile
import subprocess
import sys
from pathlib import Path

# Configuration
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, filename):
    filepath = DATA_DIR / filename
    if filepath.exists():
        print(f"File {filename} already exists. Skipping.")
        return filepath
    
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}.")
        return filepath
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None

def download_kaggle_dataset(dataset_name, output_dir_name):
    """
    Requires 'kaggle' CLI tool and credentials (kaggle.json).
    """
    output_path = DATA_DIR / output_dir_name
    if output_path.exists():
        print(f"Dataset {dataset_name} seems to be present at {output_path}. Skipping.")
        return

    print(f"Attempting to download {dataset_name} from Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(DATA_DIR), "--unzip"], check=True)
        # Rename/Organize if needed (Kaggle usually unzips into the target path or creates a mess of files)
        # For simplicity, we assume it unzips into the current folder or a subfolder. 
        # A robust script would move files into `output_path`.
        print(f"Successfully downloaded {dataset_name}.")
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please install it with 'pip install kaggle' and set up your kaggle.json key.")
        print(f"Manual instruction: Download {dataset_name} from Kaggle and extract to {output_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from Kaggle: {e}")
        print(f"Manual instruction: Download {dataset_name} from Kaggle and extract to {output_path}.")

def download_plant_village():
    print("\n--- Sourcing PlantVillage Dataset ---")
    # Using a Kaggle mirror for PlantVillage is usually the easiest way
    download_kaggle_dataset("emmarex/plantdisease", "plant_village")

def download_pest_dataset():
    print("\n--- Sourcing Agricultural Pest Dataset ---")
    download_kaggle_dataset("vbookshelf/rice-leaf-diseases", "pest_dataset") # Example pest dataset

def download_ndvi_samples():
    print("\n--- Sourcing NDVI Sample Imagery ---")
    # Using a placeholder implementation or a known public sample
    # For now, we'll create a dummy NDVI file if we can't find a direct link
    dummy_ndvi_path = DATA_DIR / "sample_ndvi.tif"
    if not dummy_ndvi_path.exists():
        print("Creating placeholder NDVI sample file...")
        with open(dummy_ndvi_path, "w") as f:
            f.write("Placeholder content for NDVI TIFF data.")
    else:
        print("NDVI sample already exists.")

def main():
    print(f"Starting Dataset Sourcing... Target Directory: {DATA_DIR.absolute()}")
    
    # 1. PlantVillage
    download_plant_village()
    
    # 2. Agricultural Pest Dataset
    download_pest_dataset()
    
    # 3. NDVI Samples
    download_ndvi_samples()
    
    print("\nDataset Sourcing process completed (check for any manual steps required).")

if __name__ == "__main__":
    main()
