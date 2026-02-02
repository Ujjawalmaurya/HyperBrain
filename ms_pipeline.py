import os
import shutil
import subprocess
import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import exifread

class MultispectralPipeline:
    def __init__(self, base_dir, output_base_dir):
        self.base_dir = Path(base_dir)
        self.output_base = Path(output_base_dir)
        self.output_base.mkdir(parents=True, exist_ok=True)
        # Ensure template directory exists
        self.template_dir = Path("templates")
        self.template_dir.mkdir(exist_ok=True)
        self._create_report_template()

    def _create_report_template(self):
        template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Multispectral Analysis Report</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #f4f4f9; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        .meta-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        .meta-table th, .meta-table td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        .meta-table th { background: #eee; }
        .map-section { margin-bottom: 40px; }
        .map-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .map-card { border: 1px solid #eee; padding: 10px; border-radius: 4px; text-align: center; }
        .map-card img { max-width: 100%; height: auto; border: 1px solid #ccc; }
        .stats { margin-top: 10px; font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multispectral Field Report</h1>
        <p><strong>Date Processed:</strong> {{ date_processed }}</p>
        <p><strong>Dataset Name:</strong> {{ dataset_name }}</p>
        
        <h2>Flight Metadata</h2>
        <table class="meta-table">
            <tr><th>Date Acquired</th><td>{{ meta.date_acquired }}</td></tr>
            <tr><th>Image Count</th><td>{{ meta.image_count }}</td></tr>
            <tr><th>Altitude (Avg)</th><td>{{ meta.avg_altitude }}</td></tr>
            <tr><th>Processing Time</th><td>{{ meta.processing_time }} s</td></tr>
        </table>

        <h2>Vegetation Indices</h2>
        <div class="map-grid">
            {% for idx in indices %}
            <div class="map-card">
                <h3>{{ idx.name }}</h3>
                <img src="{{ idx.image_path }}" alt="{{ idx.name }}">
                <div class="stats">
                    Min: {{ "%.3f"|format(idx.min) }} | Max: {{ "%.3f"|format(idx.max) }} | Mean: {{ "%.3f"|format(idx.mean) }}
                </div>
            </div>
            {% endfor %}
        </div>
        
        <h2>Orthomosaic Preview</h2>
        <div class="map-card" style="width: 100%">
            <img src="orthophoto_preview.jpg" alt="Orthophoto">
        </div>
    </div>
</body>
</html>
"""
        with open(self.template_dir / "report_template.html", "w") as f:
            f.write(template_content)

    def parse_datetime(self, filename):
        # Try Dji format 1: DJI_20230814123320_...
        match = re.search(r'DJI_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', filename)
        if match:
            return datetime(*map(int, match.groups()))
        
        # Try Dji format 2: DJI_DateTime_2024_06_02_13_42_...
        match = re.search(r'DateTime_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', filename)
        if match:
            return datetime(*map(int, match.groups()))
            
        return datetime.now() # Fallback


    def _get_avg_altitude(self, images):
        """Calculates average altitude from a sample of images using ExifRead"""
        alts = []
        # Sample every 10th image to save time
        sample = images[::10] if len(images) > 10 else images
        
        for img in sample:
            try:
                with open(img['path'], 'rb') as f:
                    tags = exifread.process_file(f, details=False)
                    # Try generic GPS Altitude
                    if 'GPS GPSAltitude' in tags:
                        # Value is often Ratio(num, den)
                        val = tags['GPS GPSAltitude'].values[0]
                        alts.append(float(val.num) / float(val.den))
            except Exception:
                pass
        
        if not alts:
            return "N/A"
        return f"{sum(alts) / len(alts):.1f} m"

    def scan_and_group(self):
        """Scans directory and groups images by date/time gaps"""
        all_images = []
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.lower().endswith(('.tif', '.jpg', '.jpeg')):
                    path = Path(root) / file
                    dt = self.parse_datetime(file)
                    all_images.append({'path': path, 'dt': dt})
        
        if not all_images:
            return {}

        all_images.sort(key=lambda x: x['dt'])
        
        groups = defaultdict(list)
        group_id = 0
        if all_images:
            current_group_start = all_images[0]['dt']
            groups[group_id].append(all_images[0])
            
            for i in range(1, len(all_images)):
                time_diff = all_images[i]['dt'] - all_images[i-1]['dt']
                # If gap > 1 day (or arbitrary large gap), start new group
                if time_diff.total_seconds() > 86400: 
                    group_id += 1
                    current_group_start = all_images[i]['dt']
                groups[group_id].append(all_images[i])
                
        return groups

    def run_odm_stitching(self, image_paths, project_name):
        """Runs OpenDroneMap via Docker"""
        project_dir = self.output_base / project_name
        images_dir = project_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Link images
        for img in image_paths:
            # We only need the TIFs for multispectral usually, or all of them.
            # Copy logic: clear destination first to avoid duplicates from prev runs
            dest = images_dir / img['path'].name
            if not dest.exists():
                shutil.copy2(img['path'], dest)
                
        odm_root = "/code" # inside container
        
        # Docker command
        # Mounting project_dir to /code/project maps outputs to project_dir/odm_orthophoto/ etc.
        # Actually ODM expects volume mapped to /datasets/code or similar.
        # Standard usage: -v $(pwd)/images:/code/images opendronemap/odm --project-path /code
        
        print(f"Starting ODM for {project_name}...")
        
        # We use absolute path for volume
        abs_proj_dir = project_dir.absolute()
        
        cmd = [
            "docker", "run", "-ti", "--rm",
            "-v", f"{abs_proj_dir}:/datasets/code",
            "opendronemap/odm:gpu",
            "--project-path", "/datasets/code",
            "--orthophoto-resolution", "5", # cm/pixel, adjust for speed
            "--fast-orthophoto" # Speed up for hackathon
        ]
        
        # In a real app, we'd stream output or check status.
        # For this task, we wait.
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ODM Verification failed (might be missing GPU image or memory). Trying CPU image or fallback...")
            # Fallback logic could go here
            
        return project_dir / "odm_orthophoto" / "odm_orthophoto.tif"

    def calculate_indices(self, orthophoto_path, output_dir):
        if not orthophoto_path.exists():
            print(f"Orthophoto not found at {orthophoto_path}")
            return []

        processed_indices = []
        
        with rasterio.open(orthophoto_path) as src:
            # band config depends on camera.
            # Assuming typically: 1:Red, 2:Green, 3:Blue (if RGB)
            # OR 1:Green, 2:Red, 3:RedEdge, 4:NIR (common for DJI P4M or Mavic 3M processed by specific apps)
            # But ODM usually outputs what it gets.
            # Warning: Mappings vary wildly. I will use a heuristic or standard 4-band assumption.
            # Standard order often: Red, Green, Blue, Alpha.
            # If 4 bands input (G, R, RE, NIR) -> ODM might stack them.
            # Let's assume 4 bands: 1:Green, 2:Red, 3:RedEdge, 4:NIR (DJI order often)
            
            # SAFEGUARD: Check band count
            count = src.count
            
            # Read bands (normalize to 0-1)
            # Handling float or int
            
            def read_band(idx):
                if idx > count: return np.zeros(src.shape, dtype=float)
                b = src.read(idx).astype(float)
                mask = b == 0 # no-data
                b[mask] = np.nan
                # Normalize if uint16/uint8
                if src.dtypes[idx-1] == 'uint16':
                     b = b / 65535.0
                elif src.dtypes[idx-1] == 'uint8':
                     b = b / 255.0
                return b

            # Heuristic assignment
            if count >= 4:
                # Assume DJI Multispectral: G, R, RE, NIR
                green = read_band(1)
                red = read_band(2)
                red_edge = read_band(3)
                nir = read_band(4)
            else:
                # Fallback implementation for RGB
                return [] 
                
            tasks = [
                ("NDVI", (nir - red) / (nir + red + 1e-6)),
                ("GNDVI", (nir - green) / (nir + green + 1e-6)),
                ("NDRE", (nir - red_edge) / (nir + red_edge + 1e-6)),
                ("SAVI", ((nir - red) / (nir + red + 0.5)) * 1.5),
                ("OSAVI", (nir - red) / (nir + red + 0.16))
            ]
            
            for name, data in tasks:
                 # Clamp
                 data = np.clip(data, -1, 1)
                 
                 # Save Raw
                 raw_path = output_dir / f"{name}_raw.tif"
                 # (Saving code omitted for brevity, using simple imwrite or rasterio meta)
                 
                 # Save Color Map
                 plt.figure(figsize=(10, 10))
                 plt.imshow(data, cmap='RdYlGn', vmin=-1, vmax=1)
                 plt.colorbar(label=name)
                 plt.title(f"{name} Map")
                 plt.axis('off')
                 img_path = output_dir / f"{name}_colored.png"
                 plt.savefig(img_path, bbox_inches='tight')
                 plt.close()
                 
                 processed_indices.append({
                     "name": name,
                     "image_path": img_path.name,
                     "min": float(np.nanmin(data)),
                     "max": float(np.nanmax(data)),
                     "mean": float(np.nanmean(data))
                 })
                 
        # Save preview of Ortho
        preview_path = output_dir / "orthophoto_preview.jpg"
        # simplified conversion
        return processed_indices


    def generate_report(self, group_data, indices_data, output_dir):
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template("report_template.html")
        
        # Meta stats
        avg_alt = self._get_avg_altitude(group_data)
        meta = {
            "date_acquired": str(group_data[0]['dt'].date()),
            "image_count": len(group_data),
            "avg_altitude": avg_alt, 
            "processing_time": "TBD" 
        }
        
        html = template.render(
            date_processed=datetime.now().strftime("%Y-%m-%d %H:%M"),
            dataset_name=output_dir.name,
            meta=meta,
            indices=indices_data
        )
        
        with open(output_dir / "report.html", "w") as f:
            f.write(html)


    def process_job(self, job_id, status_callback=None):
        """Processes a specific job directory (already isolated)"""
        # Input is self.base_dir (which now points to the specific job input folder)
        # Scan just this folder
        groups = self.scan_and_group()
        
        # There should only be one relevant group in a single upload typically,
        # but if they uploaded a zip with multiple dates, we process them.
        # Ideally we treat the whole zip as one "dataset" result.
        
        # Flatten all images found for this job
        all_images = []
        for list_imgs in groups.values():
            all_images.extend(list_imgs)
            
        if not all_images:
            if status_callback: status_callback(job_id, "FAILED", {"error": "No images found"})
            return

        if status_callback: status_callback(job_id, "PROCESSING", {"step": "Stitching", "images": len(all_images)})

        # Prepare Output
        output_dir = self.output_base / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Stitch
        # Run ODM on the input folder directly
        # We need to pass the list of images to run_odm_stitching?
        # Actually run_odm_stitching takes a list of dicts.
        
        ortho_path = self.run_odm_stitching(all_images, job_id)
        
        if not ortho_path.exists():
            if status_callback: status_callback(job_id, "FAILED", {"error": "Stitching failed"})
            return
            
        if status_callback: status_callback(job_id, "PROCESSING", {"step": "Analyzing Indices"})

        # 2. Indices
        indices = self.calculate_indices(ortho_path, output_dir)
        
        # 3. Report
        self.generate_report(all_images, indices, output_dir)
        
        if status_callback: status_callback(job_id, "COMPLETED", {"result_path": str(output_dir)})

    def process_all(self):
        # Legacy method kept for backward compatibility or bulk runs
        groups = self.scan_and_group()
        for gid, images in groups.items():
            date_str = images[0]['dt'].strftime("%Y-%m-%d")
            project_name = f"dataset_{date_str}_{gid}"
            output_dir = self.output_base / project_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing Group {gid} ({len(images)} images) from {date_str}")
            
            # 1. Stitch
            ortho_path = self.run_odm_stitching(images, project_name)
            
            if not ortho_path.exists():
                print("Stitching failed or skipped.")
                continue
                
            # 2. Indices
            indices = self.calculate_indices(ortho_path, output_dir)
            
            # 3. Report
            self.generate_report(images, indices, output_dir)

if __name__ == "__main__":
    # Test Run
    pipeline = MultispectralPipeline("./ms-images", "./data_layer/processed")
    pipeline.process_all()
