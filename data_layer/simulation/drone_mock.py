import os
import sys
import time
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw

# Add HyperBrain root to path to allow imports
current_dir = Path(__file__).resolve().parent
hyperbrain_root = current_dir.parent.parent
sys.path.append(str(hyperbrain_root))

from data_layer.metadata.db_manager import db_manager

# Configuration
RAW_DATA_DIR = Path("data/raw/drone_uploads")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

class DroneSimulator:
    def __init__(self, start_lat=28.7041, start_lon=77.1025):
        self.lat = start_lat
        self.lon = start_lon
        self.altitude = 50.0 # meters
        
    def move(self):
        """Simulate drone movement."""
        # Simple random walk / lawnmower pattern simulation
        self.lat += random.uniform(-0.0001, 0.0001)
        self.lon += random.uniform(-0.0001, 0.0001)
        self.altitude += random.uniform(-0.5, 0.5)

    def capture_image(self) -> Path:
        """Generates a dummy image."""
        img_name = f"drone_{uuid.uuid4().hex[:8]}.jpg"
        img_path = RAW_DATA_DIR / img_name
        
        # Create a basic image
        img = Image.new('RGB', (640, 480), color = (random.randint(0, 50), random.randint(100, 255), random.randint(0, 50)))
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Simulated: {datetime.now()}", fill=(255, 255, 255))
        d.text((10, 30), f"GPS: {self.lat:.4f}, {self.lon:.4f}", fill=(255, 255, 255))
        
        img.save(img_path)
        print(f"Captured: {img_path}")
        return img_path

    def upload_metadata(self, img_path: Path):
        """Uploads metadata to the DB."""
        metadata = {
            "filename": img_path.name,
            "original_filename": img_path.name,
            "capture_timestamp": datetime.utcnow(),
            "gps_latitude": self.lat,
            "gps_longitude": self.lon,
            "altitude_m": self.altitude,
            "resolution_width": 640,
            "resolution_height": 480,
            "file_size_bytes": img_path.stat().st_size,
            "format": "jpg",
            "storage_path": str(img_path),
            "processing_status": "raw"
        }
        
        db_manager.add_image_metadata(metadata)
        print(f"Metadata uploaded for {img_path.name}")

    def run_mission(self, num_images=5, interval=1):
        print(f"Starting Drone Mission... Target: {num_images} images.")
        for i in range(num_images):
            self.move()
            img_path = self.capture_image()
            self.upload_metadata(img_path)
            time.sleep(interval)
        print("Mission Complete.")

if __name__ == "__main__":
    count = 5
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass
            
    drone = DroneSimulator()
    drone.run_mission(num_images=count, interval=0.5)
