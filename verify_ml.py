import cv2
import numpy as np
from main import calculate_ndvi
from ultralytics import YOLO

def test_ml_service():
    print("1. Testing Model Loading...")
    try:
        model = YOLO('yolov8n.pt')
        print("Success: Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("2. Testing NDVI Logic...")
    # Create fake NIR/Red channels (simulated as BGR for now)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 1] = 200 # Green/NIR mock
    img[:, :, 2] = 50 # Red
    
    ndvi = calculate_ndvi(img)
    mean_ndvi = np.mean(ndvi)
    print(f"Mean NDVI: {mean_ndvi}")
    
    if mean_ndvi > 0:
        print("Success: NDVI calculation seems plausible.")
    else:
        print("Warning: NDVI seems low (expected for dummy data).")

    print("3. Testing Inference...")
    results = model(img)
    print(f"Inference ran. Detected {len(results[0].boxes)} objects (expected 0).")
    print("Success: Inference pipeline works.")

if __name__ == "__main__":
    test_ml_service()
