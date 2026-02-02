import cv2
import numpy as np
import sys
import os

# Ensure we can import from main
sys.path.append(os.getcwd())
try:
    from main import calculate_ndvi
    from ultralytics import YOLO
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_ml_service():
    print("1. Testing Model Loading...")
    model_path = 'yolov8_classification_best_accuracy/weights/best.pt'
    try:
        model = YOLO(model_path)
        print(f"Success: Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("2. Testing NDVI Logic...")
    # Create fake 4-channel image (B, G, R, NIR)
    # Case 1: High Vegetation (High NIR, Low Red)
    img_veg = np.zeros((100, 100, 4), dtype=np.uint8)
    img_veg[:, :, 2] = 50   # Red
    img_veg[:, :, 3] = 200  # NIR
    
    ndvi_veg = calculate_ndvi(img_veg)
    mean_veg = np.mean(ndvi_veg)
    print(f"Mean NDVI (Vegetation): {mean_veg:.3f} (Expected > 0.5)")
    
    if mean_veg > 0.5:
        print("Success: High NDVI correctly detected.")
    else:
        print("Failure: NDVI calculation incorrect for vegetation.")
        
    # Case 2: Water/Soil (Low NIR, High Red)
    img_soil = np.zeros((100, 100, 4), dtype=np.uint8)
    img_soil[:, :, 2] = 200 # Red
    img_soil[:, :, 3] = 50  # NIR
    
    ndvi_soil = calculate_ndvi(img_soil)
    mean_soil = np.mean(ndvi_soil)
    print(f"Mean NDVI (Soil): {mean_soil:.3f} (Expected < 0)")
    
    if mean_soil < 0:
        print("Success: Low NDVI correctly detected.")

    print("3. Testing Inference...")
    # Standard Inference with 4 channels (YOLO should handle it or ignore extra channels)
    try:
        results = model(img_veg) # Using veg image
        if results[0].probs is not None:
             top1 = results[0].probs.top1
             print(f"Inference ran on 4-channel input. Top class index: {top1}")
             print("Success: Classification inference works with multispectral input.")
        else:
             print("Failure: No probabilities returned.")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    test_ml_service()
