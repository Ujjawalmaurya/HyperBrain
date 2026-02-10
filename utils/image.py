import numpy as np
import cv2

def calculate_ndvi(image):
    """
    Calculate NDVI from image.
    Heuristic: Assume BGRN (4 channels), else standard BGR.
    """
    if len(image.shape) == 3 and image.shape[2] >= 4:
        # B, G, R, NIR
        red = image[:, :, 2].astype(float)
        nir = image[:, :, 3].astype(float)
    else:
        # Fallback for RGB: use Blue as NIR simulator (not accurate but keeps it running)
        red = image[:, :, 2].astype(float)
        nir = image[:, :, 0].astype(float) 
        
    numerator = (nir - red)
    denominator = (nir + red + 1e-6)
    return numerator / denominator
