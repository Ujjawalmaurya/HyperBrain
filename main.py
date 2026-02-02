import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('yolov8_classification_best_accuracy/weights/best.pt') 

def calculate_ndvi(image):
    # Check for Multispectral (4 channels: B, G, R, NIR)
    if image.shape[2] >= 4:
        nir = image[:, :, 3].astype(float)
        red = image[:, :, 2].astype(float) # OpenCV uses BGR
        
        # Avoid division by zero
        denominator = nir + red
        denominator[denominator == 0] = 0.001 
        
        ndvi = (nir - red) / denominator
        return ndvi
        
    # Fallback for standard RGB (Simulated NDVI)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ndvi = (gray.astype(float) / 255.0) * 2 - 1
    return ndvi

@app.get("/")
def read_root():
    return {"message": "Hyperspace AI ML Service is running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    # Read UNCHANGED to preserve NIR channel if present
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    # If standard RGB loaded as BGR (3 channels), pass as is
    # If 4 channels (BGRA/BGRN), pass as is.
    
    if len(image.shape) == 2: # Grayscale
         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    ndvi = calculate_ndvi(image)
    avg_ndvi = np.mean(ndvi)
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    results = model(image, device=device)
    
    # Classification Logic
    top5_probs = []
    disease_detected = False
    
    # Extract top 5 predictions
    if results[0].probs is not None:
        probs = results[0].probs
        top5_indices = probs.top5
        top5_conf = probs.top5conf
        
        for i in range(len(top5_indices)):
            class_index = top5_indices[i]
            confidence = float(top5_conf[i])
            label = model.names[int(class_index)]
            
            top5_probs.append({
                "label": label,
                "confidence": confidence
            })
            
            # Simple heuristic: if top prediction is NOT "healthy", assume disease
            if i == 0 and "healthy" not in label.lower() and confidence > 0.4:
                disease_detected = True

    yield_est = float(avg_ndvi * 12 + 4) 
    
    return {
        "ndvi": float(avg_ndvi),
        "disease_detected": disease_detected,
        "predictions": top5_probs,
        "yield_prediction": yield_est,
        "processing_time": 1.2, # Placeholder, could measure actual time
        "metadata": {
            "resolution": f"{image.shape[1]}x{image.shape[0]}",
            "channels": "Multispectral (Simulated)",
            "app_mode": "Classification",
            "model": "YOLOv8-Best-Accuracy"
        },
        "status": "Success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
