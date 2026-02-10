from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import numpy as np
import cv2
import torch
import uuid
from pathlib import Path
from ultralytics import YOLO
from utils.image import calculate_ndvi

router = APIRouter()

# Load models
# Note: In a production app, model loading should probably be in a separate dependency or singleton
# but for now we keep it module-level to match existing behavior.
model = YOLO('yolov8x-cls.pt') 
pest_detector = YOLO('weights/pest.pt')
weed_detector = YOLO('weights/weed.pt')

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if len(image.shape) == 2: # Grayscale
         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    ndvi = calculate_ndvi(image)
    avg_ndvi = np.mean(ndvi)
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    results = model(image, device=device)
    
    # Run Detection Models
    pest_results = pest_detector(image, device=device)
    weed_results = weed_detector(image, device=device)
    
    detected_pests = []
    for r in pest_results:
        for box in r.boxes:
            detected_pests.append({
                "label": pest_detector.names[int(box.cls)],
                "confidence": float(box.conf),
                "box": box.xyxy[0].tolist()
            })

    detected_weeds = []
    for r in weed_results:
        for box in r.boxes:
            detected_weeds.append({
                "label": weed_detector.names[int(box.cls)],
                "confidence": float(box.conf),
                "box": box.xyxy[0].tolist()
            })
    
    # Classification Logic
    top5_probs = []
    disease_detected = False
    
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
            
            if i == 0 and "healthy" not in label.lower() and confidence > 0.4:
                disease_detected = True

    yield_est = float(avg_ndvi * 12 + 4) 
    
    return {
        "ndvi": float(avg_ndvi),
        "disease_detected": disease_detected,
        "predictions": top5_probs,
        "pest_detections": detected_pests,
        "weed_detections": detected_weeds,
        "yield_prediction": yield_est,
        "processing_time": 1.2,
        "metadata": {
            "resolution": f"{image.shape[1]}x{image.shape[0]}",
            "channels": "Multispectral (Simulated)",
            "app_mode": "Classification",
            "model": "YOLOv8-Best-Accuracy"
        },
        "status": "Success"
    }

@router.post("/analyze-batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    results = []
    
    # Paths
    base_output = Path("data_layer/processed/batch") / job_id
    base_output.mkdir(parents=True, exist_ok=True)
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            continue
            
        # Run Detection Models
        pest_res = pest_detector(image, device=device)
        weed_res = weed_detector(image, device=device)
        
        img_detections = {"filename": file.filename, "pests": [], "weeds": []}
        
        annotated = image.copy()
        
        for r in pest_res:
            for box in r.boxes:
                label = pest_detector.names[int(box.cls)]
                conf = float(box.conf)
                coords = box.xyxy[0].tolist()
                img_detections["pests"].append({"label": label, "confidence": conf, "box": coords})
                
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"Pest: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for r in weed_res:
            for box in r.boxes:
                label = weed_detector.names[int(box.cls)]
                conf = float(box.conf)
                coords = box.xyxy[0].tolist()
                img_detections["weeds"].append({"label": label, "confidence": conf, "box": coords})
                
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"Weed: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_filename = f"annotated_{file.filename}"
        output_path = base_output / output_filename
        cv2.imwrite(str(output_path), annotated)
        
        img_detections["result_url"] = f"/data/batch/{job_id}/{output_filename}"
        results.append(img_detections)

    return {
        "job_id": job_id,
        "total_images": len(files),
        "processed_count": len(results),
        "results": results,
        "status": "Success"
    }
