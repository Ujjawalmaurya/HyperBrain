
import os
import shutil
import uuid
import zipfile
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
import torch
import ms_pipeline

app = FastAPI()

# Mount processed data as static files to serve images/reports
app.mount("/data", StaticFiles(directory="data_layer/processed"), name="data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('yolov8_classification_best_accuracy/weights/best.pt') 

# Simple in-memory job store (use Redis/Db for production)
JOB_STORE = {}
JOB_STORE_FILE = "job_status.json"

def load_jobs():
    global JOB_STORE
    if os.path.exists(JOB_STORE_FILE):
        try:
            with open(JOB_STORE_FILE, 'r') as f:
                JOB_STORE = json.load(f)
        except:
            JOB_STORE = {}

def save_jobs():
    with open(JOB_STORE_FILE, 'w') as f:
        json.dump(JOB_STORE, f)

load_jobs()

def update_job_status(job_id, status, details=None):
    if job_id not in JOB_STORE:
        JOB_STORE[job_id] = {}
    
    JOB_STORE[job_id]["status"] = status
    if details:
        JOB_STORE[job_id].update(details)
    save_jobs()

def run_ms_job(job_id: str, input_dir: str, output_dir: str):
    try:
        pipeline = ms_pipeline.MultispectralPipeline(input_dir, output_dir)
        pipeline.process_job(job_id, status_callback=update_job_status)
    except Exception as e:
        update_job_status(job_id, "FAILED", {"error": str(e)})


@app.post("/process-dataset")
async def process_dataset(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    
    # Paths
    base_input = Path("ms-images") / job_id
    base_output = Path("data_layer/processed")
    
    base_input.mkdir(parents=True, exist_ok=True)
    
    # Save Zip
    zip_path = base_input / "upload.zip"
    try:
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Unzip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_input)
            
        # Init Job
        JOB_STORE[job_id] = {
            "status": "QUEUED", 
            "original_filename": file.filename,
            "created_at": str(os.path.getctime(zip_path))
        }
        save_jobs()
        
        # Trigger Task
        background_tasks.add_task(run_ms_job, job_id, str(base_input), str(base_output))
        
        return {"job_id": job_id, "message": "Dataset queued for processing", "status": "QUEUED"}
        
    except Exception as e:
        # Cleanup
        if base_input.exists():
            shutil.rmtree(base_input)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOB_STORE[job_id]

@app.get("/report/{job_id}")
def get_report(job_id: str):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = JOB_STORE[job_id]
    if job['status'] != 'COMPLETED':
        return {"status": job['status'], "message": "Report not ready"}
        
    # Return path to report or JSON summary
    report_url = f"/data/{job_id}/report.html"
    return {
        "status": "COMPLETED",
        "report_url": report_url,
        "indices": [
            {"name": "NDVI", "url": f"/data/{job_id}/NDVI_colored.png"},
            {"name": "GNDVI", "url": f"/data/{job_id}/GNDVI_colored.png"},
            # ... others
        ]
    }

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
