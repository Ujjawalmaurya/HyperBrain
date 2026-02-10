from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import shutil
import uuid
import os
import zipfile
from pathlib import Path
import ms_pipeline
from utils.jobs import JOB_STORE, save_jobs, update_job_status

router = APIRouter()

def run_ms_job(job_id: str, input_dir: str, output_dir: str):
    try:
        pipeline = ms_pipeline.MultispectralPipeline(input_dir, output_dir)
        pipeline.process_job(job_id, status_callback=update_job_status)
    except Exception as e:
        update_job_status(job_id, "FAILED", {"error": str(e)})

@router.post("/process-dataset")
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

@router.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOB_STORE[job_id]

@router.get("/report/{job_id}")
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
