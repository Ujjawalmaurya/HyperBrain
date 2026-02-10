import os
import json

JOB_STORE_FILE = "job_status.json"
JOB_STORE = {}

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

def update_job_status(job_id, status, details=None):
    if job_id not in JOB_STORE:
        JOB_STORE[job_id] = {}
    
    JOB_STORE[job_id]["status"] = status
    if details:
        JOB_STORE[job_id].update(details)
    save_jobs()

def get_job(job_id):
    return JOB_STORE.get(job_id)

# Initialize on import
load_jobs()
