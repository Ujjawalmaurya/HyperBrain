
import requests
import zipfile
import time
import os
import glob

# 1. Create a dummy zip from existing images
test_zip = "test_upload.zip"
print("Creating zip...")
files = glob.glob("ms-images/Multispectral Images on Paddy- Sri Lanka/*.TIF")[:5] # Take 5 images
with zipfile.ZipFile(test_zip, 'w') as zf:
    for f in files:
        zf.write(f, os.path.basename(f))

# 2. Upload
url = "http://localhost:8000"
print(f"Uploading {test_zip} to {url}/process-dataset...")
with open(test_zip, 'rb') as f:
    resp = requests.post(f"{url}/process-dataset", files={"file": f})

if resp.status_code != 200:
    print("Upload failed:", resp.text)
    exit(1)

job_id = resp.json()['job_id']
print(f"Job ID: {job_id}")

# 3. Poll
while True:
    status_resp = requests.get(f"{url}/status/{job_id}")
    status = status_resp.json()
    print(f"Status: {status['status']} - {status.get('step', '')}")
    
    if status['status'] in ['COMPLETED', 'FAILED']:
        break
    time.sleep(2)

# 4. Report
if status['status'] == 'COMPLETED':
    report = requests.get(f"{url}/report/{job_id}").json()
    print("Report URL:", report['report_url'])
    print("Indices:", report['indices'])

# Clean up
os.remove(test_zip)
