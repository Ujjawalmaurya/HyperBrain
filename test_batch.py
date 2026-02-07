import requests
import os

url = "http://localhost:8000/analyze-batch"
image_paths = ["1.jpeg", "2.jpeg"]

files = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in image_paths if os.path.exists(p)]

if not files:
    print("No test images found (1.jpeg, 2.jpeg)")
    exit(1)

print(f"Uploading {len(files)} images to {url}...")
try:
    response = requests.post(url, files=files)
    if response.status_code == 200:
        print("Success!")
        data = response.json()
        print(f"Job ID: {data['job_id']}")
        print(f"Processed: {data['processed_count']}")
        for result in data['results']:
            print(f"- {result['filename']}: {len(result['pests'])} pests, {len(result['weeds'])} weeds")
            print(f"  URL: {result['result_url']}")
    else:
        print(f"Failed: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
finally:
    for _, (_, f, _) in files:
        f.close()
