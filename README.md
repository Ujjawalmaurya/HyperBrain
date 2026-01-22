# HyperBrain 🧠

HyperBrain is an AI-powered agricultural analysis service built with FastAPI and YOLOv8. It processes images to calculate NDVI (Normalized Difference Vegetation Index) and detects agricultural anomalies using computer vision.

## Features

- **NDVI Calculation**: Analyzes image channels to assess vegetation health.
- **Anomaly Detection**: Uses YOLOv8 to detect objects and maps them to agricultural insights (e.g., pests, diseases).
- **FastAPI Backend**: fast and efficient REST API for image processing.

## Installation

1. **Clone the repository** (if applicable).
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Locally

Start the server using `uvicorn` (implicitly via `main.py` entry point):

```bash
python main.py
```

The server will run at `http://0.0.0.0:8000`.

### API Endpoints

- **`GET /`**: Health check. Returns status message.
- **`POST /analyze`**: Upload an image file to get analysis results.
  - **Input**: Form-data with `file` field.
  - **Output**: JSON containing NDVI stats, detected objects, yield prediction, and metadata.

### Verification

Run the verification script to ensure the ML pipeline is working correctly:

```bash
python verify_ml.py
```

## Docker

Build and run the container:

```bash
docker build -t hyperbrain .
docker run -p 8000:8000 hyperbrain
```