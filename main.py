import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import Routers
from routes import analysis, processing
from utils.jobs import load_jobs

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

# Include Routers
app.include_router(processing.router)
app.include_router(analysis.router)

# Initialize Job Store
load_jobs()

@app.get("/")
def home():
    return {"message": "ML Service is running", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
