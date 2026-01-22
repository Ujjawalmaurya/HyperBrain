# Storage Architecture Strategy

## Overview
This document outlines the planned storage architecture for the HyperBrain project. The goal is to separate raw data, preprocessed data, and annotated data, while supporting both local development and future cloud deployment.

## Planned Folder Structure
We intend to use a unified directory structure that can be mapped to either a local file system or a cloud object store (S3/GCS).

```
/data
    /raw
        /drone_uploads      # Raw images uploaded from drones
        /external_datasets  # PlantVillage, Kaggle, etc.
    /preprocessed
        /tiles_256x256      # Tiled images for training
        /normalized         # Normalized images
    /annotated
        /coco_labels        # Annotations in COCO format
        /yolo_labels        # Annotations in YOLO format
```

## Abstract Storage Interface
To support switching between Local and Cloud storage without code changes, we will implement a `StorageInterface` strategy pattern.

### Base Interface (Python Protocol)
```python
class StorageInterface:
    def list_files(self, path: str) -> List[str]: ...
    def download_file(self, remote_path: str, local_path: str): ...
    def upload_file(self, local_path: str, remote_path: str): ...
    def exists(self, path: str) -> bool: ...
```

### Implementations
1.  **LocalStorage**: Direct file system operations using `pathlib` and `shutil`.
2.  **CloudStorage**: Interface using `boto3` (AWS) or `google-cloud-storage` (GCP).

## Setup Instructions (Future)
1.  **Local Dev**:
    -   Ensure the `data/` directory is in `.gitignore` (except for sample data).
    -   Use `LocalStorage` implementation.
2.  **Production**:
    -   Provision S3 Bucket / GCS Bucket.
    -   Set environment variables: `STORAGE_TYPE=s3`, `AWS_ACCESS_KEY_ID=...`, `AWS_BUCKET_NAME=...`.
    -   Deploy app using `CloudStorage` implementation.
