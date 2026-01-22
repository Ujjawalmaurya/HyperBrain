from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionalias
from datetime import datetime
import enum

Base = declarative_base()

class ValidationStatus(enum.Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"

class ImageMetadata(Base):
    __tablename__ = 'image_metadata'

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True, nullable=False)
    original_filename = Column(String, nullable=True)
    
    # Capture Info
    capture_timestamp = Column(DateTime, default=datetime.utcnow)
    gps_latitude = Column(Float, nullable=True)
    gps_longitude = Column(Float, nullable=True)
    altitude_m = Column(Float, nullable=True)
    
    # Image Properties
    resolution_width = Column(Integer, nullable=True)
    resolution_height = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    format = Column(String, nullable=True) # jpg, png, tif
    
    # Agricultural Context
    crop_type = Column(String, nullable=True, index=True) # e.g., "Wheat", "Corn"
    field_id = Column(String, nullable=True)
    
    # Storage & Processing
    storage_path = Column(String, nullable=False) # Relative path in storage
    processing_status = Column(String, default="raw") # raw, processed, annotated
    validation_status = Column(Enum(ValidationStatus), default=ValidationStatus.PENDING)
    
    annotated_path = Column(String, nullable=True)

    def __repr__(self):
        return f"<ImageMetadata(filename={self.filename}, crop={self.crop_type})>"
