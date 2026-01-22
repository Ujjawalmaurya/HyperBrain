import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .schema import Base, ImageMetadata
from typing import Optional, List

# Default to SQLite for Hackathon/Dev
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./hyperbrain_metadata.db")

class DBManager:
    def __init__(self, db_url=DATABASE_URL):
        self.engine = create_engine(
            db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.init_db()

    def init_db(self):
        """Creates tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def add_image_metadata(self, metadata: dict) -> ImageMetadata:
        session = self.get_session()
        try:
            db_item = ImageMetadata(**metadata)
            session.add(db_item)
            session.commit()
            session.refresh(db_item)
            return db_item
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_image_by_filename(self, filename: str) -> Optional[ImageMetadata]:
        session = self.get_session()
        try:
            return session.query(ImageMetadata).filter(ImageMetadata.filename == filename).first()
        finally:
            session.close()

    def get_all_images(self, skip: int = 0, limit: int = 100) -> List[ImageMetadata]:
        session = self.get_session()
        try:
            return session.query(ImageMetadata).offset(skip).limit(limit).all()
        finally:
            session.close()

# Singleton instance for easy import
db_manager = DBManager()
