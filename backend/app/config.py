"""
LIFEOS Application Configuration

Centralized settings loaded from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    # ─── Application ─────────────────────────────────
    APP_NAME: str = "LIFEOS"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ─── Database ─────────────────────────────────────
    # Defaults to zero-config SQLite for local dev & testing.
    # Set to mysql+asyncmy://user:pass@localhost:3306/lifeos in .env for MySQL 8.0 production.
    DATABASE_URL: str = "sqlite+aiosqlite:///./lifeos.db"
    DATABASE_URL_SYNC: str = "sqlite:///./lifeos.db"

    # ─── Security ─────────────────────────────────────
    SECRET_KEY: str = "lifeos-dev-secret-key-change-in-production-2024"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # ─── CORS ─────────────────────────────────────────
    CORS_ORIGINS: str = '["*"]'

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.CORS_ORIGINS)

    # ─── ChromaDB ─────────────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIR: str = "data/chroma_db"

    # ─── Upload ───────────────────────────────────────
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    # ─── ML ───────────────────────────────────────────
    ML_MODEL_DIR: str = "ml_models"
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.70

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
