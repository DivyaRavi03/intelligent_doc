"""Application configuration via pydantic-settings.

All settings are read from environment variables (or a .env file).
Import the singleton ``settings`` object wherever configuration is needed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Intelligent Document Processing platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Gemini ----------------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "models/text-embedding-004"

    # --- Database --------------------------------------------------------
    database_url: str = "postgresql+asyncpg://docuser:docpass@localhost:5432/intelligent_doc"

    # --- Redis -----------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # --- ChromaDB --------------------------------------------------------
    chroma_persist_dir: str = "./data/chroma"

    # --- Uploads ---------------------------------------------------------
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50

    # --- Chunking --------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200

    def ensure_dirs(self) -> None:
        """Create upload and chroma directories if they don't exist."""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` singleton."""
    return Settings()


settings: Settings = get_settings()
