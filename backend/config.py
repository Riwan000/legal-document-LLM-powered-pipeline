"""
Configuration management for the Legal Document Intelligence system.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: Path = Path("data/vector_store")
    FAISS_INDEX_NAME: str = "legal_documents.index"
    
    # Document Storage
    DOCUMENTS_PATH: Path = Path("data/documents")
    TEMPLATES_PATH: Path = Path("data/templates")
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # RAG Configuration
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    OCR_SIMILARITY_THRESHOLD: float = 0.35  # Lower threshold for OCR documents due to text quality issues
    MIN_SIMILARITY_THRESHOLD: float = 0.25  # Absolute minimum threshold for any search (fallback for very low scores)
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True
    )


# Global settings instance
settings = Settings()

