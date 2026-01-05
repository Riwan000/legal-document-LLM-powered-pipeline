"""
Configuration management for the Legal Document Intelligence system.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional, List, Dict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:latest"
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Vector Store Configuration
    VECTOR_STORE_PATH: Path = Path("data/vector_store")
    FAISS_INDEX_NAME: str = "legal_documents.index"
    
    # Document Storage
    DOCUMENTS_PATH: Path = Path("data/documents")
    TEMPLATES_PATH: Path = Path("data/templates")
    
    # Clause Store Configuration
    CLAUSE_STORE_PATH: Path = Path("data/clause_store")
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # RAG Configuration
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    OCR_SIMILARITY_THRESHOLD: float = 0.35  # Lower threshold for OCR documents due to text quality issues
    MIN_SIMILARITY_THRESHOLD: float = 0.25  # Absolute minimum threshold for any search (fallback for very low scores)
    
    # Legal RAG Configuration
    COVERED_TOPICS: List[str] = ["employment", "termination", "benefits", "compensation", "governing law", "dispute resolution"]
    LEGAL_HIERARCHY_KEYWORDS: Dict[str, List[str]] = {
        "law": ["law", "statute", "regulation", "governed by", "pursuant to", "in accordance with", "Saudi Labor Law", "Labor Law"],
        "supremacy": ["override", "supersede", "prevail", "notwithstanding", "subject to", "in compliance with"],
        "contract": ["agreement", "contract", "clause", "provision", "term", "stipulation"]
    }
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TIMEOUT: int = 300  # 5 minutes timeout for long operations
    
    # Case Summarization Configuration
    CASE_SUMMARY_SEED: int = 42  # Fixed seed for deterministic LLM outputs
    CASE_SUMMARY_TEMPERATURE: float = 0.0  # Zero temperature for determinism
    
    # Section-specific top-K limits (PRD requirements)
    CASE_SUMMARY_EXEC_MAX_CHUNKS: int = 8  # Executive summary max chunks
    CASE_SUMMARY_TIMELINE_MAX_CHUNKS: int = 10  # Timeline max chunks
    CASE_SUMMARY_ARGUMENTS_MAX_CHUNKS: int = 10  # Arguments max chunks (per party)
    CASE_SUMMARY_ISSUES_MAX_CHUNKS: int = 6  # Open issues max chunks
    CASE_SUMMARY_SPINE_MAX_CHUNKS: int = 10  # Case spine max chunks
    CASE_SUMMARY_MAX_CHUNKS_PER_CALL: int = 20  # Hard limit per LLM call
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True
    )


# Global settings instance
settings = Settings()

