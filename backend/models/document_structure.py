"""
Models for document structure detection and ingestion pipeline.
No LLM dependency; used by heuristic structure detection and chunking strategy selection.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chunking strategy (Literal type for strategy selection)
# ---------------------------------------------------------------------------
ChunkingStrategy = str  # Literal["clause_aware", "sentence"] at runtime


# ---------------------------------------------------------------------------
# Heuristic structure result (regex/pattern-based, no LLM)
# ---------------------------------------------------------------------------
class HeuristicStructureResult(BaseModel):
    """Result of cheap regex-based structure detection on document text."""

    has_structured_headings: bool = Field(
        False, description="Whether heading-like density exceeds threshold"
    )
    estimated_clause_count: int = Field(
        0, description="Number of heading-like segments detected"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Confidence in structure detection (0-1)"
    )


# ---------------------------------------------------------------------------
# Page info (parsing layer)
# ---------------------------------------------------------------------------
@dataclass
class PageInfo:
    """Single page from parsed document (text, page number, OCR flag)."""

    text: str
    page_number: int
    is_ocr: bool = False
    ocr_confidence: float = -1.0  # -1.0 = not measured; 0–100 from Tesseract


# ---------------------------------------------------------------------------
# Ingestion context (returned with chunks from ingest_document)
# ---------------------------------------------------------------------------
@dataclass
class IngestionContext:
    """Context produced by the ingestion pipeline for metadata and logging."""

    strategy: str  # ChunkingStrategy
    structure_detected: bool
    estimated_clause_count: int
    structure_confidence: float
    pages_processed: int
    uses_ocr: bool
    # Optional for backward compatibility
    ocr_chunks: int = 0


# ---------------------------------------------------------------------------
# Ingestion metadata (persisted per document/version)
# ---------------------------------------------------------------------------
class IngestionMetadata(BaseModel):
    """Record of a single ingestion run per document/version."""

    document_id: str = Field(..., description="Document ID (e.g. DOC-0001)")
    ingestion_version: int = Field(..., description="Aligned with document version in registry")
    chunking_strategy: str = Field(..., description="clause_aware | sentence")
    structure_detected: bool = Field(False, description="Whether heuristic detected structure")
    estimated_clause_count: int = Field(0, description="Heuristic clause count")
    embedding_model_version: str = Field(..., description="Embedding model name/version")
    created_at: datetime = Field(default_factory=datetime.now, description="When ingestion was run")
