"""
Pydantic models for document data structures.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document with metadata."""
    
    text: str = Field(..., description="The chunk text content")
    page_number: int = Field(..., description="Page number where chunk appears")
    chunk_index: int = Field(..., description="Index of chunk within document")
    document_id: str = Field(..., description="Unique identifier for the source document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentMetadata(BaseModel):
    """Metadata about an uploaded document."""
    
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension (pdf, docx)")
    upload_date: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    total_pages: int = Field(..., description="Total number of pages")
    total_chunks: int = Field(default=0, description="Number of chunks created")
    language: Optional[str] = Field(None, description="Detected language (ar, en)")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Uploaded filename")
    status: str = Field(..., description="Upload status (success, error)")
    message: str = Field(..., description="Status message")
    chunks_created: int = Field(..., description="Number of chunks created")
    pages_processed: int = Field(..., description="Number of pages processed")
    uses_ocr: Optional[bool] = Field(None, description="Whether OCR was used for text extraction")
    ocr_chunks: Optional[int] = Field(None, description="Number of chunks extracted using OCR")

