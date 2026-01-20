"""
Pydantic models for document data structures.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class LegalHierarchyLevel(str, Enum):
    """Legal hierarchy levels for clause classification."""
    LAW = "law"  # External law, statute, regulation
    CONTRACT = "contract"  # Contract clause
    POLICY = "policy"  # Employer/policy discretion


class ClauseMetadata(BaseModel):
    """Enhanced metadata for clauses with legal hierarchy information."""
    chunk_id: Optional[str] = Field(None, description="Unique identifier for the clause chunk")
    type: str = Field(..., description="Clause type (e.g., Payment Terms, Termination)")
    topics: List[str] = Field(default_factory=list, description="Topics/keywords associated with clause")
    language: Optional[str] = Field(None, description="Language of clause (ar, en)")
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction (e.g., Saudi Arabia)")
    hierarchy_level: LegalHierarchyLevel = Field(LegalHierarchyLevel.CONTRACT, description="Legal hierarchy level")
    legal_supremacy: bool = Field(False, description="Whether this clause has legal supremacy or is overridden by law")


class QueryClassification(BaseModel):
    """Classification of user query for rule-based processing."""
    query_type: str = Field(..., description="Type of query (termination, legality, benefits, compliance, etc.)")
    requires_legal_hierarchy: bool = Field(False, description="Whether query requires legal hierarchy analysis")
    scope_topics: List[str] = Field(default_factory=list, description="Topics extracted from query")
    is_legal_query: bool = Field(False, description="Whether this is a legal/compliance query requiring citations")


class AnswerResponse(BaseModel):
    """Structured response model for legal queries."""
    status: str = Field(..., description="Status: explicitly_stated, governed_by_law, not_specified, refused")
    answer: Optional[str] = Field(None, description="Direct answer to the query")
    citation: Optional[str] = Field(None, description="Citation reference (e.g., Clause 11, Page 4)")
    confidence: str = Field(..., description="Confidence level: high, medium, low")
    refusal_reason: Optional[str] = Field(None, description="Reason for refusal if status is refused")
    hierarchy_analysis: Optional[Dict[str, Any]] = Field(None, description="Legal hierarchy analysis results")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source chunks with citations")
    query: str = Field(..., description="Original query")


# Alias for backward compatibility
LegalAnswerResponse = AnswerResponse


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document with metadata."""
    
    text: str = Field(..., description="The chunk text content")
    page_number: int = Field(..., description="Page number where chunk appears")
    chunk_index: int = Field(..., description="Index of chunk within document")
    document_id: str = Field(..., description="Unique identifier for the source document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    clause_id: Optional[str] = Field(None, description="Unique identifier for the clause (if chunk represents a clause)")


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
    
    document_id: str = Field(..., description="Unique document identifier (DOC-####)")
    display_name: str = Field(..., description="User-friendly display name")
    original_filename: str = Field(..., description="Original filename from upload")
    version: int = Field(..., description="Document version number")
    status: str = Field(..., description="Upload status (success, error)")
    message: str = Field(..., description="Status message")
    chunks_created: int = Field(..., description="Number of chunks created")
    pages_processed: int = Field(..., description="Number of pages processed")
    uses_ocr: Optional[bool] = Field(None, description="Whether OCR was used for text extraction")
    ocr_chunks: Optional[int] = Field(None, description="Number of chunks extracted using OCR")
    created_at: Optional[datetime] = Field(None, description="Document creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Document update timestamp")
    
    # Backward compatibility: alias filename to original_filename
    @property
    def filename(self) -> str:
        """Backward compatibility property."""
        return self.original_filename

