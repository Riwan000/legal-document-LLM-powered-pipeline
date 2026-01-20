"""
Models for document listing and management.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentListItem(BaseModel):
    """Item in document list response."""
    
    document_id: str = Field(..., description="Unique document identifier (DOC-####)")
    display_name: str = Field(..., description="User-friendly display name")
    original_filename: str = Field(..., description="Original filename")
    version: int = Field(..., description="Version number")
    is_latest: bool = Field(..., description="Whether this is the latest version")
    document_type: str = Field(..., description="Type: 'document' or 'template'")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    total_chunks: Optional[int] = Field(None, description="Number of chunks")
    total_pages: Optional[int] = Field(None, description="Number of pages")


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    
    documents: List[DocumentListItem] = Field(..., description="List of documents")


class DocumentRenameRequest(BaseModel):
    """Request model for renaming a document."""
    
    display_name: str = Field(..., description="New display name", min_length=1, max_length=200)


class DocumentRenameResponse(BaseModel):
    """Response model for document rename."""
    
    document_id: str = Field(..., description="Document ID")
    display_name: str = Field(..., description="Updated display name")
    success: bool = Field(..., description="Whether rename was successful")
