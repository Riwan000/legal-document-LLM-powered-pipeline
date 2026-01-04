"""
Document ingestion service.
Orchestrates file parsing, chunking, and prepares documents for embedding.
"""
import uuid
from pathlib import Path
from typing import List
from datetime import datetime

from backend.utils.file_parser import FileParser
from backend.utils.chunking import TextChunker
from backend.models.document import DocumentChunk, DocumentMetadata, DocumentUploadResponse
from backend.config import settings


class DocumentIngestionService:
    """Service for ingesting and processing documents."""
    
    def __init__(self):
        """Initialize the ingestion service."""
        self.parser = FileParser()
        self.chunker = TextChunker()
        
        # Ensure directories exist
        settings.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
        settings.TEMPLATES_PATH.mkdir(parents=True, exist_ok=True)
    
    def ingest_document(
        self,
        file_path: Path,
        document_type: str = "document"
    ) -> DocumentUploadResponse:
        """
        Ingest a document: parse, chunk, and prepare for embedding.
        
        Args:
            file_path: Path to uploaded file
            document_type: Type of document ("document" or "template")
            
        Returns:
            DocumentUploadResponse with ingestion results
        """
        try:
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Determine storage path based on type
            if document_type == "template":
                storage_dir = settings.TEMPLATES_PATH
            else:
                storage_dir = settings.DOCUMENTS_PATH
            
            # Copy file to storage (in a real system, you'd save the file)
            # For now, we'll just process it from the upload location
            filename = file_path.name
            
            # Parse file to extract text with page numbers
            pages = self.parser.parse_file(file_path)
            
            if not pages:
                raise ValueError("No text content found in document")
            
            # Count OCR pages
            ocr_pages = sum(1 for page_data in pages if len(page_data) == 3 and page_data[2])
            uses_ocr = ocr_pages > 0
            
            # Chunk all pages
            chunks = self.chunker.chunk_pages(pages, document_id)
            
            # Count OCR chunks
            ocr_chunks = sum(1 for chunk in chunks if chunk.metadata.get('is_ocr', False))
            
            # Create metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_type=file_path.suffix.lower(),
                upload_date=datetime.now(),
                total_pages=len(pages),
                total_chunks=len(chunks)
            )
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                status="success",
                message="Document ingested successfully",
                chunks_created=len(chunks),
                pages_processed=len(pages),
                uses_ocr=uses_ocr,
                ocr_chunks=ocr_chunks if uses_ocr else 0
            )
            
        except Exception as e:
            return DocumentUploadResponse(
                document_id=document_id if 'document_id' in locals() else "unknown",
                filename=file_path.name if 'file_path' in locals() else "unknown",
                status="error",
                message=f"Error ingesting document: {str(e)}",
                chunks_created=0,
                pages_processed=0
            )
    
    def get_chunks_from_document(
        self,
        file_path: Path,
        document_id: str,
        use_clause_aware_chunking: bool = False,
        clauses: List[dict] = None
    ) -> List[DocumentChunk]:
        """
        Parse and chunk a document, returning the chunks.
        Used when we need chunks for embedding.
        
        Args:
            file_path: Path to document file
            document_id: Unique document identifier
            use_clause_aware_chunking: If True, use clause-aware chunking to preserve clause boundaries
            clauses: Optional list of clause dicts (required if use_clause_aware_chunking is True)
            
        Returns:
            List of DocumentChunk objects
        """
        # Parse file
        pages = self.parser.parse_file(file_path)
        
        # If clause-aware chunking is requested, organize clauses by page
        if use_clause_aware_chunking and clauses:
            clauses_by_page = {}
            for clause in clauses:
                page_num = clause.get('page_number', 0)
                if page_num not in clauses_by_page:
                    clauses_by_page[page_num] = []
                clauses_by_page[page_num].append(clause)
            
            # Chunk pages with clause awareness
            chunks = self.chunker.chunk_pages(pages, document_id, clauses_by_page)
        else:
            # Standard chunking
            chunks = self.chunker.chunk_pages(pages, document_id)
        
        return chunks

