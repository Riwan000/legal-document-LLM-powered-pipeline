"""
FastAPI backend for Legal Document Intelligence MVP.
Provides REST API endpoints for all system features.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
import uuid
import shutil
import hashlib
from typing import Optional, List
from datetime import datetime
import os

from backend.config import settings
from backend.services.document_ingestion import DocumentIngestionService
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.rag_service import RAGService
from backend.services.clause_extraction import ClauseExtractionService
from backend.services.comparison_service import ComparisonService
from backend.services.summarization_service import SummarizationService
from backend.services.translation_service import TranslationService
from backend.services.document_registry import DocumentRegistry
from backend.models.document import DocumentUploadResponse, AnswerResponse
from backend.models.document_list import DocumentListResponse, DocumentListItem, DocumentRenameRequest, DocumentRenameResponse
from backend.models.clause import StructuredClause
from backend.services.clause_store import ClauseStore
from backend.services.clause_validator import ClauseValidator


# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Intelligence API",
    description="RAG-powered legal document analysis system",
    version="1.0.0"
)

# CORS middleware (for Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances (initialized on startup)
document_registry: Optional[DocumentRegistry] = None
ingestion_service: Optional[DocumentIngestionService] = None
embedding_service: Optional[EmbeddingService] = None
vector_store: Optional[VectorStore] = None
rag_service: Optional[RAGService] = None
clause_extractor: Optional[ClauseExtractionService] = None
clause_store: Optional[ClauseStore] = None
clause_validator: Optional[ClauseValidator] = None
comparison_service: Optional[ComparisonService] = None
summarization_service: Optional[SummarizationService] = None
translation_service: Optional[TranslationService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global document_registry, ingestion_service, embedding_service, vector_store
    global rag_service, clause_extractor, clause_store, clause_validator
    global comparison_service, summarization_service, translation_service
    
    print("Initializing services...")
    
    # Initialize document registry (must be first)
    document_registry = DocumentRegistry()
    print("Document registry initialized")
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    print(f"Embedding service initialized (dimension: {embedding_service.get_embedding_dimension()})")
    
    # Initialize vector store
    embedding_dim = embedding_service.get_embedding_dimension()
    vector_store = VectorStore(embedding_dim)
    
    # Try to load existing vector store
    vector_store_path = settings.VECTOR_STORE_PATH / settings.FAISS_INDEX_NAME
    if vector_store_path.exists():
        try:
            vector_store.load()
            print(f"Loaded existing vector store ({vector_store.get_stats()['total_vectors']} vectors)")
        except Exception as e:
            print(f"Could not load vector store: {e}. Starting fresh.")
    else:
        print("No existing vector store found. Starting fresh.")
    
    # Initialize ingestion service
    ingestion_service = DocumentIngestionService()
    
    # Initialize clause store (before RAG service)
    clause_store = ClauseStore()
    print(f"Clause store initialized at {clause_store.store_path}")
    
    # Initialize clause validator
    clause_validator = ClauseValidator()
    print("Clause validator initialized")
    
    # Initialize translation service (before RAG service, as RAG may need it)
    translation_service = TranslationService(None)  # Will be set after RAG is created
    
    # Initialize RAG service (with clause store and translation service)
    rag_service = RAGService(embedding_service, vector_store, clause_store, translation_service)
    
    # Update translation service with RAG service reference
    translation_service.rag_service = rag_service
    
    # Initialize clause extractor
    clause_extractor = ClauseExtractionService()
    
    # Initialize comparison service
    comparison_service = ComparisonService()
    
    # Initialize summarization service
    summarization_service = SummarizationService(rag_service)
    
    print("All services initialized successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Save vector store on application shutdown."""
    global vector_store
    if vector_store:
        try:
            # Run synchronous save in executor to avoid blocking and handle cancellation
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, vector_store.save)
            print("Vector store saved successfully.")
        except asyncio.CancelledError:
            # Gracefully handle cancellation during shutdown
            print("Shutdown cancelled, attempting quick save...")
            try:
                # Try synchronous save as fallback
                vector_store.save()
                print("Vector store saved successfully (fallback).")
            except Exception as e:
                print(f"Error saving vector store during cancellation: {e}")
        except Exception as e:
            print(f"Error saving vector store: {e}")


@app.get("/api/extract-clauses")
async def extract_clauses_help():
    """
    Helper endpoint for browsers/tools that accidentally GET this path.
    The actual clause extraction endpoint is POST /api/extract-clauses.
    """
    return {
        "message": "Use POST /api/extract-clauses with form-data: document_id (required), file_path (optional), use_structured (optional), validate (optional)."
    }


@app.post("/api/extract-clauses")
async def extract_clauses(
    document_id: str = Form(...),
    file_path: Optional[str] = Form(None),
    use_structured: bool = Form(True),
    include_telemetry: bool = Form(False)
):
    """
    Extract clauses from a document using deterministic structure-first extraction.
    
    Args:
        document_id: Document ID
        file_path: Optional file path (if not provided, searches in documents directory)
        use_structured: Use deterministic structured extraction (default: True, always used)
        include_telemetry: Include extraction telemetry in response (default: False)
        
    Returns:
        Extraction-only schema:
        {
            "schema_version": "1.0",
            "document_id": "...",
            "extraction_mode": "structure_first_verbatim",
            "clauses": [...],
            "telemetry": {...}  # Optional, if include_telemetry=True
        }
    """
    global clause_extractor
    
    if not clause_extractor:
        raise HTTPException(status_code=500, detail="Clause extractor not initialized")
    
    def _find_document_path(doc_id: str) -> Optional[str]:
        # Exact match first (current convention: <document_id>.<ext>)
        for ext in ['.pdf', '.docx', '.doc']:
            p = settings.DOCUMENTS_PATH / f"{doc_id}{ext}"
            if p.exists():
                return str(p)
        # Fallback: prefix match (helps if doc_id is truncated / legacy naming)
        for ext in ['.pdf', '.docx', '.doc']:
            matches = list(settings.DOCUMENTS_PATH.glob(f"{doc_id}*{ext}"))
            if matches:
                return str(matches[0])
        # Last resort: any extension starting with id
        any_matches = list(settings.DOCUMENTS_PATH.glob(f"{doc_id}*"))
        if any_matches:
            return str(any_matches[0])
        return None

    # Find file if path not provided
    if not file_path:
        # Search in documents directory
        file_path = _find_document_path(document_id)
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Document file not found")
    
    try:
        # Extract clauses with async timeout protection
        import asyncio
        import concurrent.futures
        
        # Run extraction in a thread pool with timeout
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        def extract_sync():
            if include_telemetry:
                return clause_extractor.extract_clauses_with_telemetry(file_path, document_id)
            else:
                clauses = clause_extractor.extract_clauses(file_path, document_id, use_structured=True)
                return {"clauses": clauses}
        
        # Run with 4 minute timeout (240 seconds) - slightly less than frontend timeout
        try:
            extraction_result = await asyncio.wait_for(
                loop.run_in_executor(executor, extract_sync),
                timeout=240.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Clause extraction timed out after 4 minutes. The document may be too large. Try processing smaller documents or contact support."
            )
        
        # Build response with extraction-only schema
        clauses = extraction_result.get("clauses", [])
        telemetry = extraction_result.get("telemetry")
        
        response = {
            "schema_version": "1.0",
            "document_id": document_id,
            "extraction_mode": "structure_first_verbatim",
            "clauses": clauses
        }
        
        # Add telemetry if requested
        if include_telemetry and telemetry:
            response["telemetry"] = telemetry
        
        return response
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting clauses: {str(e)}")


@app.get("/api/clauses/{document_id}")
async def get_document_clauses(document_id: str):
    """
    Get all clauses for a document.
    
    Args:
        document_id: Document identifier
        
    Returns:
        List of structured clauses
    """
    global clause_store
    
    if not clause_store:
        raise HTTPException(status_code=500, detail="Clause store not initialized")
    
    try:
        clauses = clause_store.get_clauses_by_document(document_id)
        return {
            "document_id": document_id,
            "clauses": [clause.model_dump() for clause in clauses],
            "count": len(clauses)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving clauses: {str(e)}")


@app.get("/api/clauses/single/{clause_id}")
async def get_clause(clause_id: str):
    """
    Get a specific clause by ID.
    
    Args:
        clause_id: Clause identifier
        
    Returns:
        Structured clause with evidence blocks
    """
    global clause_store
    
    if not clause_store:
        raise HTTPException(status_code=500, detail="Clause store not initialized")
    
    try:
        clause = clause_store.get_clause(clause_id)
        if not clause:
            raise HTTPException(status_code=404, detail="Clause not found")
        
        return clause.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving clause: {str(e)}")


@app.post("/api/clauses/query")
async def query_clauses(
    document_id: Optional[str] = Form(None),
    clause_type: Optional[str] = Form(None),
    authority_level: Optional[str] = Form(None),
    jurisdiction: Optional[str] = Form(None),
    can_override: Optional[bool] = Form(None)
):
    """
    Query clauses by filters.
    
    Args:
        document_id: Filter by document ID
        clause_type: Filter by clause type
        authority_level: Filter by authority level
        jurisdiction: Filter by jurisdiction
        can_override: Filter by override capability
        
    Returns:
        List of matching clauses
    """
    global clause_store
    
    if not clause_store:
        raise HTTPException(status_code=500, detail="Clause store not initialized")
    
    try:
        clauses = clause_store.query_clauses(
            document_id=document_id,
            clause_type=clause_type,
            authority_level=authority_level,
            jurisdiction=jurisdiction,
            can_override=can_override
        )
        
        return {
            "clauses": [clause.model_dump() for clause in clauses],
            "count": len(clauses),
            "filters": {
                "document_id": document_id,
                "clause_type": clause_type,
                "authority_level": authority_level,
                "jurisdiction": jurisdiction,
                "can_override": can_override
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying clauses: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    global vector_store, clause_store
    clause_stats = clause_store.get_stats() if clause_store else {}
    stats = vector_store.get_stats() if vector_store else {}
    
    return {
        "status": "healthy",
        "vector_store": stats,
        "ollama_url": settings.OLLAMA_BASE_URL,
        "ollama_model": settings.OLLAMA_MODEL
    }


@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form("document"),
    force_reingest: bool = Form(False),
    display_name: Optional[str] = Form(default=None)
):
    """
    Upload and ingest a document.
    Uses document registry for identity management with proper versioning.
    
    Args:
        file: Uploaded file (PDF or DOCX)
        document_type: Type of document ("document" or "template")
        force_reingest: If True, re-index even if document already exists
        display_name: Optional user-friendly display name (defaults to filename)
        
    Returns:
        DocumentUploadResponse with ingestion results (never includes document_hash)
    """
    global document_registry, ingestion_service, embedding_service, vector_store
    
    if not document_registry:
        raise HTTPException(status_code=500, detail="Document registry not initialized")
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: .pdf, .docx"
        )
    
    # Read file content to compute hash
    file_content = await file.read()
    file.file.seek(0)  # Reset file pointer for later use
    
    # Compute content hash (internal only, never exposed)
    document_hash = document_registry.compute_hash(file_content)
    
    # Check if document with this hash already exists
    existing_doc = document_registry.find_by_hash(document_hash, is_latest=True)
    
    # If same content exists and not forcing reingest, return existing document
    if existing_doc and not force_reingest:
        # Check if already indexed in vector store
        existing_chunks = vector_store.get_chunks_by_document(existing_doc['document_id']) if vector_store else []
        already_indexed = len(existing_chunks) > 0
        
        if already_indexed:
            return DocumentUploadResponse(
                document_id=existing_doc['document_id'],
                display_name=existing_doc['display_name'],
                original_filename=existing_doc['original_filename'],
                version=existing_doc['version'],
                status="success",
                message=f"Document already ingested. {len(existing_chunks)} chunks found in index.",
                chunks_created=len(existing_chunks),
                pages_processed=existing_doc.get('total_pages', len(set(c.get('page_number', 0) for c in existing_chunks))),
                uses_ocr=any(c.get('is_ocr', False) for c in existing_chunks),
                created_at=datetime.fromisoformat(existing_doc['created_at']) if isinstance(existing_doc['created_at'], str) else existing_doc['created_at'],
                updated_at=datetime.fromisoformat(existing_doc['updated_at']) if isinstance(existing_doc['updated_at'], str) else existing_doc['updated_at']
            )
    
    # Determine storage directory
    if document_type == "template":
        storage_dir = settings.TEMPLATES_PATH
    else:
        storage_dir = settings.DOCUMENTS_PATH
    
    # Register document in registry (creates new version if hash differs)
    doc_record = document_registry.register_document(
        document_hash=document_hash,
        original_filename=file.filename,
        document_type=document_type,
        display_name=display_name
    )
    
    document_id = doc_record['document_id']
    version = doc_record['version']
    
    # Save uploaded file
    file_path = storage_dir / f"{document_id}_v{version}{file_ext}"
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Update registry with file path
    with document_registry._get_connection() as conn:
        conn.execute(
            "UPDATE documents SET file_path = ? WHERE document_id = ? AND version = ?",
            [str(file_path), document_id, version]
        )
    
    # Ingest document
    try:
        # Get chunks for embedding
        chunks = ingestion_service.get_chunks_from_document(file_path, document_id)
        
        if not chunks:
            raise ValueError("No chunks created from document")
        
        # If this is a new version, remove old chunks for this document_id
        if version > 1 and vector_store:
            old_chunks = vector_store.get_chunks_by_document(document_id)
            if old_chunks:
                # Delete old version chunks (by document_hash if available, otherwise by document_id)
                # For now, we'll delete all chunks for this document_id and re-add
                vector_store.delete_document(document_id)
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_service.embed_batch(texts)
        
        # Add to vector store with display_name and document_hash (internal)
        vector_store.add_chunks(
            embeddings, 
            chunks,
            display_name=doc_record['display_name'],
            document_hash=document_hash
        )
        
        # Save vector store
        vector_store.save()
        
        # Count OCR chunks
        ocr_chunks = sum(1 for chunk in chunks if chunk.metadata.get('is_ocr', False))
        uses_ocr = ocr_chunks > 0
        
        # Update registry with page/chunk counts
        pages = ingestion_service.parser.parse_file(file_path)
        total_pages = len(pages) if pages else 0
        
        with document_registry._get_connection() as conn:
            conn.execute("""
                UPDATE documents 
                SET total_pages = ?, total_chunks = ?, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ? AND version = ?
            """, [total_pages, len(chunks), document_id, version])
        
        # Build response (never include document_hash)
        return DocumentUploadResponse(
            document_id=document_id,
            display_name=doc_record['display_name'],
            original_filename=doc_record['original_filename'],
            version=version,
            status="success",
            message=f"Document ingested and indexed successfully. {len(chunks)} chunks created." + 
                   (f" ({ocr_chunks} from OCR)." if uses_ocr else "."),
            chunks_created=len(chunks),
            pages_processed=total_pages,
            uses_ocr=uses_ocr,
            ocr_chunks=ocr_chunks if uses_ocr else 0,
            created_at=datetime.fromisoformat(doc_record['created_at']) if isinstance(doc_record['created_at'], str) else doc_record['created_at'],
            updated_at=datetime.now()
        )
        
    except Exception as e:
        # Clean up file and registry entry on error
        if file_path.exists():
            file_path.unlink()
        document_registry.delete_document(document_id, version)
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")


@app.post("/api/search")
async def search_documents(
    query: str = Form(...),
    top_k: Optional[int] = Form(None),
    document_id: Optional[str] = Form(None),
    generate_response: bool = Form(True),
    response_language: Optional[str] = Form(None)
):
    """
    Search documents using RAG.
    
    Args:
        query: Search query
        top_k: Number of results (defaults to config)
        document_id: Optional filter by document ID
        generate_response: Whether to generate LLM response
        response_language: Optional response language ('ar', 'en', or None for auto)
        
    Returns:
        Search results with answer and sources
    """
    global rag_service
    
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        # Use multilingual query if language is specified
        if response_language:
            result = rag_service.query_multilingual(
                query=query,
                response_language=response_language,
                top_k=top_k,
                document_id_filter=document_id
            )
        else:
            result = rag_service.query(
                query=query,
                top_k=top_k,
                document_id_filter=document_id,
                generate_response=generate_response
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.post("/api/compare")
async def compare_contracts(
    contract_id: str = Form(...),
    template_id: str = Form(...),
    contract_path: Optional[str] = Form(None),
    template_path: Optional[str] = Form(None)
):
    """
    Compare a contract against a template.
    
    Args:
        contract_id: Contract document ID
        template_id: Template document ID
        contract_path: Optional contract file path
        template_path: Optional template file path
        
    Returns:
        Comparison results
    """
    global comparison_service
    
    if not comparison_service:
        raise HTTPException(status_code=500, detail="Comparison service not initialized")
    
    # Find files if paths not provided
    if not contract_path:
        for ext in ['.pdf', '.docx', '.doc']:
            potential_path = settings.DOCUMENTS_PATH / f"{contract_id}{ext}"
            if potential_path.exists():
                contract_path = str(potential_path)
                break
    
    if not template_path:
        for ext in ['.pdf', '.docx', '.doc']:
            potential_path = settings.TEMPLATES_PATH / f"{template_id}{ext}"
            if potential_path.exists():
                template_path = str(potential_path)
                break
    
    if not contract_path or not template_path:
        raise HTTPException(status_code=404, detail="Contract or template file not found")
    
    try:
        comparison = comparison_service.compare_contracts(
            contract_path=contract_path,
            template_path=template_path,
            contract_id=contract_id,
            template_id=template_id
        )
        
        # Generate report
        report = comparison_service.generate_comparison_report(comparison, format='markdown')
        
        return {
            "comparison": comparison,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing contracts: {str(e)}")


@app.post("/api/summarize")
async def summarize_case_file(
    document_id: str = Form(...),
    top_k: int = Form(10),
    include_report: bool = Form(False)
):
    """
    Generate summary of a case file (PRD-compliant).
    
    Args:
        document_id: Document ID
        top_k: Number of chunks to retrieve (legacy, uses config defaults)
        include_report: If True, include markdown report (for backward compatibility)
        
    Returns:
        Case file summary with strict JSON schema
    """
    global summarization_service
    
    if not summarization_service:
        raise HTTPException(status_code=500, detail="Summarization service not initialized")
    
    try:
        result = summarization_service.summarize_case_file(
            document_id=document_id,
            top_k=top_k
        )
        
        # Check for errors
        if "error" in result:
            error = result["error"]
            status_code = 422 if error.get("code") in ["CASE_SPINE_FAILED", "INSUFFICIENT_CONTEXT"] else 400
            raise HTTPException(
                status_code=status_code,
                detail=error
            )
        
        # Generate report if requested
        report = None
        if include_report:
            report = summarization_service.generate_summary_report(result, format='markdown')
        
        response = {"summary": result}
        if report:
            response["report"] = report
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing case file: {str(e)}")


@app.post("/api/summarize/stream")
async def summarize_case_file_stream(
    document_id: str = Form(...)
):
    """
    Stream case summary via Server-Sent Events (SSE).

    Events include: case_spine, executive_summary_item, timeline_event, claimant_argument_item,
    defendant_argument_item, open_issue_item, citations, done, error.
    """
    global summarization_service

    if not summarization_service:
        raise HTTPException(status_code=500, detail="Summarization service not initialized")

    def event_generator():
        yield from summarization_service.summarize_case_file_stream(document_id=document_id)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.post("/api/search-bilingual")
async def search_bilingual(
    query: str = Form(...),
    response_language: Optional[str] = Form(None),
    top_k: Optional[int] = Form(None),
    document_id: Optional[str] = Form(None)
):
    """
    Bilingual search with automatic language detection.
    
    Args:
        query: Search query (Arabic or English)
        response_language: Desired response language ('ar', 'en', or None for auto)
        top_k: Number of results
        document_id: Optional filter by document ID
        
    Returns:
        Search results with language metadata
    """
    global translation_service
    
    if not translation_service:
        raise HTTPException(status_code=500, detail="Translation service not initialized")
    
    try:
        result = translation_service.query_bilingual(
            query=query,
            response_language=response_language,
            top_k=top_k,
            document_id_filter=document_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in bilingual search: {str(e)}")


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(include_versions: bool = True):
    """
    List all ingested documents.
    
    Args:
        include_versions: If True, include all versions; if False, only latest
        
    Returns:
        DocumentListResponse with list of documents (never includes document_hash)
    """
    global document_registry, vector_store
    
    if not document_registry:
        return DocumentListResponse(documents=[])
    
    # Get documents from registry
    doc_records = document_registry.list_documents(include_versions=include_versions)
    
    # Build document list items
    documents = []
    for doc_record in doc_records:
        # Get chunk count from vector store if available
        total_chunks = doc_record.get('total_chunks')
        if not total_chunks and vector_store:
            chunks = vector_store.get_chunks_by_document(doc_record['document_id'])
            total_chunks = len(chunks)
        
        # Parse datetime strings if needed
        created_at = doc_record['created_at']
        updated_at = doc_record['updated_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        
        documents.append(DocumentListItem(
            document_id=doc_record['document_id'],
            display_name=doc_record['display_name'],
            original_filename=doc_record['original_filename'],
            version=doc_record['version'],
            is_latest=bool(doc_record['is_latest']),
            document_type=doc_record['document_type'],
            created_at=created_at,
            updated_at=updated_at,
            total_chunks=total_chunks,
            total_pages=doc_record.get('total_pages')
        ))
    
    return DocumentListResponse(documents=documents)


@app.put("/api/documents/{document_id}/rename", response_model=DocumentRenameResponse)
async def rename_document(
    document_id: str,
    request: DocumentRenameRequest
):
    """
    Rename a document (update display_name).
    
    Args:
        document_id: Document ID (DOC-####)
        request: Rename request with new display_name
        
    Returns:
        DocumentRenameResponse with updated display_name
    """
    global document_registry
    
    if not document_registry:
        raise HTTPException(status_code=500, detail="Document registry not initialized")
    
    # Validate display_name
    if not request.display_name or not request.display_name.strip():
        raise HTTPException(status_code=400, detail="Display name cannot be empty")
    
    # Update display name
    success = document_registry.update_display_name(document_id, request.display_name.strip())
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    # Get updated document
    doc = document_registry.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    return DocumentRenameResponse(
        document_id=document_id,
        display_name=doc['display_name'],
        success=True
    )


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    global vector_store
    
    if not vector_store:
        return {"error": "Vector store not initialized"}
    
    stats = vector_store.get_stats()
    return {
        "vector_store": stats,
        "config": {
            "ollama_url": settings.OLLAMA_BASE_URL,
            "ollama_model": settings.OLLAMA_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "top_k_results": settings.TOP_K_RESULTS
        }
    }


if __name__ == "__main__":
    import uvicorn
    # Increase timeout for long-running operations like clause extraction
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        timeout_keep_alive=300  # 5 minutes timeout
    )

