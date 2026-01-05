"""
FastAPI backend for Legal Document Intelligence MVP.
Provides REST API endpoints for all system features.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
import shutil
import hashlib
from typing import Optional, List
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
from backend.models.document import DocumentUploadResponse, AnswerResponse
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
    global ingestion_service, embedding_service, vector_store
    global rag_service, clause_extractor, clause_store, clause_validator
    global comparison_service, summarization_service, translation_service
    
    print("Initializing services...")
    
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
    
    # Initialize RAG service (with clause store)
    rag_service = RAGService(embedding_service, vector_store, clause_store)
    
    # Initialize clause extractor
    clause_extractor = ClauseExtractionService()
    
    # Initialize comparison service
    comparison_service = ComparisonService()
    
    # Initialize summarization service
    summarization_service = SummarizationService(rag_service)
    
    # Initialize translation service
    translation_service = TranslationService(rag_service)
    
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


@app.post("/api/extract-clauses")
async def extract_clauses(
    document_id: str = Form(...),
    file_path: Optional[str] = Form(None),
    use_structured: bool = Form(True),
    validate: bool = Form(True)
):
    """
    Extract structured clauses from a contract and store them.
    
    Args:
        document_id: Document ID
        file_path: Optional file path (if not provided, searches in documents directory)
        use_structured: Use structured extraction engine (default: True)
        validate: Validate clauses before storing (default: True)
        
    Returns:
        Dict with extracted clauses, validation results, and storage status
    """
    global clause_extractor, clause_store, clause_validator
    
    if not clause_extractor:
        raise HTTPException(status_code=500, detail="Clause extractor not initialized")
    if not clause_store:
        raise HTTPException(status_code=500, detail="Clause store not initialized")
    if not clause_validator:
        raise HTTPException(status_code=500, detail="Clause validator not initialized")
    
    # Find file if path not provided
    if not file_path:
        # Search in documents directory
        for ext in ['.pdf', '.docx', '.doc']:
            potential_path = settings.DOCUMENTS_PATH / f"{document_id}{ext}"
            if potential_path.exists():
                file_path = str(potential_path)
                break
        
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
            if use_structured:
                return clause_extractor.structured_extractor.extract_structured_clauses(
                    file_path, document_id
                )
            else:
                return clause_extractor.extract_clauses(file_path, document_id, use_structured=False)
        
        # Run with 4 minute timeout (240 seconds) - slightly less than frontend timeout
        try:
            structured_clauses = await asyncio.wait_for(
                loop.run_in_executor(executor, extract_sync),
                timeout=240.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Clause extraction timed out after 4 minutes. The document may be too large. Try processing smaller documents or contact support."
            )
        
        if use_structured:
            
            # Validate if requested
            validation_results = None
            if validate:
                validation_results = clause_validator.validate_clauses_batch(structured_clauses)
                # Filter out invalid clauses if any
                valid_clauses = [
                    clause for clause, result in zip(structured_clauses, validation_results['results'])
                    if result['valid']
                ]
            else:
                valid_clauses = structured_clauses
            
            # Store clauses
            clause_store.save_clauses(document_id, valid_clauses)
            
            # Convert to dict for response
            clauses_dict = [clause.model_dump() for clause in valid_clauses]
            
            return {
                "clauses": clauses_dict,
                "count": len(valid_clauses),
                "total_clauses": len(valid_clauses),  # For frontend compatibility
                "total_extracted": len(structured_clauses),
                "validation": validation_results if validation_results else None,
                "stored": True
            }
        else:
            # Legacy extraction - convert to dict format
            clauses_dict = []
            for clause in structured_clauses:
                if isinstance(clause, dict):
                    clauses_dict.append(clause)
                else:
                    clauses_dict.append(clause.model_dump() if hasattr(clause, 'model_dump') else clause)
            
            return {
                "clauses": clauses_dict,
                "count": len(clauses_dict),
                "total_clauses": len(clauses_dict),  # For frontend compatibility
                "stored": False
            }
            
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
    force_reingest: bool = Form(False)
):
    """
    Upload and ingest a document.
    Uses content-hash document IDs for determinism.
    
    Args:
        file: Uploaded file (PDF or DOCX)
        document_type: Type of document ("document" or "template")
        force_reingest: If True, re-index even if document already exists
        
    Returns:
        DocumentUploadResponse with ingestion results
    """
    global ingestion_service, embedding_service, vector_store
    
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
    
    # Compute content hash as document ID
    document_id = hashlib.sha256(file_content).hexdigest()
    
    # Determine storage directory
    if document_type == "template":
        storage_dir = settings.TEMPLATES_PATH
    else:
        storage_dir = settings.DOCUMENTS_PATH
    
    # Check if document already exists
    file_path = storage_dir / f"{document_id}{file_ext}"
    already_exists = file_path.exists()
    
    # Check if already indexed in vector store
    already_indexed = False
    if vector_store and not force_reingest:
        existing_chunks = vector_store.get_chunks_by_document(document_id)
        already_indexed = len(existing_chunks) > 0
    
    # If already indexed and not forcing reingest, return early
    if already_indexed and not force_reingest:
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="success",
            message=f"Document already ingested. {len(existing_chunks)} chunks found in index.",
            chunks_created=len(existing_chunks),
            pages_processed=len(set(c.get('page_number', 0) for c in existing_chunks)),
            uses_ocr=any(c.get('is_ocr', False) for c in existing_chunks)
        )
    
    # Save uploaded file (only if not already exists or forcing)
    if not already_exists or force_reingest:
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Ingest document
    try:
        result = ingestion_service.ingest_document(file_path, document_type, document_id=document_id)
        result.document_id = document_id  # Use our computed hash ID
        
        # Get chunks for embedding
        chunks = ingestion_service.get_chunks_from_document(file_path, document_id)
        
        if chunks:
            # If forcing reingest, remove old chunks first
            if force_reingest and already_indexed:
                vector_store.delete_document(document_id)
            
            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            embeddings = embedding_service.embed_batch(texts)
            
            # Add to vector store
            vector_store.add_chunks(embeddings, chunks)
            
            # Save vector store
            vector_store.save()
            
            # Count OCR chunks
            ocr_chunks = sum(1 for chunk in chunks if chunk.metadata.get('is_ocr', False))
            uses_ocr = ocr_chunks > 0
            
            result.chunks_created = len(chunks)
            result.uses_ocr = uses_ocr
            result.ocr_chunks = ocr_chunks if uses_ocr else 0
            
            if uses_ocr:
                result.message = f"Document ingested and indexed successfully. {len(chunks)} chunks created ({ocr_chunks} from OCR)."
            else:
                result.message = f"Document ingested and indexed successfully. {len(chunks)} chunks created."
        
        return result
        
    except Exception as e:
        # Clean up file on error (only if we created it)
        if not already_exists and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")


@app.post("/api/search")
async def search_documents(
    query: str = Form(...),
    top_k: Optional[int] = Form(None),
    document_id: Optional[str] = Form(None),
    generate_response: bool = Form(True)
):
    """
    Search documents using RAG.
    
    Args:
        query: Search query
        top_k: Number of results (defaults to config)
        document_id: Optional filter by document ID
        generate_response: Whether to generate LLM response
        
    Returns:
        Search results with answer and sources
    """
    global rag_service
    
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
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


@app.get("/api/documents")
async def list_documents():
    """List all ingested documents."""
    global vector_store
    
    if not vector_store:
        return {"documents": []}
    
    # Get unique document IDs from vector store
    stats = vector_store.get_stats()
    unique_docs = set()
    
    for metadata in vector_store.metadata:
        unique_docs.add(metadata['document_id'])
    
    documents = []
    for doc_id in unique_docs:
        chunks = vector_store.get_chunks_by_document(doc_id)
        if chunks:
            # Get first chunk to extract document info
            first_chunk = chunks[0]
            documents.append({
                "document_id": doc_id,
                "total_chunks": len(chunks),
                "pages": list(set(c.get('page_number', 0) for c in chunks))
            })
    
    return {"documents": documents}


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

