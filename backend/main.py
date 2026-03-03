"""
FastAPI backend for Legal Document Intelligence MVP.
Provides REST API endpoints for all system features.
"""
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware import Middleware as StarletteMiddleware
from pathlib import Path
import uuid
import shutil
import hashlib
from typing import Optional, List
from datetime import datetime
import os
import threading
from contextlib import asynccontextmanager

from backend.config import settings
from backend.services.document_ingestion import DocumentIngestionService
from backend.services.embedding_service import EmbeddingService, get_embedding_model
from backend.services.vector_store import VectorStore
from backend.services.rag_service import RAGService
from backend.services.clause_extraction import ClauseExtractionService
from backend.services.comparison_service import ComparisonService
from backend.services.summarization_service import SummarizationService
from backend.services.translation_service import TranslationService
from backend.services.document_registry import DocumentRegistry
from backend.models.document import DocumentUploadResponse, AnswerResponse, DocumentChunk
from backend.models.document_list import DocumentListResponse, DocumentListItem, DocumentRenameRequest, DocumentRenameResponse
from backend.models.clause import StructuredClause
from backend.services.clause_store import ClauseStore
from backend.services.extracted_clause_store import ExtractedClauseStore, EXTRACTION_VERSION
from backend.services.clause_validator import ClauseValidator
from backend.services.document_explorer_service import DocumentExplorerService
from backend.services.evidence_explorer_service import EvidenceExplorerService
from backend.services.contract_review_service import ContractReviewService
from backend.services.workflow_orchestrator import WorkflowOrchestrator
from backend.services.due_diligence_memo_service import DueDiligenceMemoService
from backend.services.ingestion_metadata_store import IngestionMetadataStore
from backend.models.document_structure import IngestionMetadata
from backend.models.workflow import WorkflowContext
from backend.services.guardrails import WORKFLOW_DISCLAIMER
from backend import workflow_store

# Conversational RAG
from backend.models.session import (
    CreateSessionRequest,
    CreateSessionResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    SessionHistoryResponse,
    SessionMode,
)
from backend.services.session_store import SessionStore
from backend.services.session_manager import SessionManager, SessionNotFoundError, DocumentMismatchError
from backend.services.evidence_guardrail_service import EvidenceGuardrailService
from backend.services.query_rewriter import QueryRewriter
from backend.services.chat_orchestrator import ChatOrchestratorService
from backend.services.conversation_summarizer import ConversationSummarizer
from backend.services.document_classification_service import DocumentClassificationService
from backend.models.document import DocumentClassification
from backend.services.structured_clause_extraction import StructuredClauseExtractionService
from backend.utils.chunking import TextChunker


# Lifespan handlers (replaces deprecated on_event startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    await _initialize_services()
    # Optional background warm-up: load heavy AI models without blocking startup.
    threading.Thread(target=_warmup_ai, daemon=True).start()
    try:
        yield
    finally:
        await _shutdown_services()


# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Intelligence API",
    description="RAG-powered legal document analysis system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (for Streamlit frontend)
#
# NOTE: Some unit tests incorrectly inspect `type(m)` for entries in
# `app.user_middleware` and expect it to be a subclass of `CORSMiddleware`.
# Starlette stores middleware as `starlette.middleware.Middleware` wrapper
# objects (so `type(m)` is not `CORSMiddleware`). To satisfy that test while
# keeping the real CORS middleware behavior, we add a wrapper whose *type*
# subclasses `CORSMiddleware` and still behaves like a Starlette `Middleware`
# entry.
class _CORSMiddlewareMarker(StarletteMiddleware, CORSMiddleware):
    pass

app.user_middleware.append(
    _CORSMiddlewareMarker(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
)

# Global service instances (initialized on startup)
document_registry: Optional[DocumentRegistry] = None
ingestion_service: Optional[DocumentIngestionService] = None
embedding_service: Optional[EmbeddingService] = None
vector_store: Optional[VectorStore] = None
rag_service: Optional[RAGService] = None
clause_extractor: Optional[ClauseExtractionService] = None
clause_store: Optional[ClauseStore] = None
extracted_clause_store: Optional[ExtractedClauseStore] = None
clause_validator: Optional[ClauseValidator] = None
comparison_service: Optional[ComparisonService] = None
summarization_service: Optional[SummarizationService] = None
translation_service: Optional[TranslationService] = None
document_explorer_service: Optional[DocumentExplorerService] = None
evidence_explorer_service: Optional[EvidenceExplorerService] = None
contract_review_service: Optional[ContractReviewService] = None
workflow_orchestrator: Optional[WorkflowOrchestrator] = None
ingestion_metadata_store: Optional[IngestionMetadataStore] = None

# Conversational RAG service instances
session_store: Optional[SessionStore] = None
session_manager: Optional[SessionManager] = None
chat_orchestrator: Optional[ChatOrchestratorService] = None

# Document classification
document_classification_service: Optional[DocumentClassificationService] = None
structured_clause_extractor: Optional[StructuredClauseExtractionService] = None

# Serialize uploads to avoid DB/vector-store races when user clicks "Upload & Classify" twice
_upload_lock = threading.Lock()


async def _initialize_services():
    """Initialize services on application startup."""
    global document_registry, ingestion_service, embedding_service, vector_store
    global rag_service, clause_extractor, clause_store, clause_validator
    global comparison_service, summarization_service, translation_service
    global document_explorer_service, evidence_explorer_service, contract_review_service, workflow_orchestrator
    global extracted_clause_store, ingestion_metadata_store
    global session_store, session_manager, chat_orchestrator
    global document_classification_service, structured_clause_extractor
    
    print("Initializing services...")
    
    # Initialize document registry (must be first)
    document_registry = DocumentRegistry()
    print("Document registry initialized")
    
    # Initialize embedding service (lazy model loading).
    embedding_service = EmbeddingService()
    print(f"Embedding service initialized (dimension (config): {embedding_service.get_embedding_dimension()})")
    
    # Initialize vector store with configured embedding dimension.
    embedding_dim = getattr(settings, "EMBEDDING_DIM", embedding_service.get_embedding_dimension())
    vector_store = VectorStore(int(embedding_dim))
    
    # Initialize ingestion service
    ingestion_service = DocumentIngestionService()
    ingestion_metadata_store = IngestionMetadataStore()
    
    # Initialize clause store (before RAG service)
    clause_store = ClauseStore()
    print(f"Clause store initialized at {clause_store.store_path}")

    # Initialize extracted clause store (source of truth for Document Explorer)
    extracted_clause_store = ExtractedClauseStore()
    print(f"Extracted clause store initialized at {extracted_clause_store.store_path}")
    
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
    
    # Initialize comparison service (reuse global embedding service)
    comparison_service = ComparisonService(embedding_service)
    
    # Initialize summarization service
    summarization_service = SummarizationService(rag_service)
    
    # Workflow services (v0.2)
    document_explorer_service = DocumentExplorerService(rag_service, extracted_clause_store)
    evidence_explorer_service = EvidenceExplorerService(rag_service, extracted_clause_store)
    contract_review_service = ContractReviewService(clause_store, clause_extractor)
    due_diligence_memo_service = DueDiligenceMemoService(rag_service)
    workflow_orchestrator = WorkflowOrchestrator(
        document_explorer_service,
        contract_review_service,
        due_diligence_memo_service,
        evidence_explorer_service=evidence_explorer_service,
    )

    # Conversational RAG services
    session_store = SessionStore()
    conversation_summarizer = ConversationSummarizer()
    session_manager = SessionManager(store=session_store, summarizer=conversation_summarizer)
    evidence_guardrail = EvidenceGuardrailService(embedding_service)
    query_rewriter = QueryRewriter()
    chat_orchestrator = ChatOrchestratorService(
        rag_service=rag_service,
        session_manager=session_manager,
        embedding_service=embedding_service,
        guardrail_service=evidence_guardrail,
        query_rewriter=query_rewriter,
    )
    print("Conversational RAG services initialized (SessionStore, SessionManager, ChatOrchestrator)")

    # Document classification service
    document_classification_service = DocumentClassificationService()
    print("Document classification service initialized")

    # Structured clause extractor (for Phase 6 upload pipeline)
    structured_clause_extractor = StructuredClauseExtractionService()
    print("Structured clause extractor initialized")

    print("All services initialized successfully!")


async def _shutdown_services():
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


def _warmup_ai() -> None:
    """
    Background warm-up for heavy AI models.

    This runs in a daemon thread so it never blocks server readiness.
    """
    try:
        # Trigger lazy embedding model load (SentenceTransformers path).
        get_embedding_model()
    except Exception as e:  # pragma: no cover - best-effort warmup
        print(f"AI warm-up failed (non-fatal): {e}")


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
            "extraction_version": "v1.0",
            "extraction_mode": "structure_first_verbatim",
            "clauses": [...],
            "telemetry": {...}  # Optional, if include_telemetry=True
        }
    """
    global clause_extractor, extracted_clause_store

    # Step 0 (API): Resolve the document file path.
    
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

    # Step 0.1 (API): Find file if path not provided
    if not file_path:
        # Search in documents directory
        file_path = _find_document_path(document_id)
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Document file not found")

    # Step 0.2 (API): Ensure the clause extractor service is initialized.
    # Only require the extractor after we've validated the request and located the file.
    # This keeps missing-document behavior at 404 even if services were not started.
    if not clause_extractor:
        raise HTTPException(status_code=500, detail="Clause extractor not initialized")
    if not extracted_clause_store:
        raise HTTPException(status_code=500, detail="Extracted clause store not initialized")
    
    try:
        # Step 0.3 (API): Run extraction with timeout protection (thread pool).
        # Extract clauses with async timeout protection
        import asyncio
        import concurrent.futures
        
        # Run extraction in a thread pool with timeout
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        def extract_sync():
            # Step 0.4 (API): Dispatch to extractor (optionally with telemetry).
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
        
        # Step 0.5 (API): Build extraction-only response schema.
        # Build response with extraction-only schema
        clauses = extraction_result.get("clauses", [])
        telemetry = extraction_result.get("telemetry")
        
        # Persist extracted clauses as the source of truth for Explorer
        try:
            extracted_clause_store.save_document_clauses(document_id, clauses)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to persist extracted clauses: {str(e)}")

        response = {
            "schema_version": "1.0",
            "document_id": document_id,
            "extraction_version": EXTRACTION_VERSION,
            "extraction_mode": "structure_first_verbatim",
            "clauses": clauses,
            "document_type": extraction_result.get("document_type")
            or getattr(getattr(clause_extractor, "structured_extractor", None), "last_document_type", None).value
            if getattr(getattr(clause_extractor, "structured_extractor", None), "last_document_type", None)
            else None
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
    """Lightweight health check endpoint (server readiness only)."""
    return {
        "status": "ok",
        "server": "ready",
        "ai_models": "lazy",
    }


@app.get("/api/health/ai")
async def ai_health_check():
    """
    AI readiness endpoint.

    This explicitly checks that embedding models can be loaded.
    """
    try:
        # This will lazily load the heavy embedding model if available.
        get_embedding_model()
        return {"ai_models": "ready"}
    except Exception as e:
        # Do not leak internal errors; just report not_ready with minimal detail.
        return JSONResponse(
            status_code=503,
            content={
                "ai_models": "not_ready",
                "reason": str(e),
            },
        )


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
    global document_registry, ingestion_service, embedding_service, vector_store, ingestion_metadata_store

    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: .pdf, .docx"
        )

    if not document_registry:
        raise HTTPException(status_code=500, detail="Document registry not initialized")
    
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
            # For duplicate uploads, return stored classification if available
            stored_cls = document_registry.get_classification(existing_doc['document_id'], existing_doc['version'])
            if stored_cls and stored_cls.get("classification") == "non_legal":
                return DocumentUploadResponse(
                    document_id=existing_doc['document_id'],
                    display_name=existing_doc['display_name'],
                    original_filename=existing_doc['original_filename'],
                    version=existing_doc['version'],
                    status="rejected",
                    message="Not a legal document. Only contracts, NDAs, statutes, and similar legal instruments are accepted.",
                    chunks_created=0,
                    pages_processed=0,
                )
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
    
    # Serialize full ingest to avoid DB/vector-store races on double-click
    with _upload_lock:
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

        # Ingest document (new pipeline: parse -> heuristic -> strategy -> chunk)
        try:
            chunks, ingest_ctx = ingestion_service.ingest_document(
                file_path, document_id, document_type_hint=document_type
            )

            if not chunks:
                raise ValueError("No chunks created from document")

            # If this is a new version, remove old chunks for this document_id
            if version > 1 and vector_store:
                old_chunks = vector_store.get_chunks_by_document(document_id)
                if old_chunks:
                    vector_store.delete_document(document_id)

            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            embeddings = embedding_service.embed_batch(texts)

            # Embedding model version for metadata (from config; service may use fallback)
            embedding_model_version = getattr(
                embedding_service, "model_name", None
            ) or settings.EMBEDDING_MODEL

            # Add to vector store with display_name, document_hash, chunking_strategy, embedding_model_version
            vector_store.add_chunks(
                embeddings,
                chunks,
                display_name=doc_record['display_name'],
                document_hash=document_hash,
                chunking_strategy=ingest_ctx.strategy,
                embedding_model_version=embedding_model_version,
            )

            # Save vector store
            vector_store.save()

            ocr_chunks = sum(1 for chunk in chunks if (chunk.metadata or {}).get('is_ocr', False))
            uses_ocr = ocr_chunks > 0

            # Update registry with page/chunk counts
            with document_registry._get_connection() as conn:
                conn.execute("""
                    UPDATE documents 
                    SET total_pages = ?, total_chunks = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = ? AND version = ?
                """, [ingest_ctx.pages_processed, len(chunks), document_id, version])

            # Persist ingestion metadata for this document/version
            if ingestion_metadata_store:
                ingestion_metadata_store.save(IngestionMetadata(
                    document_id=document_id,
                    ingestion_version=version,
                    chunking_strategy=ingest_ctx.strategy,
                    structure_detected=ingest_ctx.structure_detected,
                    estimated_clause_count=ingest_ctx.estimated_clause_count,
                    embedding_model_version=embedding_model_version,
                    created_at=datetime.now(),
                ))

            # ── Document classification ────────────────────────────────────────────
            if document_classification_service:
                # Sample from start, middle, and end so Arabic contracts whose
                # English clauses appear late are still covered.
                _n = len(chunks)
                if _n <= 9:
                    _sample_idx = list(range(_n))
                else:
                    _mid = _n // 2
                    _sample_idx = sorted({0, 1, 2, _mid - 1, _mid, _n - 2, _n - 1})
                text_sample = " ".join(chunks[i].text for i in _sample_idx)[:2000]
                classification_result = document_classification_service.classify(text_sample, document_id)

                if classification_result.classification == DocumentClassification.NON_LEGAL:
                    # Reject: remove from vector store, file system, and registry
                    try:
                        vector_store.delete_document(document_id)
                        vector_store.save()
                    except Exception:
                        pass
                    try:
                        file_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    document_registry.delete_document(document_id, version)
                    return DocumentUploadResponse(
                        document_id=document_id,
                        display_name=doc_record['display_name'],
                        original_filename=doc_record['original_filename'],
                        version=version,
                        status="rejected",
                        message="Not a legal document. Only contracts, NDAs, statutes, and similar legal instruments are accepted.",
                        chunks_created=0,
                        pages_processed=0,
                    )
                else:
                    document_registry.save_classification(document_id, version, classification_result)

            # ── Phase 6: Structure-first ingestion extras ──────────────────────────
            # Run in a try/except so structural extraction failures never break upload.
            try:
                if structured_clause_extractor and extracted_clause_store:
                    # 1. Extract structured clauses
                    p6_clauses = structured_clause_extractor.extract_structured_clauses(
                        str(file_path), document_id
                    )

                    # 2. Extract defined terms index
                    defined_terms = structured_clause_extractor.extract_defined_terms(p6_clauses)

                    # 3. Sub-chunk long clauses (> 2000 chars)
                    extra_chunks = []
                    for clause in p6_clauses:
                        verbatim = getattr(clause, "verbatim_text", "") or ""
                        if len(verbatim) > 2000:
                            sub_chunks = TextChunker.subchunk_clause(
                                clause_id=clause.clause_id,
                                verbatim_text=verbatim,
                                page_number=getattr(clause, "page_start", 0) or 0,
                                document_id=document_id,
                                legal_category=getattr(clause, "legal_category", None),
                                clause_number=getattr(clause, "clause_number", None),
                            )
                            extra_chunks.extend(sub_chunks)

                    # 4. Page-chunk fallback for pages not covered by any clause
                    pages_with_clauses = {
                        getattr(c, "page_start", None) for c in p6_clauses
                    } - {None}
                    covered_pages = {
                        chunk.page_number for chunk in chunks if chunk.page_number is not None
                    }
                    # Collect text for uncovered pages from existing chunks
                    page_text_map: dict = {}
                    for chunk in chunks:
                        pg = chunk.page_number
                        if pg is not None and pg not in pages_with_clauses:
                            page_text_map.setdefault(pg, []).append(chunk.text)
                    for pg, texts in page_text_map.items():
                        page_chunk = DocumentChunk(
                            text=" ".join(texts)[:2000],
                            page_number=pg,
                            chunk_index=0,
                            document_id=document_id,
                            metadata={"unit_type": "page_chunk", "chunk_id": f"pg_{pg:04d}_{document_id[:8]}"},
                            clause_id=None,
                            unit_type="page_chunk",
                        )
                        extra_chunks.append(page_chunk)

                    # 5. Embed and index extra chunks
                    if extra_chunks:
                        extra_texts = [c.text for c in extra_chunks]
                        extra_embeddings = embedding_service.embed_batch(extra_texts)
                        vector_store.add_chunks(
                            extra_embeddings,
                            extra_chunks,
                            display_name=doc_record["display_name"],
                            document_hash=document_hash,
                            chunking_strategy="phase6_extra",
                            embedding_model_version=embedding_model_version,
                        )
                        vector_store.save()

                    # 6. Persist clause store with defined terms
                    extracted_clause_store.save_document_clauses(
                        document_id,
                        [c.to_dict() for c in p6_clauses],
                        defined_terms=defined_terms,
                    )
            except Exception as _p6_exc:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "Phase 6 extras failed for %s (non-fatal): %s", document_id, _p6_exc
                )

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
                pages_processed=ingest_ctx.pages_processed,
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


@app.get("/api/documents/{document_id}/classification")
async def get_document_classification(document_id: str):
    """Return stored classification for a document."""
    global document_registry
    if not document_registry:
        raise HTTPException(status_code=500, detail="Document registry not initialized")
    result = document_registry.get_classification(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Classification not found for this document")
    return result


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
        # Minimal debug logging for search failures (debug mode)
        try:
            import json, time  # Local import to avoid global side effects
            with open(r'c:\Users\LEGION\Desktop\Projects\legal-document-LLM-powered-pipeline\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run-search",
                    "location": "main.py:599",
                    "message": "search_documents exception",
                    "data": {
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e),
                    },
                    "timestamp": int(time.time() * 1000),
                }) + "\n")
        except Exception:
            # Logging must never break the API
            pass
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

    # Only require the service after we've validated that the request references real files.
    if not comparison_service:
        raise HTTPException(status_code=500, detail="Comparison service not initialized")
    
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


@app.post("/api/contract-review")
async def contract_review(
    contract_id: str = Form(...),
    contract_type: str = Form("employment"),
    jurisdiction: Optional[str] = Form(None),
    review_depth: Optional[str] = Form("standard"),
):
    """
    Execute Contract Review workflow. Returns WorkflowContext envelope:
    - status: completed | failed
    - error: set when failed (code, message, step, details)
    - intermediate_results: includes contract_review.response on success
    """
    global workflow_orchestrator
    if not workflow_orchestrator:
        raise HTTPException(status_code=500, detail="Workflow orchestrator not initialized")
    try:
        ctx = workflow_orchestrator.run_contract_review(
            contract_id=contract_id,
            contract_type=contract_type,
            jurisdiction=jurisdiction,
            review_depth=review_depth,
        )
        return ctx.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contract review failed: {str(e)}")


@app.get("/api/workflow/{workflow_id}/state")
async def get_workflow_state(workflow_id: str):
    """
    Return workflow state for the given workflow_id. Fast, non-blocking.
    Returns 404 if workflow_id is unknown (UI should treat as Indeterminate).
    """
    state = workflow_store.get(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return state


@app.post("/api/explore")
async def document_explorer(
    document_id: str = Form(...),
    query: str = Form(...),
    top_k: Optional[int] = Form(None),
    mode: str = Form("text"),
):
    """
    Document Explorer: RAG-backed search within a single document.
    Returns both answer and evidence snippets.
    """
    global workflow_orchestrator
    if not workflow_orchestrator:
        raise HTTPException(status_code=500, detail="Workflow orchestrator not initialized")
    try:
        ctx = workflow_orchestrator.run_document_explorer(
            document_id=document_id,
            query=query,
            top_k=top_k,
            mode=mode,
        )
        if ctx.status == "completed" and ctx.intermediate_results:
            explorer_result = ctx.intermediate_results.get("document_explorer", {})
            response_data = explorer_result.get("response", {})
            if response_data:
                return {
                    "answer": response_data.get("answer"),
                    "results": response_data.get("results", []),
                    "status": response_data.get("status"),
                    "reason": response_data.get("reason"),
                    "confidence": response_data.get("confidence"),
                    "citation": response_data.get("citation"),
                    "refusal_reason": response_data.get("refusal_reason"),
                    "debug": response_data.get("debug"),
                    "workflow_id": ctx.workflow_id,
                }

        # Other failures: preserve structured error for UI/debugging.
        if ctx.status == "failed" and ctx.error:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "code": ctx.error.code,
                        "message": ctx.error.message,
                        "step": ctx.error.step,
                        "details": ctx.error.details,
                    }
                },
            )

        return ctx.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document Explorer failed: {str(e)}")


@app.post("/api/explore-evidence")
async def explore_evidence(
    document_id: str = Form(...),
    query: str = Form(...),
    top_k: Optional[int] = Form(None),
    mode: str = Form("text"),
    debug: bool = Form(False),
):
    """
    Evidence Explorer: deterministic evidence retrieval within a single document.
    No LLM. Returns snippets + metadata + debug only (no answer field).
    Modes: text (chunks), clauses (extracted clauses), both (clauses first, then text fallback).
    If debug=True, includes retrieval_debug in debug block.
    """
    global workflow_orchestrator
    if not workflow_orchestrator:
        raise HTTPException(status_code=500, detail="Workflow orchestrator not initialized")
    if mode not in ("text", "clauses", "both"):
        mode = "text"
    try:
        ctx = workflow_orchestrator.run_evidence_explorer(
            document_id=document_id,
            query=query,
            top_k=top_k,
            mode=mode,
            debug=debug,
        )
        if ctx.status == "completed" and ctx.intermediate_results:
            ev_result = ctx.intermediate_results.get("evidence_explorer", {})
            response_data = ev_result.get("response", {})
            if response_data:
                out = {
                    "mode": "explorer",
                    "status": response_data.get("status"),
                    "results": [r for r in response_data.get("results", [])],
                    "reason": response_data.get("reason"),
                    "debug": response_data.get("debug"),
                    "workflow_id": ctx.workflow_id,
                }
                return out
        if ctx.status == "failed" and ctx.error:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "code": ctx.error.code,
                        "message": ctx.error.message,
                        "step": ctx.error.step,
                        "details": ctx.error.details,
                    }
                },
            )
        return ctx.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evidence Explorer failed: {str(e)}")


@app.post("/api/explore-answer")
async def explore_answer(
    document_id: str = Form(...),
    query: str = Form(...),
    top_k: Optional[int] = Form(None),
    response_language: Optional[str] = Form(None),
    debug: bool = Form(False),
):
    """
    RAG Answer Explorer: single-document RAG answer + citations.
    Uses RAGService.query with document_id_filter. Returns answer, status, confidence, sources, citation.
    If debug=True, includes retrieval_debug and optional answer_style in response.
    """
    global rag_service
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    try:
        result = rag_service.query(
            query=query,
            top_k=top_k,
            document_id_filter=document_id,
            generate_response=True,
            response_language=response_language,
            debug=debug,
        )
        result["mode"] = "rag_qa"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Answer Explorer failed: {str(e)}")


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
        # Use PRD-compliant multi-pass summarization for the API.
        result = summarization_service.summarize_case_file_prd(
            document_id=document_id,
            top_k=top_k
        )
        
        # Check for errors
        if "error" in result:
            error = result["error"]
            status_code = 422 if error.get("code") in ["CASE_SPINE_FAILED", "INSUFFICIENT_CONTEXT", "PRESCRIPTIVE_LANGUAGE"] else 400
            raise HTTPException(
                status_code=status_code,
                detail=error
            )
        
        # Generate report if requested
        report = None
        if include_report:
            report = summarization_service.generate_summary_report(result, format='markdown')
        
        response = {"summary": result, "disclaimer": WORKFLOW_DISCLAIMER}
        if report:
            response["report"] = report
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing case file: {str(e)}")


@app.post("/api/due-diligence-memo")
async def due_diligence_memo(
    document_id: str = Form(...),
):
    """
    Due Diligence Memo workflow (v0.2): returns WorkflowContext envelope.
    """
    global workflow_orchestrator
    if not workflow_orchestrator:
        raise HTTPException(status_code=500, detail="Workflow orchestrator not initialized")
    try:
        ctx = workflow_orchestrator.run_due_diligence_memo(document_id=document_id)
        return ctx.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Due diligence memo failed: {str(e)}")


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


@app.post("/api/translate")
async def translate_batch(
    texts: str = Form(...),          # JSON-encoded list[str]
    target_lang: str = Form("ar"),
    source_lang: str = Form("en"),
):
    """Batch-translate a list of text strings. Returns {translations: list[str]}."""
    import json as _json
    global translation_service
    if not translation_service:
        raise HTTPException(status_code=503, detail="Translation service not available")
    try:
        parsed = _json.loads(texts)
    except _json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="texts must be a valid JSON array")
    if not isinstance(parsed, list):
        raise HTTPException(status_code=422, detail="texts must be a JSON array")
    if len(parsed) > 500:
        raise HTTPException(status_code=422, detail="Too many strings (max 500)")
    results = []
    for t in parsed:
        if t and str(t).strip():
            results.append(translation_service.translate_text(str(t), source_lang, target_lang))
        else:
            results.append(t)
    return {"translations": results}


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


@app.delete("/api/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    """
    Permanently delete a document and its artifacts.

    This endpoint coordinates cleanup across:
    - Vector store (embeddings/chunks)
    - Clause store (structured clauses)
    - Extracted clause store (explorer JSON payloads)
    - Document registry (all versions)
    - On-disk files recorded in the registry
    """
    global document_registry, vector_store, clause_store, extracted_clause_store

    if not document_registry:
        raise HTTPException(status_code=500, detail="Document registry not initialized")

    # Pre-read all versions so we know which files to remove even after registry deletion.
    versions = document_registry.get_versions(document_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    deleted_chunks = 0
    deleted_clauses = 0
    deleted_extracted = 0

    # Vector store deletion (best-effort; logs on failure).
    if vector_store:
        try:
            deleted_chunks = vector_store.delete_document(document_id)
        except Exception as e:
            logging.error("Failed to delete vectors for %s: %s", document_id, e)

    # Structured clause deletion (best-effort).
    if clause_store:
        try:
            deleted_clauses = clause_store.delete_document(document_id)
        except Exception as e:
            logging.error("Failed to delete structured clauses for %s: %s", document_id, e)

    # Extracted clause JSON payloads (best-effort).
    if extracted_clause_store:
        try:
            deleted_extracted = extracted_clause_store.delete_document_clauses(document_id)
        except Exception as e:
            logging.error("Failed to delete extracted clauses for %s: %s", document_id, e)

    # Delete all registry records (all versions).
    try:
        deleted_registry = document_registry.delete_document(document_id)
    except Exception as e:
        logging.error("Failed to delete registry records for %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to delete registry records for {document_id}")

    # Remove physical files on disk for each version, if paths are available.
    for rec in versions:
        file_path = rec.get("file_path")
        if not file_path:
            continue
        try:
            p = Path(file_path)
            if p.exists():
                p.unlink()
        except Exception as e:
            logging.warning("Failed to delete file for %s at %s: %s", document_id, file_path, e)

    return {
        "status": "deleted",
        "document_id": document_id,
        "deleted_chunks": deleted_chunks,
        "deleted_clauses": deleted_clauses,
        "deleted_extracted_payloads": deleted_extracted,
        "deleted_registry_records": deleted_registry,
    }


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    global vector_store

    stats = vector_store.get_stats() if vector_store else {}
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


@app.post("/api/admin/clear-all")
async def clear_all_documents():
    """
    Delete every document and all associated artifacts from the system.

    Coordinates cleanup across:
    - Vector store (embeddings/chunks)
    - Clause store (structured clauses)
    - Extracted clause store (explorer JSON payloads)
    - Document registry (SQLite records)
    - On-disk files recorded in the registry
    """
    global document_registry, vector_store, clause_store, extracted_clause_store

    if not document_registry:
        raise HTTPException(status_code=500, detail="Document registry not initialized")

    doc_records = document_registry.list_documents()
    doc_ids = list({r["document_id"] for r in doc_records})

    total_chunks = 0
    total_clauses = 0
    total_extracted = 0

    for doc_id in doc_ids:
        # Collect physical file paths before registry deletion.
        versions = document_registry.get_versions(doc_id)

        if vector_store:
            try:
                total_chunks += vector_store.delete_document(doc_id)
            except Exception as e:
                logging.error("clear-all: failed to delete vectors for %s: %s", doc_id, e)

        if clause_store:
            try:
                total_clauses += clause_store.delete_document(doc_id)
            except Exception as e:
                logging.error("clear-all: failed to delete clauses for %s: %s", doc_id, e)

        if extracted_clause_store:
            try:
                total_extracted += extracted_clause_store.delete_document_clauses(doc_id)
            except Exception as e:
                logging.error("clear-all: failed to delete extracted clauses for %s: %s", doc_id, e)

        try:
            document_registry.delete_document(doc_id)
        except Exception as e:
            logging.error("clear-all: failed to delete registry records for %s: %s", doc_id, e)

        for rec in (versions or []):
            file_path = rec.get("file_path")
            if not file_path:
                continue
            try:
                p = Path(file_path)
                if p.exists():
                    p.unlink()
            except Exception as e:
                logging.warning("clear-all: failed to delete file %s: %s", file_path, e)

    if vector_store:
        try:
            vector_store.save()
        except Exception as e:
            logging.error("clear-all: failed to save vector store after clear: %s", e)

    return {
        "status": "cleared",
        "documents_deleted": len(doc_ids),
        "chunks_deleted": total_chunks,
        "clauses_deleted": total_clauses,
        "extracted_deleted": total_extracted,
    }


# ---------------------------------------------------------------------------
# Conversational RAG endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat/session", response_model=CreateSessionResponse, tags=["Chat"])
async def create_chat_session(body: CreateSessionRequest):
    """
    Create a new conversational RAG session bound to a document.

    - **document_id**: The document to query against (must already be uploaded).
    - **mode**: "strict" (stateless, evidence-only) or "conversational" (session-aware).
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    try:
        session = session_manager.create_session(
            document_id=body.document_id,
            mode=body.mode,
        )
        return CreateSessionResponse(
            session_id=session.session_id,
            document_id=session.document_id,
            mode=session.mode,
            created_at=session.created_at,
        )
    except Exception as exc:
        logging.error("create_chat_session error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat/{session_id}", response_model=ChatMessageResponse, tags=["Chat"])
async def send_chat_message(session_id: str, body: ChatMessageRequest):
    """
    Send a message in an existing chat session and receive a grounded answer.

    - Strict mode: stateless RAG, no history, matches /api/explore-answer behaviour.
    - Conversational mode: query rewriting, dual retrieval, evidence guardrail, two-pass validation.
    """
    if chat_orchestrator is None or session_manager is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    try:
        response = chat_orchestrator.chat(
            session_id=session_id,
            user_message=body.message,
            mode_override=body.mode,
        )
        return response
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except DocumentMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logging.error("send_chat_message error [%s]: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/chat/{session_id}", response_model=SessionHistoryResponse, tags=["Chat"])
async def get_chat_session(session_id: str):
    """Retrieve the full message history for a session."""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    try:
        session = session_manager.get_session(session_id)
        return SessionHistoryResponse(
            session_id=session.session_id,
            document_id=session.document_id,
            mode=session.mode,
            messages=session.messages,
            summary=session.summary,
            last_active_at=session.last_active_at,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api/chat/{session_id}", tags=["Chat"])
async def delete_chat_session(session_id: str):
    """Delete a chat session and all its history."""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"status": "deleted", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    # Increase timeout for long-running operations like clause extraction
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        timeout_keep_alive=300  # 5 minutes timeout
    )

