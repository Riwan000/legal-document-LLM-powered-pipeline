"""
Document ingestion service.
Orchestrates file parsing, heuristic structure detection, strategy selection, and chunking.
"""
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from datetime import datetime

from backend.utils.file_parser import FileParser
from backend.utils.chunking import TextChunker
from backend.models.document import DocumentChunk, DocumentMetadata, DocumentUploadResponse
from backend.models.document_structure import (
    PageInfo,
    HeuristicStructureResult,
    IngestionContext,
)
from backend.config import settings
from backend.services.structure_heuristic_service import StructureHeuristicService
from backend.services.chunking_strategy_service import ChunkingStrategyService

logger = logging.getLogger(__name__)

# Safety thresholds (plan)
COVERAGE_THRESHOLD = 0.95
MAX_CHUNKS_PER_PAGE = getattr(settings, "MAX_REASONABLE_CHUNKS_PER_PAGE", 200)


def detect_document_language(pages: List[Tuple[str, int]]) -> str:
    """
    Detect primary language of document from parsed pages.
    Accepts list of (text, page_number) or list of PageInfo (uses .text).
    """
    sample_text = ""
    sample_size = min(5, len(pages))
    indices = [0] + [len(pages) // 2] + list(range(min(3, len(pages))))

    for idx in set(indices):
        if idx < len(pages):
            page_data = pages[idx]
            if hasattr(page_data, "text"):
                sample_text += page_data.text + " "
            elif len(page_data) >= 1:
                sample_text += page_data[0] + " "

    if not sample_text.strip():
        return "en"
    arabic_chars = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
    text_chars = set(sample_text.replace(" ", ""))
    arabic_ratio = len(text_chars.intersection(arabic_chars)) / max(len(text_chars), 1)
    if arabic_ratio > settings.LANGUAGE_DETECTION_THRESHOLD:
        return "ar"
    return "en"


class DocumentIngestionService:
    """Service for ingesting and processing documents with heuristic structure and strategy selection."""

    def __init__(self):
        self.parser = FileParser()
        self.chunker = TextChunker()
        self.structure_heuristic = StructureHeuristicService()
        self.chunking_strategy_service = ChunkingStrategyService()
        settings.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
        settings.TEMPLATES_PATH.mkdir(parents=True, exist_ok=True)

    def parse_document(self, file_path: Path) -> List[PageInfo]:
        """Parse file and return list of PageInfo (text, page_number, is_ocr)."""
        return self.parser.parse_file_to_pages(file_path)

    def ingest_document(
        self,
        file_path: Path,
        document_id: str,
        document_type_hint: Optional[str] = None,
    ) -> Tuple[List[DocumentChunk], IngestionContext]:
        """
        Run the full ingestion pipeline: parse, detect structure, select strategy, chunk.
        Returns chunks and ingestion context for embedding/indexing and metadata storage.
        """
        pages = self.parse_document(file_path)
        if not pages:
            raise ValueError("No text content found in document")

        heuristics = self.structure_heuristic.detect(pages)
        strategy = self.chunking_strategy_service.select(heuristics, document_type_hint)

        use_clause_aware = getattr(settings, "USE_CLAUSE_AWARE_INGESTION", False)
        if not use_clause_aware:
            strategy = "sentence"
            logger.info(
                "USE_CLAUSE_AWARE_INGESTION=False; using sentence chunking (would-be strategy: %s)",
                self.chunking_strategy_service.select(heuristics, document_type_hint),
            )

        if strategy == "clause_aware":
            clauses_by_page = self.structure_heuristic.extract_heuristic_clauses_by_page(pages)
            page_tuples = [(p.text, p.page_number, p.is_ocr) for p in pages]
            chunks = self.chunker.chunk_pages(page_tuples, document_id, clauses_by_page=clauses_by_page)
            chunks, any_fallback = self._apply_safety_fallbacks(pages, chunks, document_id)
            if any_fallback:
                strategy = "sentence"
        else:
            page_tuples = [(p.text, p.page_number, p.is_ocr) for p in pages]
            chunks = self.chunker.chunk_pages(page_tuples, document_id)

        uses_ocr = any(p.is_ocr for p in pages)
        ocr_chunks = sum(1 for c in chunks if (c.metadata or {}).get("is_ocr", False))
        lang = detect_document_language(pages)
        for c in chunks:
            if c.metadata is None:
                c.metadata = {}
            c.metadata["language"] = lang

        ctx = IngestionContext(
            strategy=strategy,
            structure_detected=heuristics.has_structured_headings,
            estimated_clause_count=heuristics.estimated_clause_count,
            structure_confidence=heuristics.confidence,
            pages_processed=len(pages),
            uses_ocr=uses_ocr,
            ocr_chunks=ocr_chunks,
        )
        return chunks, ctx

    def _apply_safety_fallbacks(
        self,
        pages: List[PageInfo],
        chunks: List[DocumentChunk],
        document_id: str,
    ) -> Tuple[List[DocumentChunk], bool]:
        """
        Apply coverage and max-chunks-per-page checks. Replace clause-aware chunks for a page
        with sentence chunks when safety fails. Returns (new_chunks, True if any fallback occurred).
        """
        page_text_by_num = {p.page_number: p.text for p in pages}
        chunks_by_page: dict = {}
        for c in chunks:
            chunks_by_page.setdefault(c.page_number, []).append(c)
        any_fallback = False
        result = []
        for page in pages:
            pnum = page.page_number
            page_chunks = chunks_by_page.get(pnum, [])
            page_text = page_text_by_num.get(pnum, "")
            if not page_text.strip():
                result.extend(page_chunks)
                continue
            # Coverage check
            reconstructed = "".join(c.text for c in page_chunks)
            coverage = len(reconstructed) / len(page_text) if page_text else 0
            if coverage < COVERAGE_THRESHOLD:
                logger.warning(
                    "Clause-aware coverage %.2f < %.2f for page %s; falling back to sentence chunking",
                    coverage, COVERAGE_THRESHOLD, pnum,
                )
                sentence_chunks = TextChunker.chunk_text(
                    page_text, pnum, document_id, is_ocr=page.is_ocr
                )
                for c in sentence_chunks:
                    if c.metadata is None:
                        c.metadata = {}
                    c.metadata["chunk_type"] = "sentence"
                result.extend(sentence_chunks)
                any_fallback = True
                continue
            if len(page_chunks) > MAX_CHUNKS_PER_PAGE:
                logger.warning(
                    "Clause-aware chunks %s > %s for page %s; falling back to sentence chunking",
                    len(page_chunks), MAX_CHUNKS_PER_PAGE, pnum,
                )
                sentence_chunks = TextChunker.chunk_text(
                    page_text, pnum, document_id, is_ocr=page.is_ocr
                )
                for c in sentence_chunks:
                    if c.metadata is None:
                        c.metadata = {}
                    c.metadata["chunk_type"] = "sentence"
                result.extend(sentence_chunks)
                any_fallback = True
                continue
            result.extend(page_chunks)
        return result, any_fallback

    def get_chunks_from_document(
        self,
        file_path: Path,
        document_id: str,
        use_clause_aware_chunking: bool = False,
        clauses: Optional[List[dict]] = None,
    ) -> List[DocumentChunk]:
        """
        Parse and chunk a document, returning chunks (backward-compatible).
        When use_clause_aware_chunking and clauses are provided, uses those clauses;
        otherwise delegates to the new pipeline (ingest_document) and returns chunks only.
        """
        if use_clause_aware_chunking and clauses:
            pages = self.parser.parse_file(file_path)
            if not pages:
                return []
            clauses_by_page = {}
            for clause in clauses:
                pnum = clause.get("page_number", 0)
                clauses_by_page.setdefault(pnum, []).append(clause)
            chunks = self.chunker.chunk_pages(pages, document_id, clauses_by_page=clauses_by_page)
        else:
            chunks, _ = self.ingest_document(file_path, document_id, document_type_hint=None)
        # Ensure language on chunks (ingest_document path already sets it; clause path may not)
        if chunks and (chunks[0].metadata or {}).get("language") is None:
            pages_for_lang = self.parser.parse_file(file_path)
            lang = detect_document_language(pages_for_lang)
            for c in chunks:
                if c.metadata is None:
                    c.metadata = {}
                c.metadata["language"] = lang
        return chunks

    def ingest_document_with_response(
        self,
        file_path: Path,
        document_type: str = "document",
        document_id: Optional[str] = None,
    ) -> DocumentUploadResponse:
        """
        Legacy: Ingest a document and return DocumentUploadResponse (no registry/vector).
        For upload flow use ingest_document() and wire response in main.
        """
        if document_id is None:
            document_id = str(uuid.uuid4())
        filename = file_path.name
        try:
            chunks, ctx = self.ingest_document(file_path, document_id, document_type_hint=document_type)
            display_name = Path(filename).stem
            return DocumentUploadResponse(
                document_id=document_id,
                display_name=display_name,
                original_filename=filename,
                version=1,
                status="success",
                message="Document ingested successfully",
                chunks_created=len(chunks),
                pages_processed=ctx.pages_processed,
                uses_ocr=ctx.uses_ocr,
                ocr_chunks=ctx.ocr_chunks,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        except Exception as e:
            fallback_document_id = document_id
            fallback_filename = file_path.name
            fallback_display_name = Path(fallback_filename).stem
            return DocumentUploadResponse(
                document_id=fallback_document_id,
                display_name=fallback_display_name,
                original_filename=fallback_filename,
                version=1,
                status="error",
                message=f"Error ingesting document: {str(e)}",
                chunks_created=0,
                pages_processed=0,
                uses_ocr=False,
                ocr_chunks=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
