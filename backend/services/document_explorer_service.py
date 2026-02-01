"""
Document Explorer service.

Provides a read-only, evidence-only semantic search workflow over a single
document, with strict query classification to prevent interpretive/advisory use.

Key properties:
- NEVER calls an LLM
- Single-document scope only
- Returns chunks/snippets with page numbers and scores
- Classifies queries as "locational" vs "interpretive" using simple rules
"""

from __future__ import annotations

import re
from typing import List

from backend.config import settings
from backend.models.workflow import (
    WorkflowContext,
    DocumentExplorerRequest,
    DocumentExplorerResponse,
    DocumentExplorerResult,
)
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore


INTERPRETIVE_PATTERNS = [
    # General interpretation / advice patterns (English)
    r"^is\b",
    r"^are\b",
    r"^can\b",
    r"^does\b",
    r"^should\b",
    r"\bshould\b",
    r"\bmust\b",
    r"\brecommend\b",
    r"legal",
    r"enforceable",
    r"\benforce\b",
    r"allowed",
    r"\bvalid\b",
    r"\billegal\b",
]


def classify_query_for_explorer(query: str) -> str:
    """
    Classify a Document Explorer query as "interpretive" or "locational".

    This is intentionally simple and rule-based to avoid surprises.
    """
    q = query.lower().strip()
    for pattern in INTERPRETIVE_PATTERNS:
        if re.search(pattern, q):
            return "interpretive"
    return "locational"


class DocumentExplorerService:
    """
    Read-only, evidence-only document explorer.

    The public contract is expressed via WorkflowContext:
    - On success: ctx.status == "completed" and
      ctx.intermediate_results["document_explorer"]["response"] contains a
      DocumentExplorerResponse.model_dump().
    - On interpretive queries: ctx.status == "failed" with code="INTERPRETIVE_QUERY".
    - On no evidence: ctx.status == "failed" with code="NO_EVIDENCE" and a
      human-readable message "Not specified in the provided document.".
    """

    STEP_NAME = "document_explorer"

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def run(self, ctx: WorkflowContext, request: DocumentExplorerRequest) -> WorkflowContext:
        """
        Execute the Document Explorer workflow for a single document.

        This method NEVER calls an LLM. It only:
        - Classifies the query
        - Performs semantic search (and simple fallback)
        - Returns evidence snippets
        """
        if ctx.status == "failed":
            # Respect existing failure; do not proceed.
            return ctx

        ctx.current_step = f"{self.STEP_NAME}.classify_query"

        classification = classify_query_for_explorer(request.query)
        if classification == "interpretive":
            ctx.fail(
                code="INTERPRETIVE_QUERY",
                message=(
                    "Document Explorer supports evidence location only, not legal interpretation."
                ),
                step=self.STEP_NAME,
                details={
                    "query": request.query,
                    "document_id": request.document_id,
                },
            )
            return ctx

        # Semantic search within a single document
        ctx.current_step = f"{self.STEP_NAME}.semantic_search"

        try:
            query_embedding = self.embedding_service.embed_text(request.query)
        except Exception as e:
            ctx.fail(
                code="EMBEDDING_ERROR",
                message=f"Failed to embed query for Document Explorer: {str(e)}",
                step=self.STEP_NAME,
                details={"query": request.query},
            )
            return ctx

        top_k = request.top_k or settings.DOCUMENT_EXPLORER_MAX_RESULTS

        try:
            chunks = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                document_id_filter=request.document_id,
            )
        except Exception as e:
            ctx.fail(
                code="VECTOR_SEARCH_ERROR",
                message=f"Vector search failed for Document Explorer: {str(e)}",
                step=self.STEP_NAME,
                details={"document_id": request.document_id},
            )
            return ctx

        # Fallback pass: if nothing matches the default threshold, retry with the minimum
        # threshold (still guarded, but more permissive for short queries / noisy OCR).
        if not chunks:
            try:
                chunks = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    document_id_filter=request.document_id,
                    similarity_threshold=settings.MIN_SIMILARITY_THRESHOLD,
                )
            except Exception as e:
                ctx.fail(
                    code="VECTOR_SEARCH_ERROR",
                    message=f"Vector search failed for Document Explorer (fallback): {str(e)}",
                    step=self.STEP_NAME,
                    details={"document_id": request.document_id},
                )
                return ctx

        # If nothing found, fail closed with the mandated message.
        if not chunks:
            ctx.fail(
                code="NO_EVIDENCE",
                message="Not specified in the provided document.",
                step=self.STEP_NAME,
                details={"document_id": request.document_id, "query": request.query},
            )
            return ctx

        # Map chunks to DocumentExplorerResult objects
        results: List[DocumentExplorerResult] = []
        for chunk in chunks:
            text = chunk.get("text") or ""
            # build a short snippet (conservative)
            snippet = text
            if len(snippet) > 600:
                snippet = snippet[:600].rstrip() + "..."

            results.append(
                DocumentExplorerResult(
                    document_id=chunk.get("document_id", request.document_id),
                    page_number=chunk.get("page_number", 0),
                    chunk_index=chunk.get("chunk_index", 0),
                    text_snippet=snippet,
                    score=float(chunk.get("score", 0.0)),
                )
            )

        response = DocumentExplorerResponse(results=results)

        ctx.add_result(
            self.STEP_NAME,
            {
                "request": request.model_dump(),
                "response": response.model_dump(),
            },
        )
        ctx.status = "completed"
        return ctx

