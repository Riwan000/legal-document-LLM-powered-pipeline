"""
Document Explorer service.

RAG-backed explorer that returns both an answer and evidence snippets
for a single document scope.

Evidence contract (see EVIDENCE_EXPLORER_SPEC.md):
- Evidence Explorer returns each chunk or clause independently; it never merges,
  summarizes, or stitches text.
- Lexical matches are always eligible; semantic helps ranking, not eligibility.
- No hard score cutoff once lexical/heading match exists.
"""

from __future__ import annotations

from typing import List, Dict, Any

from backend.config import settings
from backend.models.workflow import (
    WorkflowContext,
    DocumentExplorerRequest,
    DocumentExplorerResponse,
    DocumentExplorerResult,
)
from backend.services.rag_service import RAGService
from backend.services.extracted_clause_store import ExtractedClauseStore
from backend.services.text_normalizer import normalize_for_match


class DocumentExplorerService:
    """
    RAG-backed document explorer (single-document scope).
    """

    STEP_NAME = "document_explorer"

    def __init__(self, rag_service: RAGService, extracted_clause_store: ExtractedClauseStore):
        self.rag_service = rag_service
        self.extracted_clause_store = extracted_clause_store

    def _lexical_keywords(self, query: str) -> List[str]:
        normalized = normalize_for_match(query)
        if "termination" in normalized or "terminate" in normalized:
            return [
                "terminate",
                "termination",
                "terminated",
                "termination notice",
                "end this agreement",
            ]
        return [t for t in normalized.split() if len(t) > 2]

    def _lexical_fallback(self, chunks: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        if not keywords:
            return []
        matches: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = (chunk.get("text") or "").lower()
            if any(k in text for k in keywords):
                entry = dict(chunk)
                entry.setdefault("score", 1.0)
                entry["lexical_match"] = True
                matches.append(entry)
        return matches

    def _search_text_mode(
        self,
        document_id: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Evidence Explorer text search: semantic + lexical union, recall-first.

        - Lexical matches are always eligible results, even if semantic score is low or zero.
        - Semantic search helps ranking, not eligibility. No hard score cutoff is applied
          once a lexical (or heading) match exists (EVIDENCE_EXPLORER_SPEC §1, §2).
        """
        top_k = 25
        keywords = self._lexical_keywords(query)
        chunks: List[Dict[str, Any]] = []
        vector_store = getattr(self.rag_service, "vector_store", None)
        if vector_store:
            chunks = vector_store.get_chunks_by_document(document_id)

        # Semantic: use no threshold so score is used for ranking only, not eligibility
        semantic_results = self.rag_service.search(
            query=query,
            top_k=top_k,
            document_id_filter=document_id,
            similarity_threshold=None,
        )
        lexical_results = self._lexical_fallback(chunks, keywords)

        # Union: lexical matches are always eligible; add semantic results not already in lexical
        seen = {(r.get("document_id"), r.get("chunk_index")) for r in lexical_results}
        combined: List[Dict[str, Any]] = list(lexical_results)
        for r in semantic_results:
            key = (r.get("document_id"), r.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                combined.append(r)
        # Sort by score desc (ranking only; no filtering by score)
        combined.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        # Cap for response size (still no score cutoff that would exclude lexical)
        combined = combined[:top_k]

        return {
            "results": combined,
            "semantic": len(semantic_results),
            "lexical": len(lexical_results),
        }

    def _search_clause_mode(
        self,
        document_id: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        clauses = self.extracted_clause_store.get_document_clauses(document_id)
        normalized_query = normalize_for_match(query)
        if not clauses or not normalized_query:
            return []
        matches = []
        for clause in clauses:
            heading = normalize_for_match(str(clause.get("clause_heading") or ""))
            body = normalize_for_match(str(clause.get("verbatim_text") or ""))
            if normalized_query in heading or normalized_query in body:
                matches.append(clause)
        return matches

    def run(self, ctx: WorkflowContext, request: DocumentExplorerRequest) -> WorkflowContext:
        """
        Execute the Document Explorer workflow for a single document.

        This method uses RAG to return both answer and evidence snippets.
        """
        if ctx.status == "failed":
            # Respect existing failure; do not proceed.
            return ctx
        mode = request.mode or "text"
        if mode not in {"text", "clauses"}:
            ctx.fail(
                code="INVALID_MODE",
                message="Invalid explorer mode. Use 'text' or 'clauses'.",
                step=self.STEP_NAME,
                details={"mode": mode},
            )
            return ctx
        # For now, both modes use full-document text search; mode is only logged.
        ctx.current_step = f"{self.STEP_NAME}.text_search"
        search_result = self._search_text_mode(request.document_id, request.query)
        if not search_result["results"]:
            response = DocumentExplorerResponse(
                status="not_found",
                reason=(
                    "No termination-related text found in raw document content after semantic "
                    "and lexical search."
                ),
                results=[],
                debug={
                    "debug_version": "v1",
                    "mode": mode,
                    "semantic_results": search_result["semantic"],
                    "lexical_results": search_result["lexical"],
                },
            )
        else:
            ctx.current_step = f"{self.STEP_NAME}.rag_answer"
            try:
                rag_result = self.rag_service.query(
                    query=request.query,
                    top_k=request.top_k or settings.DOCUMENT_EXPLORER_MAX_RESULTS,
                    document_id_filter=request.document_id,
                    generate_response=True,
                    chunks_override=search_result["results"],
                )
            except Exception as e:
                ctx.fail(
                    code="RAG_ERROR",
                    message=f"Document Explorer failed: {str(e)}",
                    step=self.STEP_NAME,
                    details={"document_id": request.document_id},
                )
                return ctx

            sources = rag_result.get("sources", []) or []
            results: List[DocumentExplorerResult] = []
            for source in sources:
                text = source.get("text", "")
                snippet = text[:600].rstrip() + "..." if len(text) > 600 else text
                results.append(
                    DocumentExplorerResult(
                        document_id=source.get("document_id", request.document_id),
                        page_number=source.get("page_number", 0),
                        chunk_index=source.get("chunk_index", 0),
                        text_snippet=snippet,
                        score=float(source.get("score", 0.0)),
                        display_name=source.get("display_name"),
                        citation=source.get("citation"),
                    )
                )

            # Evidence is authority for absence: RAG cannot return not_specified when we have evidence (EVIDENCE_EXPLORER_SPEC §4)
            status = rag_result.get("status")
            if status == "not_specified" and search_result["results"]:
                status = "explicitly_stated"

            response = DocumentExplorerResponse(
                answer=rag_result.get("answer"),
                status=status,
                confidence=rag_result.get("confidence"),
                citation=rag_result.get("citation"),
                refusal_reason=rag_result.get("refusal_reason"),
                results=results,
                debug={
                    "debug_version": "v1",
                    "mode": mode,
                    "semantic_results": search_result["semantic"],
                    "lexical_results": search_result["lexical"],
                },
            )

        ctx.add_result(
            self.STEP_NAME,
            {
                "request": request.model_dump(),
                "response": response.model_dump(),
            },
        )
        ctx.status = "completed"
        return ctx

