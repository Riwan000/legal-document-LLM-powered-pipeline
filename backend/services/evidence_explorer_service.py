"""
Evidence Explorer service.

Deterministic, single-document evidence retrieval. No LLM.
Supports modes: text (chunks), clauses (extracted clauses), both (clauses first, then text fallback).
Uses OCR-tolerant normalization, query families, and heading-prioritized scoring for clauses.
Returns evidence-only response (no answer field).
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

from backend.config import settings
from backend.models.workflow import (
    WorkflowContext,
    EvidenceExplorerRequest,
    EvidenceExplorerResponse,
    EvidenceExplorerResult,
)
from backend.services.rag_service import RAGService
from backend.services.extracted_clause_store import ExtractedClauseStore
from backend.services.text_normalizer import normalize_for_match, detect_ocr_noise

# Query families for expansion (plan §4)
QUERY_FAMILIES: Dict[str, List[str]] = {
    "termination": [
        "terminate",
        "termination",
        "terminated",
        "notice",
        "expiry",
        "end of agreement",
        "probation",
    ],
    "notice": ["notice", "notify", "notification", "written", "days"],
    "compensation": ["compensation", "payment", "salary", "wages", "benefits"],
    "governing_law": ["governing law", "jurisdiction", "law", "applicable"],
}


def expand_query_terms(query: str) -> List[str]:
    """
    Normalize query and return expanded terms: if any family key/term appears
    in normalized query, return that family's list; else tokenized normalized terms.
    """
    normalized = normalize_for_match(query)
    if not normalized:
        return []
    tokens = [t for t in normalized.split() if len(t) > 2]
    for _key, terms in QUERY_FAMILIES.items():
        for term in terms:
            if term in normalized or any(term in t or t in term for t in tokens):
                return list(terms)
    return tokens if tokens else [normalized]


def score_clause(clause: Dict[str, Any], expanded_terms: List[str]) -> Tuple[int, int, float]:
    """
    Heading-prioritized scoring: heading_hits * 2 + body_hits.
    Returns (heading_hits, body_hits, score).
    """
    heading = normalize_for_match(str(clause.get("clause_heading") or ""))
    body = normalize_for_match(str(clause.get("verbatim_text") or ""))
    heading_hits = sum(1 for t in expanded_terms if t in heading)
    body_hits = sum(1 for t in expanded_terms if t in body)
    score = 2.0 * heading_hits + 1.0 * body_hits
    return heading_hits, body_hits, score


class EvidenceExplorerService:
    """
    Deterministic evidence retrieval for a single document.
    No LLM; returns snippets + metadata + debug only.
    """

    STEP_NAME = "evidence_explorer"

    def __init__(self, rag_service: RAGService, extracted_clause_store: ExtractedClauseStore):
        self.rag_service = rag_service
        self.extracted_clause_store = extracted_clause_store

    def run(self, ctx: WorkflowContext, request: EvidenceExplorerRequest) -> WorkflowContext:
        """
        Run evidence explorer. Validates mode and branches: text, clauses, or both.
        """
        if ctx.status == "failed":
            return ctx

        mode = request.mode or "text"
        if mode not in {"text", "clauses", "both"}:
            ctx.fail(
                code="INVALID_MODE",
                message="Invalid evidence explorer mode. Use 'text', 'clauses', or 'both'.",
                step=self.STEP_NAME,
                details={"mode": mode},
            )
            return ctx

        top_k = request.top_k or 25
        expanded_keywords = expand_query_terms(request.query)
        ocr_noise_detected = detect_ocr_noise(request.query)

        include_retrieval_debug = getattr(request, "debug", False)
        if mode == "both":
            response, debug = self._run_both(request.document_id, request.query, top_k, expanded_keywords, ocr_noise_detected, include_retrieval_debug)
        elif mode == "clauses":
            response, debug = self._run_clauses(request.document_id, request.query, top_k, expanded_keywords, ocr_noise_detected, include_retrieval_debug)
        else:
            response, debug = self._run_text(request.document_id, request.query, top_k, expanded_keywords, ocr_noise_detected, include_retrieval_debug)

        debug["mode"] = mode
        debug["expanded_keywords"] = expanded_keywords
        debug["ocr_noise_detected"] = ocr_noise_detected
        response.debug = debug

        ctx.add_result(
            self.STEP_NAME,
            {"request": request.model_dump(), "response": response.model_dump()},
        )
        ctx.status = "completed"
        return ctx

    def _run_text(
        self,
        document_id: str,
        query: str,
        top_k: int,
        expanded_keywords: List[str],
        ocr_noise_detected: bool,
        include_retrieval_debug: bool = False,
    ) -> Tuple[EvidenceExplorerResponse, Dict[str, Any]]:
        """
        Text mode: semantic search with explorer threshold, then lexical fallback.
        """
        explorer_threshold = settings.SIMILARITY_THRESHOLDS.get("explorer", 0.45)
        semantic_results = self.rag_service.search(
            query=query,
            top_k=top_k,
            document_id_filter=document_id,
            similarity_threshold=explorer_threshold,
        )

        chunks_searched = 0
        vector_store = getattr(self.rag_service, "vector_store", None)
        if vector_store:
            all_chunks = vector_store.get_chunks_by_document(document_id)
            chunks_searched = len(all_chunks)
        else:
            all_chunks = []

        lexical_results: List[Dict[str, Any]] = []
        if not semantic_results and expanded_keywords:
            for chunk in all_chunks:
                text_norm = normalize_for_match(chunk.get("text") or "")
                if any(kw in text_norm for kw in expanded_keywords):
                    lexical_results.append({
                        **chunk,
                        "score": 1.0,
                        "lexical_match": True,
                    })
            lexical_results = lexical_results[:top_k]

        combined: List[Dict[str, Any]] = list(semantic_results)
        seen = {(r.get("document_id"), r.get("chunk_index")) for r in combined}
        for r in lexical_results:
            key = (r.get("document_id"), r.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                combined.append(r)
        combined.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        combined = combined[:top_k]

        results: List[EvidenceExplorerResult] = []
        for r in combined:
            text = r.get("text", "")
            snippet = text[:600].rstrip() + "..." if len(text) > 600 else text
            results.append(
                EvidenceExplorerResult(
                    document_id=r.get("document_id", document_id),
                    page_number=r.get("page_number", 0),
                    chunk_index=r.get("chunk_index", 0),
                    text_snippet=snippet,
                    score=float(r.get("score", 0.0)),
                    source_type="chunk",
                    display_name=r.get("display_name"),
                    citation=r.get("citation"),
                )
            )

        debug = {
            "clauses_searched": 0,
            "chunks_searched": chunks_searched,
            "embeddings_used": True,
            "semantic_results": len(semantic_results),
            "lexical_results": len(lexical_results),
        }

        if not results:
            if include_retrieval_debug:
                debug["retrieval_debug"] = {
                    "candidate_clause_ids": [],
                    "candidate_pages": [],
                    "scores": [],
                    "query_normalized": normalize_for_match(query),
                }
            return (
                EvidenceExplorerResponse(
                    status="not_found",
                    results=[],
                    reason="not_found",
                    debug=debug,
                ),
                debug,
            )
        if include_retrieval_debug:
            debug["retrieval_debug"] = {
                "candidate_clause_ids": [f"{r.document_id}:{r.chunk_index}" for r in results],
                "candidate_pages": list(dict.fromkeys(r.page_number for r in results)),
                "scores": [r.score for r in results],
                "query_normalized": normalize_for_match(query),
            }
        return (
            EvidenceExplorerResponse(status="ok", results=results, reason="ok", debug=debug),
            debug,
        )

    def _run_clauses(
        self,
        document_id: str,
        query: str,
        top_k: int,
        expanded_keywords: List[str],
        ocr_noise_detected: bool,
        include_retrieval_debug: bool = False,
    ) -> Tuple[EvidenceExplorerResponse, Dict[str, Any]]:
        """
        Clause mode: match over extracted clauses with heading-prioritized scoring.
        """
        clauses = self.extracted_clause_store.get_document_clauses(document_id)
        clauses_searched = len(clauses)

        if not clauses:
            debug_empty = {
                "clauses_searched": 0,
                "chunks_searched": 0,
                "embeddings_used": False,
                "semantic_results": 0,
                "lexical_results": 0,
            }
            if include_retrieval_debug:
                debug_empty["retrieval_debug"] = {
                    "candidate_clause_ids": [],
                    "candidate_pages": [],
                    "scores": [],
                    "query_normalized": normalize_for_match(query),
                }
            return (
                EvidenceExplorerResponse(
                    status="not_found",
                    results=[],
                    reason="not_found",
                    debug=debug_empty,
                ),
                debug_empty,
            )

        if not expanded_keywords:
            debug_no_kw = {
                "clauses_searched": clauses_searched,
                "chunks_searched": 0,
                "embeddings_used": False,
                "semantic_results": 0,
                "lexical_results": 0,
            }
            if include_retrieval_debug:
                debug_no_kw["retrieval_debug"] = {
                    "candidate_clause_ids": [],
                    "candidate_pages": [],
                    "scores": [],
                    "query_normalized": normalize_for_match(query),
                }
            return (
                EvidenceExplorerResponse(
                    status="not_found",
                    results=[],
                    reason="not_found",
                    debug=debug_no_kw,
                ),
                debug_no_kw,
            )

        matches: List[Tuple[float, Dict[str, Any]]] = []
        for clause in clauses:
            heading_hits, body_hits, score = score_clause(clause, expanded_keywords)
            if heading_hits >= 1 or score > 0:
                matches.append((score, clause))
        matches.sort(key=lambda x: x[0], reverse=True)
        matches = matches[:top_k]

        results: List[EvidenceExplorerResult] = []
        for score, clause in matches:
            verbatim = str(clause.get("verbatim_text") or "")
            snippet = verbatim[:600].rstrip() + "..." if len(verbatim) > 600 else verbatim
            page_start = clause.get("page_start", 0)
            page_end = clause.get("page_end") or page_start
            results.append(
                EvidenceExplorerResult(
                    document_id=document_id,
                    page_number=page_start,
                    chunk_index=0,
                    text_snippet=snippet,
                    score=float(score),
                    source_type="clause",
                    display_name=None,
                    citation=None,
                    clause_id=clause.get("clause_id"),
                    page_end=page_end,
                )
            )

        debug = {
            "clauses_searched": clauses_searched,
            "chunks_searched": 0,
            "embeddings_used": False,
            "semantic_results": 0,
            "lexical_results": len(results),
        }

        if not results:
            if include_retrieval_debug:
                debug["retrieval_debug"] = {
                    "candidate_clause_ids": [],
                    "candidate_pages": [],
                    "scores": [],
                    "query_normalized": normalize_for_match(query),
                }
            return (
                EvidenceExplorerResponse(
                    status="not_found",
                    results=[],
                    reason="not_found",
                    debug=debug,
                ),
                debug,
            )
        if include_retrieval_debug:
            debug["retrieval_debug"] = {
                "candidate_clause_ids": [r.clause_id or f"{document_id}:{r.page_number}" for r in results],
                "candidate_pages": list(dict.fromkeys(r.page_number for r in results)),
                "scores": [r.score for r in results],
                "query_normalized": normalize_for_match(query),
            }
        return (
            EvidenceExplorerResponse(status="ok", results=results, reason="ok", debug=debug),
            debug,
        )

    def _run_both(
        self,
        document_id: str,
        query: str,
        top_k: int,
        expanded_keywords: List[str],
        ocr_noise_detected: bool,
        include_retrieval_debug: bool = False,
    ) -> Tuple[EvidenceExplorerResponse, Dict[str, Any]]:
        """
        Both mode: run clauses first; if any clause hits return those; else text mode.
        """
        clause_response, clause_debug = self._run_clauses(
            document_id, query, top_k, expanded_keywords, ocr_noise_detected, include_retrieval_debug
        )
        if clause_response.results:
            return clause_response, clause_debug
        return self._run_text(document_id, query, top_k, expanded_keywords, ocr_noise_detected, include_retrieval_debug)
