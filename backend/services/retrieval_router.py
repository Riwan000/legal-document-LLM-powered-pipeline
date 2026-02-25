"""
RetrievalRouter — 7-engine multi-strategy retrieval for document-scoped queries.

Engines:
  1. definition_engine    — keyword scan for unit_type="definition" + term match
  2. clause_title_engine  — BM25-simplified keyword overlap on clause_title/clause_number
  3. clause_semantic      — standard semantic search filtered to clause/clause_subchunk/None
  4. category_engine      — metadata scan for legal_category == target slug
  5. page_fallback_engine — semantic search filtered to unit_type="page_chunk"
  6. title_engine         — force-include first-page/title chunks for summary/classification
  7. binary_engine        — full-doc keyword scan + heading boost + page diversity for yes/no queries

Fused score = engine_weight × base_score.  Deduplication by chunk_id; highest wins.
All new metadata accessed via .get(field, None) for backward compatibility.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.services.vector_store import VectorStore
    from backend.services.embedding_service import EmbeddingService
    from backend.models.document import QueryClassification

logger = logging.getLogger(__name__)

# Map query_type slug → legal_category slug (for category engine)
_QUERY_TYPE_TO_CATEGORY: Dict[str, str] = {
    "termination": "termination",
    "compensation": "payment",
    "benefits": "compensation_benefits",
    "compliance": "compliance",
    "notice": "termination",
    "probation": "termination",
    "legality": "governing_law",
}

# Stop words stripped from binary questions before keyword extraction
_BINARY_STRIP_WORDS: frozenset = frozenset([
    "is", "are", "does", "can", "has", "will", "did", "do", "was", "were",
    "the", "a", "an", "this", "that", "contract", "agreement", "document",
    "it", "its", "there", "any", "in", "of", "for", "with",
])

# Detect "clause N" / "article N" / "section N.M" patterns
_CLAUSE_REF_RE = re.compile(
    r"\b(?:clause|article|section)\s+(\d+(?:\.\d+)?|[IVXLCDM]+)\b",
    re.IGNORECASE,
)


class RetrievalRouter:
    """Multi-engine retrieval router for document-scoped semantic + structural search."""

    def __init__(self, vector_store: "VectorStore", embedding_service: "EmbeddingService"):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    # ── Public entry point ──────────────────────────────────────────────────────

    def route(
        self,
        query: str,
        classification: "QueryClassification",
        document_id: Optional[str],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Route the query through applicable engines, fuse scores, and deduplicate.

        Returns list of chunk metadata dicts with 'score' and 'engines_used' keys.
        """
        query_embedding = self.embedding_service.embed_text(query)
        fused: Dict[str, Dict[str, Any]] = {}  # chunk_id → best result

        def _merge(results: List[Dict[str, Any]], weight: float, engine_name: str) -> None:
            for r in results:
                cid = r.get("chunk_id") or r.get("metadata", {}).get("chunk_id") or _chunk_key(r)
                fused_score = weight * float(r.get("score", 0.5))
                if cid not in fused or fused_score > fused[cid]["score"]:
                    entry = dict(r)
                    entry["score"] = fused_score
                    entry["engines_used"] = fused.get(cid, {}).get("engines_used", []) + [engine_name]
                    fused[cid] = entry
                else:
                    fused[cid]["engines_used"].append(engine_name)

        # ── Engine selection ────────────────────────────────────────────────────
        query_types = getattr(classification, "query_types", []) or []

        # Engine 1 — Definition (triggered by definition_lookup intent or phrase)
        if "definition_lookup" in query_types or self._has_definition_phrase(query):
            res = self._definition_engine(query, query_embedding, document_id, top_k)
            _merge(res, 1.0, "definition_engine")

        # Engine 2 — Clause title (triggered by "clause N" / "article N" / "section N" reference)
        clause_ref = _CLAUSE_REF_RE.search(query)
        if clause_ref:
            res = self._clause_title_engine(query, clause_ref.group(1), document_id, top_k)
            _merge(res, 0.9, "clause_title_engine")

        # Engine 4 — Category batch (triggered by category query + "all clauses"/"summarize all")
        category_slug = self._query_type_to_category(query_types)
        if category_slug and self._is_category_batch_query(query):
            res = self._category_engine(query_embedding, category_slug, document_id, top_k)
            _merge(res, 0.8, "category_engine")

        # Engine 6 — Title/first-page (triggered by summary or classification intent)
        if "summary" in query_types or "classification" in query_types:
            res = self._title_engine(document_id, top_k)
            _merge(res, 0.9, "title_engine")

        # Engine 7 — Binary question (triggered by binary intent)
        if "binary" in query_types:
            res = self._binary_engine(query, document_id, top_k)
            _merge(res, 0.85, "binary_engine")

        # Engine 3 — Clause semantic (always, primary)
        res = self._clause_semantic_engine(query_embedding, document_id, top_k)
        _merge(res, 1.0, "clause_semantic_engine")

        # Engine 5 — Page fallback (always, low weight backstop)
        res = self._page_fallback_engine(query_embedding, document_id, top_k)
        _merge(res, 0.4, "page_fallback_engine")

        # Sort by fused score, return top_k
        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    # ── Engine implementations ──────────────────────────────────────────────────

    def _definition_engine(
        self,
        query: str,
        query_embedding: np.ndarray,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Scan for unit_type='definition' chunks and score by term match."""
        all_meta = self.vector_store.metadata
        results = []
        query_lower = query.lower()

        for entry in all_meta:
            if document_id and entry.get("document_id") != document_id:
                continue
            if entry.get("unit_type") != "definition":
                continue
            text = (entry.get("text") or "").lower()
            # Simple term overlap score
            query_words = set(re.findall(r"\b\w{3,}\b", query_lower)) - {"what", "does", "mean", "the", "define"}
            hits = sum(1 for w in query_words if w in text)
            score = min(1.0, hits / max(len(query_words), 1))
            if score > 0:
                results.append({**entry, "score": score})

        # Semantic fallback if keyword scan returns nothing
        if not results:
            raw = self.vector_store.search(
                query_embedding, top_k=top_k, document_id_filter=document_id,
                similarity_threshold=None,
            )
            results = [r for r in raw if r.get("unit_type") == "definition"]

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]

    def _clause_title_engine(
        self,
        query: str,
        ref_number: str,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """BM25-simplified: match clause_number / clause_title against query."""
        all_meta = self.vector_store.metadata
        results = []
        ref_lower = ref_number.lower().strip()
        query_lower = query.lower()

        for entry in all_meta:
            if document_id and entry.get("document_id") != document_id:
                continue
            clause_num = (entry.get("clause_number") or "").lower().strip()
            clause_title = (entry.get("clause_title") or entry.get("clause_heading") or "").lower()
            if not clause_num and not clause_title:
                continue

            # Exact clause number match → score 1.0
            if clause_num == ref_lower:
                results.append({**entry, "score": 1.0})
                continue

            # Partial title overlap
            query_words = set(re.findall(r"\b\w{3,}\b", query_lower))
            title_words = set(re.findall(r"\b\w{3,}\b", clause_title))
            overlap = query_words & title_words
            if overlap:
                score = len(overlap) / max(len(query_words), 1)
                results.append({**entry, "score": score})

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]

    def _clause_semantic_engine(
        self,
        query_embedding: np.ndarray,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Standard semantic search filtered to clause/clause_subchunk/None unit_types (covers legacy)."""
        raw = self.vector_store.search(
            query_embedding,
            top_k=top_k * 2,  # fetch extra, then filter
            document_id_filter=document_id,
            similarity_threshold=None,
        )
        results = []
        for r in raw:
            ut = r.get("unit_type")
            if ut in ("clause", "clause_subchunk", None):
                results.append(r)
        return results[:top_k]

    def _category_engine(
        self,
        query_embedding: np.ndarray,
        category_slug: str,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Scan metadata for matching legal_category; semantic rerank where possible."""
        all_meta = self.vector_store.metadata
        results = []
        for entry in all_meta:
            if document_id and entry.get("document_id") != document_id:
                continue
            if entry.get("legal_category") == category_slug:
                results.append({**entry, "score": 0.7})
        # For old docs with no legal_category: return empty (Engine 3 covers them)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]

    def _page_fallback_engine(
        self,
        query_embedding: np.ndarray,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Semantic search filtered to unit_type='page_chunk'."""
        raw = self.vector_store.search(
            query_embedding,
            top_k=top_k * 2,
            document_id_filter=document_id,
            similarity_threshold=None,
        )
        return [r for r in raw if r.get("unit_type") == "page_chunk"][:top_k]

    def _title_engine(
        self,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Force-include document title and first-page chunks for summary/classification queries.

        Selection criteria (any one of):
          - unit_type == "title"
          - page_number <= 1
          - chunk_index <= 2  (near start of document)

        Fixed score of 0.85 ensures inclusion alongside semantic results without
        outranking a directly relevant clause (which scores ~0.9 × 1.0 = 0.9).
        """
        results = []
        for entry in self.vector_store.metadata:
            if document_id and entry.get("document_id") != document_id:
                continue
            try:
                page = int(entry.get("page_number") or 999)
            except (TypeError, ValueError):
                page = 999
            try:
                idx = int(entry.get("chunk_index") or 999)
            except (TypeError, ValueError):
                idx = 999
            unit_type = entry.get("unit_type")
            if unit_type == "title" or page <= 1 or idx <= 2:
                results.append({**entry, "score": 0.85})

        # Stable sort: title chunks first, then by page, then by chunk index
        results.sort(key=lambda x: (
            0 if x.get("unit_type") == "title" else 1,
            int(x.get("page_number") or 999),
            int(x.get("chunk_index") or 999),
        ))
        return results[:top_k]

    def _binary_engine(
        self,
        query: str,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Full-document keyword scan for binary (yes/no) questions.

        Strategy:
          1. Extract content keywords (strip binary stop words).
          2. Scan all chunks for keyword containment.
          3. Apply +0.2 heading boost for clause/section chunks.
          4. Enforce page diversity: max 2 chunks per page_number.
          5. Return top-k across diverse pages.
        """
        keywords = [
            w for w in re.findall(r"\b\w{3,}\b", query.lower())
            if w not in _BINARY_STRIP_WORDS
        ]
        if not keywords:
            return []

        scored: List[Dict[str, Any]] = []
        for entry in self.vector_store.metadata:
            if document_id and entry.get("document_id") != document_id:
                continue
            text = (entry.get("text") or "").lower()
            if not text:
                continue
            hits = sum(1 for kw in keywords if kw in text)
            if hits == 0:
                continue
            score = hits / len(keywords)
            unit_type = entry.get("unit_type")
            if unit_type in ("clause", "section") or entry.get("clause_number") is not None:
                score = min(1.0, score + 0.2)  # heading boost
            scored.append({**entry, "score": score})

        scored.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Page diversity: max 2 chunks per page_number
        page_counts: Dict[int, int] = {}
        diverse: List[Dict[str, Any]] = []
        for entry in scored:
            try:
                pk = int(entry.get("page_number") or -1)
            except (TypeError, ValueError):
                pk = -1
            if page_counts.get(pk, 0) < 2:
                diverse.append(entry)
                page_counts[pk] = page_counts.get(pk, 0) + 1
            if len(diverse) >= top_k:
                break
        return diverse

    # ── Helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _has_definition_phrase(query: str) -> bool:
        lower = query.lower()
        return any(p in lower for p in ("what does", "definition of", "defined as", "meaning of"))

    @staticmethod
    def _is_category_batch_query(query: str) -> bool:
        lower = query.lower()
        return any(p in lower for p in ("all clauses", "summarize all", "list all", "show all"))

    @staticmethod
    def _query_type_to_category(query_types: List[str]) -> Optional[str]:
        for qt in query_types:
            if qt in _QUERY_TYPE_TO_CATEGORY:
                return _QUERY_TYPE_TO_CATEGORY[qt]
        return None


def _chunk_key(entry: Dict[str, Any]) -> str:
    """Stable deduplication key when chunk_id is absent."""
    return f"{entry.get('document_id')}:{entry.get('page_number')}:{entry.get('chunk_index')}"
