from __future__ import annotations

"""
Supporting retrieval tools (Phase 2): scoped semantic search and
evidence-grounded question answering over the documents in scope.

Both tools wrap RAGService unchanged. answer_question inherits the
evidence guardrails that already run inside RAGService.query, so the
executor records guardrail="inherited" for it.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.models.agent import ToolParam, ToolResult, ToolSpec

if TYPE_CHECKING:
    from backend.agent.tool_registry import ToolRegistry
    from backend.agent.tools import AgentToolServices

_SEARCH_DEFAULT_TOP_K = 5
_SUMMARY_ANSWER_CHARS = 200


def _citation_from_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Compact citation entry; text is kept so the verifier can ground claims."""
    return {
        "document_id": chunk.get("document_id"),
        "page_number": chunk.get("page_number"),
        "chunk_id": chunk.get("chunk_id"),
        "score": chunk.get("score"),
        "text": chunk.get("text", ""),
    }


def register_retrieval_tools(registry: "ToolRegistry", services: "AgentToolServices") -> None:
    def search_document(
        document_ids: List[str],
        query: str,
        jurisdiction: Optional[str] = None,
        top_k: int = _SEARCH_DEFAULT_TOP_K,
    ) -> ToolResult:
        merged: List[Dict[str, Any]] = []
        for doc_id in document_ids:
            # Retrieval always goes through document_id_filter (isolation rule).
            results = services.rag_service.search(
                query=query, top_k=top_k, document_id_filter=doc_id
            )
            merged.extend(results or [])
        merged.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        merged = merged[:top_k]

        pages = sorted(
            {c.get("page_number") for c in merged if c.get("page_number") is not None}
        )
        return ToolResult(
            tool_name="search_document",
            status="ok",
            summary=(
                f"Found {len(merged)} chunk(s) across {len(document_ids)} document(s)"
                f"{'; pages: ' + ', '.join(str(p) for p in pages[:8]) if pages else ''}."
            ),
            payload={"chunks": merged},
            citations=[_citation_from_chunk(c) for c in merged],
        )

    registry.register(
        ToolSpec(
            name="search_document",
            description=(
                "Semantic search over the documents in scope; returns the most "
                "relevant text chunks with page citations. Use to locate evidence "
                "or check whether a topic is covered before answering."
            ),
            params=[
                ToolParam(
                    name="query",
                    type="string",
                    description="What to search for, phrased as a short natural-language query.",
                ),
                ToolParam(
                    name="top_k",
                    type="integer",
                    description="Number of chunks to return (default 5).",
                    required=False,
                ),
            ],
        ),
        search_document,
    )

    def answer_question(
        document_ids: List[str],
        question: str,
        jurisdiction: Optional[str] = None,
        top_k: int = _SEARCH_DEFAULT_TOP_K,
    ) -> ToolResult:
        response = services.rag_service.query(
            query=question, top_k=top_k, document_id_filter=document_ids[0]
        )
        answer = (response or {}).get("answer") or ""
        rag_status = (response or {}).get("status", "unknown")
        confidence = (response or {}).get("confidence", "low")
        sources = (response or {}).get("sources") or []

        # A RAG refusal ("not covered in the documents") is itself the grounded
        # answer, not a tool failure — return ok so the run completes with it.
        return ToolResult(
            tool_name="answer_question",
            status="ok",
            summary=(
                f"Answer ({rag_status}, {confidence} confidence): "
                f"{answer[:_SUMMARY_ANSWER_CHARS]}"
            ),
            payload={
                "answer_text": answer,
                "rag_response": response,
                "guardrail": "inherited",
            },
            citations=[_citation_from_chunk(s) for s in sources],
        )

    registry.register(
        ToolSpec(
            name="answer_question",
            description=(
                "Answer a specific question from the first document in scope using "
                "evidence-grounded retrieval with citations. Use for factual questions "
                "about document content when no full workflow (review, memo, summary) fits."
            ),
            params=[
                ToolParam(
                    name="question",
                    type="string",
                    description="The question to answer from the document.",
                ),
                ToolParam(
                    name="top_k",
                    type="integer",
                    description="Number of chunks to retrieve (default 5).",
                    required=False,
                ),
            ],
            is_terminal=True,
        ),
        answer_question,
    )
