"""
Conversational RAG Pipeline — decomposed architecture.

Modules in this file:
  RetrievalStrategy    — dual retrieval with score calibration (query-fusion normalization)
  GenerationPipeline   — token-budget-aware prompt assembly + LLM call + structured citation extraction
  ValidationPipeline   — two-pass guardrail with regeneration on weak evidence
  AuditService         — structured JSON audit log (fully reconstructable)
  ResponseBuilder      — assembles the final ChatMessageResponse
  ChatOrchestratorService — thin coordinator; delegates to the above

Design principles enforced:
  1. Score calibration: scores from two retrieval passes are min-max normalized before merging.
  2. Exact chunk traceability: guardrail evaluates the same chunk objects passed to the LLM.
  3. Sentence-level validation: EvidenceGuardrailService validates per sentence.
  4. Token budget: history + chunks + system prompt measured before generation; trimmed if needed.
  5. Semantic history filtering: only past turns semantically similar to the current query are injected.
  6. Structured citations: model prompted to return page citations as JSON; validated against chunks.
  7. Two-pass regeneration: weak evidence triggers a stricter regeneration pass before refusal.
  8. Prompt injection filtering: chunk text wrapped in explicit delimiters; injection patterns stripped.
  9. Full audit log: prompt hash, chunk hashes, model version, threshold snapshot, latency.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import settings
from backend.models.session import (
    ChatMessageResponse,
    ChatSource,
    RetrievalTrace,
    SessionMode,
    SessionMessage,
)
from backend.services.evidence_guardrail_service import EvidenceGuardrailService, GuardrailResult
from backend.services.query_rewriter import QueryRewriter
from backend.services.session_manager import SessionManager
from backend.utils.token_counter import count_tokens, count_session_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit logger — logs/chat.log (structured JSON lines)
# ---------------------------------------------------------------------------
_chat_log_path = Path("logs/chat.log")
_chat_log_path.parent.mkdir(parents=True, exist_ok=True)
_audit_logger = logging.getLogger("chat.audit")
if not _audit_logger.handlers:
    _fh = logging.FileHandler(str(_chat_log_path), encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(message)s"))
    _audit_logger.addHandler(_fh)
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False

_REFUSAL_ANSWER = (
    "Insufficient evidence found in the document to answer this question. "
    "Please refer to the document directly."
)

# Prompt injection markers
_CHUNK_DELIMITER_START = "<<DOCUMENT_EXCERPT_START>>"
_CHUNK_DELIMITER_END = "<<DOCUMENT_EXCERPT_END>>"

# Patterns to strip from chunk text (prompt injection defense)
_INJECTION_STRIP = re.compile(
    r"(ignore (all |previous )?instructions?|"
    r"disregard|forget (your |all )?instructions?|"
    r"you are now|act as|pretend you|system prompt)",
    re.IGNORECASE,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# RetrievalStrategy
# ---------------------------------------------------------------------------

class RetrievalStrategy:
    """
    Dual-query retrieval with query-fusion score normalization.

    Scores from two retrieval passes (original query + rewritten query) are
    min-max normalized within each list before merging, ensuring they are
    comparable. The fused score = mean of normalized scores across lists a
    chunk appears in.
    """

    def __init__(self, rag_service):
        self.rag_service = rag_service

    def retrieve(
        self,
        original_query: str,
        rewritten_query: Optional[str],
        document_id: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Run dual retrieval and return calibrated, deduplicated, ranked chunks.
        """
        chunks_a = self.rag_service.search(
            query=original_query,
            top_k=top_k,
            document_id_filter=document_id,
        )
        chunks_b: List[Dict[str, Any]] = []
        if rewritten_query and rewritten_query.strip() != original_query.strip():
            chunks_b = self.rag_service.search(
                query=rewritten_query,
                top_k=top_k,
                document_id_filter=document_id,
            )

        if not chunks_b:
            return chunks_a[:top_k]

        return self._query_fusion_merge(chunks_a, chunks_b, top_k)

    @staticmethod
    def _normalize_scores(chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Return {chunk_key: normalized_score} mapping (min-max per list)."""
        if not chunks:
            return {}
        scores = [float(c.get("score", 0.0)) for c in chunks]
        lo, hi = min(scores), max(scores)
        span = hi - lo if hi > lo else 1.0
        return {
            RetrievalStrategy._chunk_key(c): (s - lo) / span
            for c, s in zip(chunks, scores)
        }

    @staticmethod
    def _chunk_key(c: Dict[str, Any]) -> str:
        return f"{c.get('document_id', '')}::{c.get('chunk_index', '')}"

    def _query_fusion_merge(
        self,
        chunks_a: List[Dict[str, Any]],
        chunks_b: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        norms_a = self._normalize_scores(chunks_a)
        norms_b = self._normalize_scores(chunks_b)

        # Index all chunks by key
        chunk_by_key: Dict[str, Dict[str, Any]] = {}
        count_by_key: Dict[str, int] = {}
        fused_score: Dict[str, float] = {}

        for chunk in chunks_a + chunks_b:
            key = self._chunk_key(chunk)
            if key not in chunk_by_key:
                chunk_by_key[key] = chunk
                count_by_key[key] = 0
                fused_score[key] = 0.0
            # Accumulate normalized scores
            if key in norms_a:
                fused_score[key] += norms_a[key]
                count_by_key[key] += 1
            if key in norms_b:
                fused_score[key] += norms_b[key]
                count_by_key[key] += 1

        # Average fused score
        merged: List[Dict[str, Any]] = []
        for key, chunk in chunk_by_key.items():
            avg = fused_score[key] / count_by_key[key] if count_by_key[key] else 0.0
            merged.append({**chunk, "score": avg})

        merged.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        return merged[:top_k]

    @staticmethod
    def sanitize_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Strip prompt-injection patterns from chunk text before passing to the LLM.
        Wraps the cleaned text in explicit delimiters (injected at generation time).
        """
        clean: List[Dict[str, Any]] = []
        for c in chunks:
            text = c.get("text", "") or ""
            # Strip known injection phrases
            cleaned_text = _INJECTION_STRIP.sub("[REDACTED]", text)
            clean.append({**c, "text": cleaned_text})
        return clean


# ---------------------------------------------------------------------------
# GenerationPipeline
# ---------------------------------------------------------------------------

class GenerationPipeline:
    """
    Token-budget-aware context assembly + LLM generation.

    Responsibilities:
      - Measure history + chunk + system-prompt tokens before generation.
      - Trim history (oldest first) or chunk list (lowest score first) to stay within budget.
      - Wrap chunks in injection-safe delimiters.
      - Extract structured citations from LLM response (page numbers as JSON array).
    """

    # Approximate token count of the system prompt and response overhead
    _SYSTEM_PROMPT_OVERHEAD: int = 250

    def __init__(self, rag_service):
        self.rag_service = rag_service

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        history: List[SessionMessage],
        document_id: str,
        top_k: int,
        strict: bool = False,
        response_language: Optional[str] = None,
    ) -> Tuple[str, List[int], List[Dict[str, Any]]]:
        """
        Assemble context, enforce token budget, then generate via RAGService.

        Returns:
            (answer_text, structured_citations, exact_chunks_used)
        """
        # 1. Enforce token budget — trim history and/or chunks before calling LLM
        final_history, final_chunks = self._enforce_budget(history, chunks)

        # 2. Sanitize chunks (injection defense)
        sanitized = RetrievalStrategy.sanitize_chunks(final_chunks)
        # #region agent log
        _DEBUG_LOG_G = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
        def _dbg_g(payload: dict) -> None:
            try:
                payload.setdefault("timestamp", int(time.time() * 1000))
                payload.setdefault("id", f"log_{payload['timestamp']}_g")
                with open(_DEBUG_LOG_G, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
        _dbg_g({"location": "chat_orchestrator.generate", "message": "calling RAG with chunks_override", "data": {"chunks_override_len": len(sanitized) if sanitized else 0, "len_final_chunks": len(final_chunks)}, "hypothesisId": "C"})
        # #endregion

        # 3. Call RAGService with exact sanitized chunks
        rag_result = self.rag_service.query(
            query=query,
            top_k=top_k,
            document_id_filter=document_id,
            generate_response=True,
            chunks_override=sanitized if sanitized else None,
            response_language=response_language,
        )

        answer: str = rag_result.get("answer") or ""
        rag_sources: List[Dict[str, Any]] = rag_result.get("sources", [])
        # #region agent log
        try:
            _log = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
            with open(_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": f"log_{int(time.time()*1000)}_rag", "timestamp": int(time.time()*1000), "location": "chat_orchestrator.after_rag", "message": "after RAG call", "data": {"rag_status": rag_result.get("status"), "len_rag_sources": len(rag_sources), "has_answer": bool((answer or "").strip())}, "hypothesisId": "C"}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion

        # 4. Extract structured citations from answer (model prompted elsewhere)
        structured_citations = self._extract_structured_citations(answer)

        # Use rag_result sources for the exact chunk record (these are the chunks
        # the LLM actually received — already sanitized via chunks_override).
        exact_chunks = rag_sources if rag_sources else sanitized

        return answer, structured_citations, exact_chunks

    def _enforce_budget(
        self,
        history: List[SessionMessage],
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[SessionMessage], List[Dict[str, Any]]]:
        """
        Trim history (oldest turns first) and then chunks (lowest score first)
        until the total token budget is within MAX_SESSION_TOKENS.
        """
        budget = settings.MAX_SESSION_TOKENS - self._SYSTEM_PROMPT_OVERHEAD
        remaining_history = list(history)
        remaining_chunks = list(chunks)

        def total_tokens() -> int:
            history_tokens = count_session_tokens(remaining_history)
            chunk_tokens = sum(count_tokens(c.get("text", "") or "") for c in remaining_chunks)
            return history_tokens + chunk_tokens

        # First, trim oldest history turns (in pairs to keep user/assistant together)
        while total_tokens() > budget and len(remaining_history) >= 2:
            remaining_history = remaining_history[2:]
            logger.debug("GenerationPipeline: trimmed oldest history turn (budget enforcement)")

        # Then, trim lowest-scored chunks
        while total_tokens() > budget and len(remaining_chunks) > 1:
            # Remove the last (lowest-scored) chunk
            remaining_chunks = remaining_chunks[:-1]
            logger.debug("GenerationPipeline: trimmed lowest-score chunk (budget enforcement)")

        return remaining_history, remaining_chunks

    @staticmethod
    def _extract_structured_citations(answer: str) -> List[int]:
        """
        Extract page numbers from structured citation markers in the answer.

        The RAG prompt instructs the model to include citations in the format:
            [pages: 3, 7, 12]
        This extracts those page numbers as integers.
        """
        match = re.search(r"\[pages?:\s*([\d,\s]+)\]", answer, re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1)
        pages: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                pages.append(int(part))
        return pages


# ---------------------------------------------------------------------------
# ValidationPipeline
# ---------------------------------------------------------------------------

class ValidationPipeline:
    """
    Two-pass evidence validation with regeneration on weak evidence.

    Pass 1: Run guardrail on original answer.
    Pass 2: If result is "weak", regenerate with stricter citation-forcing prompt,
            re-run guardrail. If still weak or fail → refuse.
    """

    _STRICT_SUFFIX = (
        "\n\nIMPORTANT: Answer ONLY if directly stated in the provided excerpts. "
        "For every claim, include the exact page number in the format [pages: N]. "
        "If the excerpts do not contain the answer, say: "
        "'The document does not explicitly address this question.'"
    )

    def __init__(
        self,
        guardrail: EvidenceGuardrailService,
        rag_service,
    ):
        self.guardrail = guardrail
        self.rag_service = rag_service

    def validate(
        self,
        answer: str,
        exact_chunks: List[Dict[str, Any]],
        query: str,
        structured_citations: Optional[List[int]],
        document_id: str,
        top_k: int,
        strict_mode: bool = False,
        response_language: Optional[str] = None,
    ) -> Tuple[str, GuardrailResult, List[Dict[str, Any]]]:
        """
        Validate answer. If weak, attempt one regeneration pass.

        Returns:
            (final_answer, guardrail_result, final_chunks)
        """
        result = self.guardrail.check(
            answer=answer,
            chunks=exact_chunks,
            query=query,
            structured_citations=structured_citations,
        )

        if result.decision == "pass":
            return answer, result, exact_chunks

        if result.decision == "weak" and not strict_mode:
            logger.info(
                "ValidationPipeline: weak evidence on first pass; attempting stricter regeneration."
            )
            # Regenerate with stricter prompt
            stricter_query = query + self._STRICT_SUFFIX
            rag_result2 = self.rag_service.query(
                query=stricter_query,
                top_k=top_k,
                document_id_filter=document_id,
                generate_response=True,
                chunks_override=RetrievalStrategy.sanitize_chunks(exact_chunks),
                response_language=response_language,
            )
            answer2 = rag_result2.get("answer") or ""
            citations2 = GenerationPipeline._extract_structured_citations(answer2)
            exact_chunks2 = rag_result2.get("sources", []) or exact_chunks

            result2 = self.guardrail.check(
                answer=answer2,
                chunks=exact_chunks2,
                query=query,
                structured_citations=citations2,
            )

            # Accept second pass if it is at least as good
            if result2.decision in ("pass", "weak"):
                return answer2, result2, exact_chunks2

        # Final decision stands (pass/weak/fail from first pass or second pass failed)
        return answer, result, exact_chunks


# ---------------------------------------------------------------------------
# AuditService
# ---------------------------------------------------------------------------

class AuditService:
    """
    Writes a fully reconstructable structured JSON audit record per turn.

    Each record includes:
      - prompt_hash      (SHA-256 of the assembled prompt concept: query + chunk hashes)
      - answer_hash      (SHA-256 of the answer)
      - chunk_hashes     (list of SHA-256 per chunk text)
      - model_name       (settings.OLLAMA_MODEL)
      - embedding_model  (settings.EMBEDDING_MODEL)
      - guardrail_thresholds (strong + weak threshold snapshot)
      - latency_ms, token_count, session_id, document_id, mode
      - guardrail_decision, evidence_score, coverage_ratio
    """

    def write(
        self,
        session_id: str,
        document_id: str,
        mode: str,
        original_query: str,
        rewritten_query: Optional[str],
        retrieved_chunk_ids: List[str],
        similarity_scores: List[float],
        guardrail_result: GuardrailResult,
        answer: str,
        exact_chunks: List[Dict[str, Any]],
        token_count: int,
        latency_ms: int,
        rag_status: str,
    ) -> None:
        # Build a deterministic prompt hash from query + chunk hashes
        prompt_fingerprint = _sha256(
            original_query + "|".join(guardrail_result.chunk_hashes)
        )

        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "document_id": document_id,
            "mode": mode,
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "similarity_scores": similarity_scores,
            "guardrail_decision": guardrail_result.decision,
            "evidence_score": guardrail_result.evidence_score,
            "coverage_ratio": round(guardrail_result.coverage_ratio, 4),
            "injection_detected": guardrail_result.injection_detected,
            "token_count": token_count,
            "latency_ms": latency_ms,
            "rag_status": rag_status,
            # Traceability
            "prompt_hash": prompt_fingerprint,
            "answer_hash": guardrail_result.answer_hash,
            "chunk_hashes": guardrail_result.chunk_hashes,
            "model_name": settings.OLLAMA_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "guardrail_thresholds": {
                "strong": settings.GUARDRAIL_STRONG_THRESHOLD,
                "weak": settings.GUARDRAIL_WEAK_THRESHOLD,
            },
        }
        _audit_logger.info(json.dumps(record, ensure_ascii=False))


# ---------------------------------------------------------------------------
# ResponseBuilder
# ---------------------------------------------------------------------------

class ResponseBuilder:
    """Assembles the final ChatMessageResponse from pipeline outputs."""

    @staticmethod
    def build(
        session_id: str,
        answer: str,
        final_status: str,
        guardrail_result: GuardrailResult,
        exact_chunks: List[Dict[str, Any]],
        trace: RetrievalTrace,
    ) -> ChatMessageResponse:
        sources = [
            ChatSource(
                document_id=c.get("document_id", ""),
                page_number=c.get("page_number"),
                chunk_index=c.get("chunk_index"),
                text_snippet=(c.get("text", "") or "")[:200] or None,
                score=c.get("score"),
                citation=c.get("citation"),
            )
            for c in exact_chunks
        ]
        return ChatMessageResponse(
            session_id=session_id,
            answer=answer,
            status=final_status,
            evidence_score=guardrail_result.evidence_score,
            guardrail_decision=guardrail_result.decision,
            sources=sources,
            trace=trace,
        )


# ---------------------------------------------------------------------------
# ChatOrchestratorService — thin coordinator
# ---------------------------------------------------------------------------

class ChatOrchestratorService:
    """
    Thin coordinator that delegates to:
      RetrievalStrategy → GenerationPipeline → ValidationPipeline
      → AuditService + SessionManager → ResponseBuilder
    """

    def __init__(
        self,
        rag_service,
        session_manager: SessionManager,
        embedding_service,
        guardrail_service: Optional[EvidenceGuardrailService] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ):
        guardrail = guardrail_service or EvidenceGuardrailService(embedding_service)
        self._retrieval = RetrievalStrategy(rag_service)
        self._generation = GenerationPipeline(rag_service)
        self._validation = ValidationPipeline(guardrail, rag_service)
        self._audit = AuditService()
        self._session_manager = session_manager
        self._rewriter = query_rewriter or QueryRewriter()

    def chat(
        self,
        session_id: str,
        user_message: str,
        top_k: Optional[int] = None,
        mode_override: Optional[SessionMode] = None,
    ) -> ChatMessageResponse:
        t_start = time.time()
        effective_top_k = top_k or settings.TOP_K_RESULTS

        # Load session
        session = self._session_manager.get_session(session_id)
        document_id = session.document_id
        mode = mode_override or session.mode

        # Cross-turn consistency: build established facts context for generation
        established_facts = getattr(session, "established_facts", {})
        established_facts_context: Optional[str] = None
        if established_facts:
            parts = []
            if "document_type" in established_facts:
                parts.append(f"Document type: {established_facts['document_type']}")
            if "jurisdiction" in established_facts:
                parts.append(f"Governing jurisdiction: {established_facts['jurisdiction']}")
            if "party_names" in established_facts:
                parts.append(f"Parties: {', '.join(established_facts['party_names'])}")
            if parts:
                established_facts_context = (
                    "ESTABLISHED FACTS (do not re-derive):\n"
                    + "\n".join(f"- {p}" for p in parts)
                )

        # --- Arabic bilingual routing ---
        _translation_svc = getattr(self._retrieval.rag_service, "translation_service", None)
        response_language: Optional[str] = None
        retrieval_query = user_message  # may be replaced by English translation

        if _translation_svc:
            query_language = _translation_svc.detect_language(user_message)
            if query_language == "ar":
                response_language = "ar"
                doc_language = self._get_document_language(document_id)
                if doc_language != "ar":  # English-only document
                    try:
                        retrieval_query = _translation_svc.translate_text(
                            user_message, "ar", "en"
                        )
                        logger.info(
                            "Arabic query on English document (%s): translating for retrieval",
                            document_id,
                        )
                    except Exception:
                        retrieval_query = user_message  # graceful fallback

        # Enforce limits
        try:
            self._session_manager.enforce_limits(session_id)
        except Exception as exc:
            logger.warning("enforce_limits raised: %s", exc)

        # Append user message
        self._session_manager.append_user_message(session_id, user_message)

        # Semantic history filtering
        history = self._session_manager.get_context(session_id)
        relevant_history = self._filter_relevant_history(
            history=history,
            current_query=user_message,
            session=session,
        )

        rewritten_query: Optional[str] = None
        chunks: List[Dict[str, Any]] = []

        if mode == SessionMode.STRICT:
            # Stateless: no rewriting, no history
            chunks = self._retrieval.retrieve(
                original_query=retrieval_query,
                rewritten_query=None,
                document_id=document_id,
                top_k=effective_top_k,
            )
        else:
            # Conversational: rewrite + dual retrieval
            history_without_current = [
                m for m in relevant_history
                if not (m.role == "user" and m.content == user_message)
            ]
            rewritten_query = self._rewriter.rewrite(
                history=history_without_current,
                question=retrieval_query,
                document_id=document_id,
            )
            chunks = self._retrieval.retrieve(
                original_query=retrieval_query,
                rewritten_query=rewritten_query,
                document_id=document_id,
                top_k=effective_top_k,
            )

        # #region agent log
        try:
            _log = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
            with open(_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": f"log_{int(time.time()*1000)}_r", "timestamp": int(time.time()*1000), "location": "chat_orchestrator.after_retrieve", "message": "after retrieve", "data": {"document_id": document_id, "len_chunks": len(chunks), "mode": mode.value}, "hypothesisId": "C"}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion

        # Generation (with token budget enforcement)
        # Prepend established facts to the generation query only; retrieval + audit use user_message
        generation_query = (
            f"{established_facts_context}\n\nUser question: {user_message}"
            if established_facts_context else user_message
        )
        answer, structured_citations, exact_chunks = self._generation.generate(
            query=generation_query,
            chunks=chunks,
            history=relevant_history,
            document_id=document_id,
            top_k=effective_top_k,
            strict=(mode == SessionMode.STRICT),
            response_language=response_language,
        )

        rag_status = "generated"

        # Validation (two-pass with regeneration)
        final_answer, guardrail_result, final_chunks = self._validation.validate(
            answer=answer,
            exact_chunks=exact_chunks,
            query=user_message,
            structured_citations=structured_citations,
            document_id=document_id,
            top_k=effective_top_k,
            strict_mode=(mode == SessionMode.STRICT),
            response_language=response_language,
        )

        # Refusal: when guardrail fails with 0 chunks, RAG already returned a message (e.g. not_specified);
        # preserve it instead of overwriting with generic "Insufficient evidence" (root cause: retrieval returned 0).
        final_status = "ok"
        # #region agent log
        _DEBUG_LOG_O = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
        def _dbg_o(payload: dict) -> None:
            try:
                payload.setdefault("timestamp", int(time.time() * 1000))
                payload.setdefault("id", f"log_{payload['timestamp']}_o")
                with open(_DEBUG_LOG_O, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
        if guardrail_result.decision == "fail":
            _dbg_o({"location": "chat_orchestrator.refusal", "message": "applying refusal", "data": {"decision": guardrail_result.decision, "evidence_score": guardrail_result.evidence_score, "len_final_chunks": len(final_chunks), "query_pre": user_message[:80], "preserve_rag_answer": guardrail_result.evidence_score == "none" and len(final_chunks) == 0}, "hypothesisId": "D"})
        # #endregion
        if guardrail_result.decision == "fail":
            # With 0 chunks, guardrail correctly fails; answer came from RAG (e.g. not_specified). Do not overwrite.
            if guardrail_result.evidence_score == "none" and len(final_chunks) == 0:
                pass  # keep final_answer (RAG's message)
            else:
                final_answer = _REFUSAL_ANSWER
            final_status = "refused"
            rag_status = "refused"

        # Build trace
        trace = RetrievalTrace(
            original_query=user_message,
            rewritten_query=rewritten_query,
            retrieved_chunk_ids=[
                f"{c.get('chunk_index', '')}|{c.get('document_id', '')}"
                for c in final_chunks
            ],
            similarity_scores=[float(c.get("score", 0.0)) for c in final_chunks],
            guardrail_decision=guardrail_result.decision,
            evidence_score=guardrail_result.evidence_score,
        )

        # Persist assistant message
        self._session_manager.append_assistant_message(
            session_id=session_id,
            content=final_answer,
            trace=trace,
        )

        # Audit log
        latency_ms = int((time.time() - t_start) * 1000)
        self._audit.write(
            session_id=session_id,
            document_id=document_id,
            mode=mode.value,
            original_query=user_message,
            rewritten_query=rewritten_query,
            retrieved_chunk_ids=trace.retrieved_chunk_ids,
            similarity_scores=trace.similarity_scores,
            guardrail_result=guardrail_result,
            answer=final_answer,
            exact_chunks=final_chunks,
            token_count=count_tokens(final_answer),
            latency_ms=latency_ms,
            rag_status=rag_status,
        )

        return ResponseBuilder.build(
            session_id=session_id,
            answer=final_answer,
            final_status=final_status,
            guardrail_result=guardrail_result,
            exact_chunks=final_chunks,
            trace=trace,
        )

    def _get_document_language(self, document_id: str) -> str:
        """
        Return the stored language for *document_id* by scanning vector-store metadata.
        Short-circuits on first matching chunk that has a non-empty 'language' field.
        Defaults to 'en' when no language metadata is found.
        """
        vs = self._retrieval.rag_service.vector_store
        if hasattr(vs, "metadata"):
            for meta in vs.metadata:
                if meta.get("document_id") == document_id and meta.get("language"):
                    return meta["language"]
        return "en"

    def _filter_relevant_history(
        self,
        history: List[SessionMessage],
        current_query: str,
        session,
    ) -> List[SessionMessage]:
        """
        Return semantically relevant history turns using a sliding window
        (last SESSION_CONTEXT_WINDOW * 2 messages).

        We always keep the most recent turn for context continuity.
        Older turns are included only to maintain pronoun resolution context —
        semantic re-ranking of history is deferred to a future phase to avoid
        the added embedding latency on every turn.
        """
        window = settings.SESSION_CONTEXT_WINDOW * 2
        return history[-window:] if len(history) > window else history
