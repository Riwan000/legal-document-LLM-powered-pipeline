"""
EvidenceGuardrailService — sentence-level evidence validation for chat answers.

Distinct from guardrails.py (which checks prescriptive language).
This service validates evidentiary support — ensuring every claim in the answer
is grounded in the exact document chunks that were used during generation.

Key design principles:
  - Evaluate per-sentence, not per-answer (avoids partial-hallucination blind spots).
  - Accept only the exact chunks passed into generation (no re-retrieval).
  - Structured citation validation: match chunk_id/page_number, not string fragments.
  - Two-pass regeneration hook: caller can invoke with a stricter prompt if result is "weak".

Decision matrix (based on sentence coverage ratio):
  ratio ≥ GUARDRAIL_STRONG_THRESHOLD  → evidence_score = "strong"  → decision = "pass"
  ratio ≥ GUARDRAIL_WEAK_THRESHOLD    → evidence_score = "moderate" → decision = "weak"
  ratio > 0.0 (but below weak)        → evidence_score = "weak"     → decision = "weak"
  ratio == 0.0                        → evidence_score = "none"     → decision = "fail"
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config import settings

logger = logging.getLogger(__name__)

# #region agent log
_DEBUG_LOG = (Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log")
def _dbg(payload: dict) -> None:
    try:
        payload.setdefault("timestamp", int(time.time() * 1000))
        payload.setdefault("id", f"log_{payload['timestamp']}_g")
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion

# Instruction-like phrases that should be detected in retrieved content (injection signals)
_INJECTION_PATTERNS = re.compile(
    r"(ignore (all |previous )?instructions?|"
    r"disregard|forget (your |all )?instructions?|"
    r"you are now|act as|pretend you|system prompt)",
    re.IGNORECASE,
)

# Sentence splitter (handles . ? ! followed by space or end of string, respects abbreviations)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class SentenceValidation:
    sentence: str
    max_similarity: float
    supporting_chunk_id: Optional[str]  # chunk_index|document_id of best-matching chunk
    grounded: bool                       # True if max_similarity ≥ GUARDRAIL_WEAK_THRESHOLD


@dataclass
class GuardrailResult:
    decision: str        # "pass" | "weak" | "fail"
    evidence_score: str  # "strong" | "moderate" | "weak" | "none"
    coverage_ratio: float = 0.0          # fraction of sentences that are grounded
    sentence_results: List[SentenceValidation] = field(default_factory=list)
    injection_detected: bool = False
    reason: str = ""
    # Traceability fields
    chunk_hashes: List[str] = field(default_factory=list)    # SHA-256 of each chunk text
    answer_hash: str = ""                                    # SHA-256 of the answer


class EvidenceGuardrailService:
    """
    Post-generation evidence guardrail.

    Validates that every sentence in the answer is grounded in one of the
    EXACT chunks that were passed into the generation step (chunk traceability).
    """

    def __init__(self, embedding_service):
        """
        Args:
            embedding_service: EmbeddingService instance with embed_text(str) -> np.ndarray.
        """
        self.embedding_service = embedding_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
        query: str,
        structured_citations: Optional[List[int]] = None,
    ) -> GuardrailResult:
        """
        Validate `answer` against the EXACT `chunks` passed to generation.

        Args:
            answer:               The LLM-generated answer string.
            chunks:               The exact chunks passed to the LLM (not re-retrieved).
            query:                The user query (for logging context).
            structured_citations: Optional list of page_numbers the model cited.
                                  When provided, used instead of string matching.

        Returns:
            GuardrailResult with full traceability metadata.
        """
        answer_hash = _sha256(answer)
        chunk_hashes = [_sha256(c.get("text", "") or "") for c in chunks]
        # #region agent log
        _dbg({"location": "evidence_guardrail_service.check.entry", "message": "guardrail check entry", "data": {"len_chunks": len(chunks), "query_pre": (query or "")[:80], "chunks_have_text": sum(1 for c in chunks if (c.get("text") or "").strip())}, "hypothesisId": "C"})
        # #endregion

        # Edge cases
        if not answer or not answer.strip():
            return GuardrailResult(
                decision="fail", evidence_score="none",
                answer_hash=answer_hash, chunk_hashes=chunk_hashes,
                reason="Empty answer",
            )
        if not chunks:
            return GuardrailResult(
                decision="fail", evidence_score="none",
                answer_hash=answer_hash, chunk_hashes=chunk_hashes,
                reason="No chunks available for validation",
            )

        # Injection detection
        injection = self._detect_injection(chunks)

        # Sentence-level semantic coverage
        sentences = self._split_sentences(answer)
        sentence_results = self._validate_sentences(sentences, chunks)

        grounded_count = sum(1 for s in sentence_results if s.grounded)
        coverage_ratio = grounded_count / len(sentence_results) if sentence_results else 0.0

        # Structured citation cross-check (if model returned structured citations)
        citation_valid = True
        if structured_citations is not None:
            citation_valid = self._validate_structured_citations(structured_citations, chunks)
            if not citation_valid:
                logger.warning(
                    "Guardrail: structured citations %s not found in chunk page_numbers",
                    structured_citations,
                )

        # Determine evidence score
        if coverage_ratio >= settings.GUARDRAIL_STRONG_THRESHOLD and citation_valid:
            evidence_score = "strong"
        elif coverage_ratio >= settings.GUARDRAIL_STRONG_THRESHOLD:
            evidence_score = "moderate"
        elif coverage_ratio >= settings.GUARDRAIL_WEAK_THRESHOLD:
            evidence_score = "moderate"
        elif coverage_ratio > 0.0:
            evidence_score = "weak"
        else:
            evidence_score = "none"

        # Map to decision
        if evidence_score == "strong":
            decision = "pass"
        elif evidence_score in ("moderate", "weak"):
            decision = "weak"
        else:
            decision = "fail"

        # Injection always escalates to fail
        if injection:
            decision = "fail"
            evidence_score = "none"

        # #region agent log
        _dbg({"location": "evidence_guardrail_service.check.after_coverage", "message": "after coverage and decision", "data": {"coverage_ratio": coverage_ratio, "grounded_count": grounded_count, "num_sentences": len(sentence_results), "evidence_score": evidence_score, "decision": decision, "injection": injection, "citation_valid": citation_valid}, "hypothesisId": "A"})
        # #endregion

        # Holistic fallback: when sentence coverage is zero but chunks exist, check full-answer vs chunks
        if coverage_ratio == 0.0 and evidence_score == "none" and decision == "fail" and not injection and chunks:
            # #region agent log
            _dbg({"location": "evidence_guardrail_service.check.holistic_enter", "message": "holistic fallback entered", "data": {}, "hypothesisId": "A"})
            # #endregion
            holistic_sim = self._holistic_fallback(answer, chunks)
            thresh = settings.GUARDRAIL_HOLISTIC_THRESHOLD
            upgraded = holistic_sim >= thresh
            # #region agent log
            _dbg({"location": "evidence_guardrail_service.check.holistic_result", "message": "holistic fallback result", "data": {"holistic_sim": round(holistic_sim, 4), "threshold": thresh, "upgraded": upgraded}, "hypothesisId": "B"})
            # #endregion
            logger.info(
                "Guardrail holistic fallback: coverage_ratio=0, holistic_sim=%.3f, threshold=%.2f",
                holistic_sim, thresh,
            )
            if upgraded:
                evidence_score = "weak"
                decision = "weak"
                logger.info("Guardrail: upgraded to weak via holistic fallback (overview/synthesized answer).")
        else:
            # #region agent log
            _dbg({"location": "evidence_guardrail_service.check.holistic_skipped", "message": "holistic fallback not run", "data": {"coverage_ratio": coverage_ratio, "evidence_score": evidence_score, "decision": decision, "injection": injection, "has_chunks": bool(chunks), "num_sentences": len(sentence_results)}, "hypothesisId": "E"})
            # #endregion

        reason = (
            f"coverage={coverage_ratio:.2f} ({grounded_count}/{len(sentence_results)} sentences), "
            f"citation_valid={citation_valid}, injection={injection}"
        )
        logger.debug("Guardrail [%s]: %s", query[:60], reason)

        return GuardrailResult(
            decision=decision,
            evidence_score=evidence_score,
            coverage_ratio=coverage_ratio,
            sentence_results=sentence_results,
            injection_detected=injection,
            reason=reason,
            chunk_hashes=chunk_hashes,
            answer_hash=answer_hash,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Split answer into sentences, filtering out very short fragments."""
        raw = _SENTENCE_SPLIT.split(text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 15]

    def _validate_sentences(
        self,
        sentences: List[str],
        chunks: List[Dict[str, Any]],
    ) -> List[SentenceValidation]:
        """Embed each sentence and find its best-matching chunk."""
        results: List[SentenceValidation] = []
        # Pre-embed all chunks once
        chunk_embeddings: List[Optional[np.ndarray]] = []
        for c in chunks:
            text = c.get("text", "") or ""
            try:
                chunk_embeddings.append(self.embedding_service.embed_text(text) if text.strip() else None)
            except Exception as exc:
                logger.warning("Guardrail: failed to embed chunk: %s", exc)
                chunk_embeddings.append(None)

        for sentence in sentences:
            try:
                sent_emb = self.embedding_service.embed_text(sentence)
            except Exception as exc:
                logger.warning("Guardrail: failed to embed sentence: %s", exc)
                results.append(SentenceValidation(
                    sentence=sentence, max_similarity=0.0,
                    supporting_chunk_id=None, grounded=False,
                ))
                continue

            best_sim = 0.0
            best_chunk_id: Optional[str] = None
            for i, chunk_emb in enumerate(chunk_embeddings):
                if chunk_emb is None:
                    continue
                sim = _cosine_similarity(sent_emb, chunk_emb)
                if sim > best_sim:
                    best_sim = sim
                    c = chunks[i]
                    best_chunk_id = f"{c.get('chunk_index', i)}|{c.get('document_id', '')}"

            grounded = best_sim >= settings.GUARDRAIL_WEAK_THRESHOLD
            results.append(SentenceValidation(
                sentence=sentence,
                max_similarity=best_sim,
                supporting_chunk_id=best_chunk_id,
                grounded=grounded,
            ))
        return results

    def _holistic_fallback(self, answer: str, chunks: List[Dict[str, Any]]) -> float:
        """
        When sentence-level coverage is zero, check if the full answer is semantically
        similar to the chunks as a whole (mean of chunk embeddings). Used for overview/
        synthesized answers that are grounded in chunks but no single sentence hits threshold.
        """
        if not chunks:
            return 0.0
        vectors: List[np.ndarray] = []
        for c in chunks:
            text = c.get("text", "") or ""
            if not text.strip():
                continue
            try:
                emb = self.embedding_service.embed_text(text)
                vectors.append(np.asarray(emb))
            except Exception as exc:
                logger.warning("Guardrail holistic fallback: failed to embed chunk: %s", exc)
                return 0.0
        if not vectors:
            return 0.0
        try:
            answer_emb = np.asarray(self.embedding_service.embed_text(answer))
        except Exception as exc:
            logger.warning("Guardrail holistic fallback: failed to embed answer: %s", exc)
            return 0.0
        stack = np.stack(vectors)
        mean_chunk = np.mean(stack, axis=0)
        norm_mean = np.linalg.norm(mean_chunk)
        norm_ans = np.linalg.norm(answer_emb)
        if norm_mean == 0 or norm_ans == 0:
            return 0.0
        mean_chunk = mean_chunk / norm_mean
        answer_emb = answer_emb / norm_ans
        return float(_cosine_similarity(answer_emb, mean_chunk))

    def _validate_structured_citations(
        self,
        cited_pages: List[int],
        chunks: List[Dict[str, Any]],
    ) -> bool:
        """
        Verify every cited page_number exists in the retrieved chunks.
        Returns False if any cited page is NOT present in chunks.
        """
        available_pages = {c.get("page_number") for c in chunks if c.get("page_number") is not None}
        for page in cited_pages:
            if page not in available_pages:
                return False
        return True

    def _detect_injection(self, chunks: List[Dict[str, Any]]) -> bool:
        """Return True if any chunk text contains prompt-injection patterns."""
        for chunk in chunks:
            text = chunk.get("text", "") or ""
            if _INJECTION_PATTERNS.search(text):
                logger.warning(
                    "Guardrail: prompt injection pattern detected in chunk (doc=%s, page=%s)",
                    chunk.get("document_id"), chunk.get("page_number"),
                )
                return True
        return False
