"""
Document classification service.

Two-stage pipeline:
  Stage 1 — Ollama LLM (legal vs non-legal), heuristic fallback when Ollama is offline.
  Stage 2 — DistilBERT `briefme-io/legal_document_classifier` (Contract / Amendment / Other).

Returns ClassificationResult. Never raises — on any crash, returns UNCERTAIN and allows the
document through (per the Error Handling table in CLAUDE.md).
"""
import json
import logging
import re
import time
import threading
from typing import Dict, List, Optional

import httpx

from backend.config import settings
from backend.models.document import ClassificationResult, DocumentClassification

logger = logging.getLogger(__name__)

# ── Stage 1 constants ──────────────────────────────────────────────────────────
_LEGAL_KEYWORDS = [
    "whereas", "agreement", "contract", "jurisdiction", "clause",
    "party", "governing law", "indemnification", "arbitration", "termination notice",
    # Employment-specific English terms common in Saudi/GCC bilingual contracts
    "employee", "employer", "salary", "probation", "remuneration",
    "employment agreement", "notice period",
]
_KEYWORD_THRESHOLD = 3

# Arabic legal keywords for Saudi/GCC employment contracts and agreements.
# A lower threshold is used because each Arabic term is highly domain-specific.
_ARABIC_LEGAL_KEYWORDS = [
    "عقد عمل",        # employment contract
    "عقد",            # contract
    "اتفاقية",        # agreement
    "الموظف",         # the employee
    "صاحب العمل",     # the employer
    "الراتب",         # salary
    "الأجر",          # wage / remuneration
    "مدة العقد",      # contract duration
    "نظام العمل",     # Labor Law
    "إنهاء العقد",    # contract termination
    "فترة التجربة",   # probation period
    "التعويض",        # compensation
    "بدل",            # allowance
    "المملكة العربية السعودية",  # Saudi Arabia
]
_ARABIC_KEYWORD_THRESHOLD = 2

_LEGAL_CONFIDENCE_THRESHOLD = 0.6

_OLLAMA_CLASSIFY_PROMPT = """You are a document classifier. Respond ONLY with JSON (no markdown, no extra text):
{{
  "is_legal": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "one sentence",
  "contract_type": "employment" | "nda" | "msa" | "other" | null,
  "jurisdiction": "KSA" | "UAE" | "UK" | "US" | "Generic GCC" | "International" | null
}}

Legal = contracts, NDAs, employment agreements, statutes, court orders, regulations, court judgments, tribunal decisions.
Non-legal = invoices, news, emails, receipts, academic papers, marketing materials.

MULTILINGUAL: The document may be written in Arabic, English, or both. Classify based on legal
content, not language. Arabic employment contracts (عقد عمل) and Arabic agreements (اتفاقية) are
legal documents. Saudi/GCC documents referencing نظام العمل (Labor Law) or containing الموظف /
صاحب العمل (employee/employer) are legal.

contract_type rules:
- Only populate if is_legal=true AND the document is a contractual agreement between parties.
- Set to null if the document is a court judgment, ruling, tribunal decision, statute, or regulation — these are legal but NOT contracts.

jurisdiction: Infer from parties, governing law clauses, or country-specific legal references.
             Use "UK" for England & Wales / UK courts, "US" for United States courts or law.
             Return null if jurisdiction cannot be determined.

IMPORTANT: Court judgments, rulings, and tribunal decisions are legal documents (is_legal=true) but are NOT contracts.
Signals of a court document: claimant/defendant/appellant/respondent, judgment/ruling/verdict, judge titles (Mr Justice, Lord Justice), neutral citations (EWHC, EWCA), case numbers, hearing dates, "before the honourable".

Document excerpt:
---
{text}
---"""

# ── Stage 2 constants ──────────────────────────────────────────────────────────
_DISTILBERT_MODEL = "briefme-io/legal_document_classifier"
_CONTRACT_LABELS = {"contract", "amendment"}  # → is_contract=True

# Strong indicators that a document is a court/tribunal case file, NOT a contract.
# >= _CASE_FILE_THRESHOLD unique signals in the text → Stage 2 is skipped and
# the document is classified as LEGAL_NON_CONTRACT.
_CASE_FILE_SIGNALS: list[str] = [
    # Procedural party labels
    "claimant", "defendant", "plaintiff", "appellant", "respondent",
    "petitioner", "prosecution",
    # Judgment/order vocabulary
    "judgment", "judgement", "ruling", "verdict", "sentencing",
    "order of the court", "court order", "injunction", "decree",
    # UK judicial officer titles
    "mr justice", "mrs justice", "ms justice", "lord justice", "lady justice",
    "his honour", "her honour", "his lordship", "her ladyship",
    "master of the rolls", "lord chief justice",
    # US judicial officer titles
    "circuit judge", "district judge", "chief justice",
    "associate justice", "magistrate judge",
    # GCC/Arabic court terms
    "القاضي", "المحكمة", "حكم المحكمة", "دائرة قضائية",
    # UK court names
    "high court", "court of appeal", "supreme court", "crown court",
    "county court", "magistrates court", "chancery division",
    "queen's bench", "king's bench", "family division",
    # US court names
    "district court", "court of appeals", "circuit court",
    "superior court", "court of claims", "bankruptcy court",
    # International/GCC tribunals
    "tribunal", "international court", "icj", "icsid", "echr",
    # UK neutral citation markers
    "neutral citation", "ewhc", "ewca", "ewcop",
    # Case number prefixes
    "case no.", "case no ", "case number", "claim no.", "claim number",
    "qb-", "ch-", "comm-", "tlq-",
    # Hearing language
    "hearing date", "handed down", "approved judgment",
    "before the honourable", "before mr justice",
    # Litigation procedural terms
    "pleadings", "particulars of claim", "statement of case",
    "witness statement", "skeleton argument", "written submissions",
    "costs order", "summary judgment", "strike out",
    # Criminal-specific
    "the accused", "the prosecution", "guilty", "not guilty",
    "acquitted", "convicted",
    # Statute/citation patterns
    "state immunity act", "human rights act",
]
_CASE_FILE_THRESHOLD: int = 3

# Keyword sets used by the heuristic fallback to infer contract type when Ollama is unavailable.
_HEURISTIC_CONTRACT_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "employment": ["employee", "employer", "salary", "probation", "employment agreement", "notice period"],
    "nda":        ["non disclosure", "confidential information", "disclosing party", "receiving party"],
    "msa":        ["service provider", "master service agreement", "sla", "deliverables", "service level"],
}

# Lazy-loaded pipeline; protected by a lock so we only download once.
_distilbert_pipeline = None
_distilbert_lock = threading.Lock()
_distilbert_failed = False  # Set True permanently if download fails.


def _get_distilbert_pipeline():
    """Return (or lazily create) the HuggingFace text-classification pipeline."""
    global _distilbert_pipeline, _distilbert_failed
    if _distilbert_failed:
        return None
    if _distilbert_pipeline is not None:
        return _distilbert_pipeline
    with _distilbert_lock:
        if _distilbert_pipeline is not None:
            return _distilbert_pipeline
        if _distilbert_failed:
            return None
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading %s from HuggingFace (first use)…", _DISTILBERT_MODEL)
            _distilbert_pipeline = hf_pipeline(
                "text-classification",
                model=_DISTILBERT_MODEL,
                truncation=True,
                max_length=512,
            )
            logger.info("DistilBERT classifier loaded.")
            return _distilbert_pipeline
        except Exception as exc:
            logger.warning("Failed to load DistilBERT classifier: %s", exc)
            _distilbert_failed = True
            return None


def _count_case_signals(text: str) -> int:
    """Count unique case-file signal phrases in lowercased text.
    Returns >= _CASE_FILE_THRESHOLD for court/tribunal documents."""
    lower = text.lower()
    return sum(1 for sig in _CASE_FILE_SIGNALS if sig in lower)


class DocumentClassificationService:
    """Orchestrates the two-stage document classification pipeline."""

    def __init__(self, ollama_base_url: Optional[str] = None, ollama_model: Optional[str] = None):
        self._ollama_base_url = ollama_base_url or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        self._ollama_model = ollama_model or getattr(settings, "OLLAMA_MODEL", "qwen2.5:3b")

    # ── Public entry point ──────────────────────────────────────────────────────

    def classify(self, text_sample: str, document_id: str = "") -> ClassificationResult:
        """
        Classify a document given a text sample (first ~2000 chars recommended).

        Never raises. On any unhandled exception returns UNCERTAIN.
        """
        try:
            return self._classify_internal(text_sample, document_id)
        except Exception as exc:
            logger.exception("Classification crash for %s: %s", document_id, exc)
            return ClassificationResult(
                is_legal=False,
                is_contract=False,
                classification=DocumentClassification.UNCERTAIN,
                confidence=0.0,
                method="llm_only",
                error=f"Classification crash: {exc}",
            )

    # ── Internal pipeline ───────────────────────────────────────────────────────

    def _classify_internal(self, text_sample: str, document_id: str) -> ClassificationResult:
        if not text_sample or not text_sample.strip():
            # No text → keyword score will be 0 → uncertain
            return ClassificationResult(
                is_legal=False,
                is_contract=False,
                classification=DocumentClassification.UNCERTAIN,
                confidence=0.0,
                method="llm_only",
                error="Could not read document text",
            )

        # ── Stage 1: LLM or heuristic ──────────────────────────────────────────
        stage1_is_legal, stage1_confidence, stage1_reasoning, used_heuristic, \
            detected_contract_type, detected_jurisdiction = self._stage1(text_sample)
        method_prefix = "heuristic" if used_heuristic else "llm"

        if not stage1_is_legal:
            return ClassificationResult(
                is_legal=False,
                is_contract=False,
                classification=DocumentClassification.NON_LEGAL,
                confidence=stage1_confidence,
                reasoning=stage1_reasoning,
                method=f"{method_prefix}+distilbert" if not used_heuristic else "heuristic+distilbert",
            )

        # ── Case-file gate: skip Stage 2 for court/tribunal documents ──────────
        case_signal_count = _count_case_signals(text_sample)
        if case_signal_count >= _CASE_FILE_THRESHOLD:
            logger.info(
                "Case-file gate triggered (signals=%d) for document_id=%r — "
                "classifying as LEGAL_NON_CONTRACT without running Stage 2.",
                case_signal_count, document_id,
            )
            return ClassificationResult(
                is_legal=True,
                is_contract=False,
                classification=DocumentClassification.LEGAL_NON_CONTRACT,
                confidence=stage1_confidence,
                contract_confidence=None,
                reasoning=(
                    stage1_reasoning
                    or f"Case file detected ({case_signal_count} court-document signals found)"
                ),
                method=f"{method_prefix}+case_gate",
                detected_contract_type=None,
                detected_jurisdiction=detected_jurisdiction,
            )

        # ── Stage 2: DistilBERT contract classifier ────────────────────────────
        is_contract, contract_confidence, stage2_error = self._stage2(text_sample)

        if stage2_error:
            # Stage 2 failed → llm_only, classify as legal_non_contract
            return ClassificationResult(
                is_legal=True,
                is_contract=False,
                classification=DocumentClassification.LEGAL_NON_CONTRACT,
                confidence=stage1_confidence,
                reasoning=stage1_reasoning,
                method="llm_only",
                error=stage2_error,
                detected_contract_type=detected_contract_type,
                detected_jurisdiction=detected_jurisdiction,
            )

        # ── Post-Stage-2 case-file override (safety net) ───────────────────────
        # case_signal_count was computed in the case-file gate above — always in scope here.
        if is_contract and case_signal_count >= _CASE_FILE_THRESHOLD:
            logger.warning(
                "Post-Stage-2 override: DistilBERT returned 'contract' but "
                "case_signal_count=%d >= threshold=%d for document_id=%r. "
                "Overriding to LEGAL_NON_CONTRACT.",
                case_signal_count, _CASE_FILE_THRESHOLD, document_id,
            )
            is_contract = False
            detected_contract_type = None

        classification = (
            DocumentClassification.LEGAL_CONTRACT
            if is_contract
            else DocumentClassification.LEGAL_NON_CONTRACT
        )
        return ClassificationResult(
            is_legal=True,
            is_contract=is_contract,
            classification=classification,
            confidence=stage1_confidence,
            contract_confidence=contract_confidence,
            reasoning=stage1_reasoning,
            method=f"{method_prefix}+distilbert",
            detected_contract_type=detected_contract_type,
            detected_jurisdiction=detected_jurisdiction,
        )

    # ── Stage 1 helpers ─────────────────────────────────────────────────────────

    def _stage1(self, text: str):
        """
        Returns (is_legal, confidence, reasoning, used_heuristic, contract_type, jurisdiction).
        Tries Ollama first; falls back to keyword heuristic on any failure.
        """
        try:
            result = self._ollama_classify(text)
            if result is not None:
                is_legal, confidence, reasoning, contract_type, jurisdiction = result
                return is_legal, confidence, reasoning, False, contract_type, jurisdiction
        except Exception as exc:
            logger.warning("Ollama classification failed (%s), using heuristic fallback", exc)

        # Heuristic fallback — contract_type inferred from keywords, no jurisdiction detection
        is_legal, confidence, contract_type = self._keyword_classify(text)
        return is_legal, confidence, None, True, contract_type, None

    def _ollama_classify(self, text: str):
        """
        Call Ollama generate endpoint.
        Returns (is_legal, confidence, reasoning, contract_type, jurisdiction) or None on failure.
        """
        prompt = _OLLAMA_CLASSIFY_PROMPT.format(text=text[:2000])
        try:
            resp = httpx.post(
                f"{self._ollama_base_url}/api/generate",
                json={"model": self._ollama_model, "prompt": prompt, "stream": False},
                timeout=30.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
        except Exception as exc:
            logger.warning("Ollama request failed: %s", exc)
            return None

        return self._parse_stage1_response(raw)

    def _parse_stage1_response(self, raw: str):
        """
        Extract JSON from LLM response.
        Returns (is_legal, confidence, reasoning, contract_type, jurisdiction) or None.
        """
        _VALID_CONTRACT_TYPES = {"employment", "nda", "msa", "other"}
        _VALID_JURISDICTIONS = {"KSA", "UAE", "UK", "US", "Generic GCC", "International"}

        def _extract_fields(data: dict):
            ct = data.get("contract_type")
            jur = data.get("jurisdiction")
            if ct not in _VALID_CONTRACT_TYPES:
                ct = None
            if jur not in _VALID_JURISDICTIONS:
                jur = None
            return (
                bool(data["is_legal"]),
                float(data["confidence"]),
                data.get("reasoning"),
                ct,
                jur,
            )

        # Try direct parse
        try:
            data = json.loads(raw.strip())
            return _extract_fields(data)
        except Exception:
            pass

        # Regex fallback: extract first {...} block
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return _extract_fields(data)
            except Exception:
                pass

        logger.warning("Could not parse stage1 LLM JSON from: %r", raw[:200])
        return None  # Caller will fall back to heuristic

    def _keyword_classify(self, text: str):
        """Bilingual keyword scan (English + Arabic). Returns (is_legal, confidence, contract_type)."""
        lower = text.lower()
        en_hits = sum(1 for kw in _LEGAL_KEYWORDS if kw in lower)
        # Arabic keywords are checked against the original text (case is irrelevant for Arabic)
        ar_hits = sum(1 for kw in _ARABIC_LEGAL_KEYWORDS if kw in text)

        en_legal = en_hits >= _KEYWORD_THRESHOLD
        ar_legal = ar_hits >= _ARABIC_KEYWORD_THRESHOLD
        is_legal = en_legal or ar_legal

        en_conf = min(0.9, en_hits / len(_LEGAL_KEYWORDS))
        ar_conf = min(0.9, ar_hits / len(_ARABIC_LEGAL_KEYWORDS))
        confidence = max(en_conf, ar_conf)

        # Infer contract type from keyword matches
        contract_type: Optional[str] = None
        if is_legal:
            ct_scores = {
                ct: sum(1 for kw in kws if kw in lower)
                for ct, kws in _HEURISTIC_CONTRACT_TYPE_KEYWORDS.items()
            }
            best = max(ct_scores.items(), key=lambda x: x[1])
            if best[1] > 0:
                contract_type = best[0]

        return is_legal, confidence, contract_type

    # ── Stage 2 helper ──────────────────────────────────────────────────────────

    def _stage2(self, text: str):
        """
        Run DistilBERT classifier.
        Returns (is_contract, contract_confidence, error_str_or_None).
        """
        pipe = _get_distilbert_pipeline()
        if pipe is None:
            return False, None, "DistilBERT model unavailable (download failed or not installed)"

        try:
            # 2000 chars ≈ 400 tokens, safely within DistilBERT 512 token limit
            results = pipe(text[:2000], truncation=True, max_length=512)
            if not results:
                return False, None, "DistilBERT returned empty result"
            top = results[0] if isinstance(results[0], dict) else results[0][0]
            label: str = top["label"].lower()
            score: float = float(top["score"])
            is_contract = label in _CONTRACT_LABELS
            return is_contract, score, None
        except Exception as exc:
            logger.warning("DistilBERT inference failed: %s", exc)
            return False, None, str(exc)
