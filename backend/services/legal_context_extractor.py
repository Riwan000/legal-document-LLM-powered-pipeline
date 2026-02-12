"""
Legal context extractor: jurisdiction detection from document chunks.

Rule-based only; no LLM. Uses statutory and country-name patterns to infer
jurisdiction and confidence (high / medium / low).
"""
import re
from typing import List, Dict

from backend.models.legal_context import LegalContext, LegalContextEvidence

# Country -> list of regex patterns (statutory or country references)
COUNTRY_PATTERNS: Dict[str, List[str]] = {
    "Saudi Arabia": [
        r"Kingdom of Saudi Arabia",
        r"Saudi Arabia",
        r"Labour.*Workman.*Law",
        r"Article\s+83",
    ],
    "India": [
        r"Emigration Act",
        r"India",
    ],
}

# Explicit country-name patterns only (for medium vs low confidence when primary_hits == 1)
COUNTRY_NAME_PATTERNS: Dict[str, List[str]] = {
    "Saudi Arabia": [
        r"Kingdom of Saudi Arabia",
        r"Saudi Arabia",
    ],
    "India": [
        r"India",
    ],
}


def extract_legal_context(chunks: List[dict]) -> LegalContext:
    """
    Extract jurisdiction and confidence from chunks (each must have "text" and "page_number").

    Confidence:
    - high: primary_hits >= 2
    - medium: primary_hits == 1 and explicit country name present in evidence
    - low: primary_hits == 1 and no explicit country name (statutory-only)
    - None: no hits
    """
    evidence: List[LegalContextEvidence] = []
    country_hits: Dict[str, int] = {}
    # Collect evidence texts per country for explicit-name check
    evidence_texts_by_country: Dict[str, List[str]] = {}

    for chunk in chunks:
        text = chunk.get("text") or ""
        page = int(chunk.get("page_number", 0))
        if not text.strip():
            continue

        for country, patterns in COUNTRY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    country_hits[country] = country_hits.get(country, 0) + 1
                    evidence.append(
                        LegalContextEvidence(
                            page_number=page,
                            text=text.strip()[:300],
                        )
                    )
                    if country not in evidence_texts_by_country:
                        evidence_texts_by_country[country] = []
                    evidence_texts_by_country[country].append(text)
                    break

    if not country_hits:
        return LegalContext(jurisdiction=None, confidence=None, evidence=[])

    primary = max(country_hits, key=country_hits.get)
    primary_hits = country_hits[primary]

    if primary_hits >= 2:
        confidence = "high"
    elif primary_hits == 1:
        name_patterns = COUNTRY_NAME_PATTERNS.get(primary, [])
        collected_texts = evidence_texts_by_country.get(primary, [])
        explicit_country_name_present = any(
            re.search(pat, t, re.IGNORECASE)
            for t in collected_texts
            for pat in name_patterns
        )
        confidence = "medium" if explicit_country_name_present else "low"
    else:
        confidence = "medium"

    return LegalContext(
        jurisdiction=primary,
        confidence=confidence,
        evidence=evidence,
    )
