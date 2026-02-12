"""
Legal context model for jurisdiction extraction.

Used by the legal context extractor to represent detected jurisdiction
and confidence from statutory/country references in document chunks.
"""
from typing import List, Optional

from pydantic import BaseModel


class LegalContextEvidence(BaseModel):
    """Single evidence snippet with page reference."""

    page_number: int
    text: str


class LegalContext(BaseModel):
    """
    Detected legal context (jurisdiction) from document text.

    confidence: "high" | "medium" | "low" | None
    - high: repeated statutory + country signals
    - medium: single strong country reference
    - low: weak statutory signal only (no explicit country name)
    - None: no jurisdictional signal
    """

    jurisdiction: Optional[str] = None
    confidence: Optional[str] = None  # high | medium | low
    evidence: List[LegalContextEvidence] = []
