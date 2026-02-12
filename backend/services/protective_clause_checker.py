"""
Protective / supremacy clause detection from document chunks.

Rule-based only; no LLM. Flags clauses that contain supremacy or anti-fraud language.
"""
from typing import List, Dict, Any

PROTECTIVE_KEYWORDS = [
    "supersede",
    "only valid",
    "null and void",
    "substitution",
    "cancel",
    "no other agreement",
]


def detect_protective_clauses(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return list of findings: each has type, category, page_number, evidence (substring of chunk).
    No hallucinated text; evidence is chunk["text"][:300].
    """
    findings: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").lower()
        page_number = int(chunk.get("page_number", 0))
        if not text:
            continue
        for keyword in PROTECTIVE_KEYWORDS:
            if keyword in text:
                findings.append({
                    "type": "supremacy_clause",
                    "category": "protective",
                    "page_number": page_number,
                    "evidence": (chunk.get("text") or "")[:300],
                })
                break
    return findings
