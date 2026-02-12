"""
Heuristic structure detection for documents (no LLM).
Uses regex/patterns to detect structured headings and estimate clause-like sections.
"""
import re
import logging
from typing import List, Dict, Any

from backend.models.document_structure import PageInfo, HeuristicStructureResult

logger = logging.getLogger(__name__)

# Patterns for heading-like lines (contracts, legal docs)
PATTERN_NUMBERED = re.compile(r"^\d+\.\s+", re.MULTILINE)  # "1. Term", "2. Definitions"
PATTERN_ARTICLE = re.compile(r"^Article\s+\d+", re.IGNORECASE | re.MULTILINE)
PATTERN_CLAUSE = re.compile(r"^Clause\s+\d+", re.IGNORECASE | re.MULTILINE)
PATTERN_UPPERCASE_HEADING = re.compile(r"^[A-Z][A-Z\s]{5,}$", re.MULTILINE)  # ALL CAPS line
# Section headings like "Section 1", "Part A"
PATTERN_SECTION = re.compile(r"^(?:Section|Part)\s+[\dA-Z]+", re.IGNORECASE | re.MULTILINE)

ALL_PATTERNS = [
    ("numbered", PATTERN_NUMBERED),
    ("article", PATTERN_ARTICLE),
    ("clause", PATTERN_CLAUSE),
    ("section", PATTERN_SECTION),
    ("uppercase", PATTERN_UPPERCASE_HEADING),
]


def _count_heading_matches(text: str) -> int:
    """Count lines that match any heading pattern."""
    count = 0
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        for _name, pat in ALL_PATTERNS:
            if pat.search(line_stripped) or (pat.match(line_stripped)):
                count += 1
                break
    return count


def _find_heading_positions(text: str) -> List[int]:
    """Return start indices of lines that look like headings (for splitting)."""
    positions = [0]
    offset = 0
    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped:
            for _name, pat in ALL_PATTERNS:
                if pat.search(line_stripped) or pat.match(line_stripped):
                    positions.append(offset)
                    break
        offset += len(line) + 1  # +1 for newline
    return sorted(set(positions))


class StructureHeuristicService:
    """Fast, deterministic structure detection using regex (no LLM)."""

    def detect(self, pages: List[PageInfo]) -> HeuristicStructureResult:
        """
        Run heuristic structure detection on parsed pages.
        On any error, returns safe fallback (no structure, 0 confidence).
        """
        try:
            full_text = "\n".join(p.text for p in pages)
            if not full_text.strip():
                return HeuristicStructureResult(
                    has_structured_headings=False,
                    estimated_clause_count=0,
                    confidence=0.0,
                )
            total_lines = len([l for l in full_text.splitlines() if l.strip()])
            heading_count = _count_heading_matches(full_text)
            # Estimated "clauses" = heading-like segments (split by headings)
            positions = _find_heading_positions(full_text)
            estimated_clause_count = max(1, len(positions) - 1) if len(positions) > 1 else 0
            if heading_count > 0 and total_lines > 0:
                density = heading_count / total_lines
                # Require at least some density and multiple headings
                has_structured = density >= 0.02 and heading_count >= 3
                # Confidence from density and consistency (multiple heading types)
                confidence = min(1.0, density * 10.0 + (0.2 if heading_count >= 5 else 0))
            else:
                has_structured = False
                confidence = 0.0
            return HeuristicStructureResult(
                has_structured_headings=has_structured,
                estimated_clause_count=estimated_clause_count,
                confidence=round(confidence, 2),
            )
        except Exception as e:
            logger.warning("Structure heuristic detection failed: %s", e, exc_info=True)
            return HeuristicStructureResult(
                has_structured_headings=False,
                estimated_clause_count=0,
                confidence=0.0,
            )

    def extract_heuristic_clauses_by_page(
        self, pages: List[PageInfo]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Split each page by heading patterns into segment "clauses" for clause-aware chunking.
        Returns dict: page_number -> list of clause dicts with keys text, start_index, type, clause_id.
        """
        result: Dict[int, List[Dict[str, Any]]] = {}
        for page in pages:
            text = page.text
            if not text.strip():
                result[page.page_number] = []
                continue
            positions = _find_heading_positions(text)
            if len(positions) <= 1:
                # No headings: single segment
                result[page.page_number] = [
                    {
                        "text": text.strip(),
                        "start_index": 0,
                        "type": "Section",
                        "clause_id": f"h_p{page.page_number}_0",
                    }
                ]
                continue
            clauses = []
            for idx in range(len(positions)):
                start = positions[idx]
                end = positions[idx + 1] if idx + 1 < len(positions) else len(text)
                segment = text[start:end].strip()
                if not segment:
                    continue
                clauses.append({
                    "text": segment,
                    "start_index": start,
                    "type": "Section",
                    "clause_id": f"h_p{page.page_number}_{idx}",
                })
            result[page.page_number] = clauses
        return result
