"""
Chunking strategy selection based on heuristic structure (no LLM).
Chooses between clause_aware and sentence strategies using conservative thresholds.
"""
from typing import Optional

from backend.models.document_structure import HeuristicStructureResult

MIN_CLAUSE_THRESHOLD = 5
MIN_STRUCTURE_CONFIDENCE = 0.6


class ChunkingStrategyService:
    """Select chunking strategy from heuristic structure result."""

    def select(
        self,
        heuristics: HeuristicStructureResult,
        document_type_hint: Optional[str] = None,
        *,
        force_strategy: Optional[str] = None,
    ) -> str:
        """
        Choose clause_aware or sentence strategy from heuristics.
        Does not force clause_aware based on document_type_hint alone.
        force_strategy: optional override for tests/admin ("clause_aware" | "sentence").
        """
        if force_strategy in ("clause_aware", "sentence"):
            return force_strategy
        if (
            heuristics.has_structured_headings
            and heuristics.estimated_clause_count >= MIN_CLAUSE_THRESHOLD
            and heuristics.confidence >= MIN_STRUCTURE_CONFIDENCE
        ):
            return "clause_aware"
        return "sentence"
