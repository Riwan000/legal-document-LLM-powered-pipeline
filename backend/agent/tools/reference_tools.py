from __future__ import annotations

"""
Supporting reference tools (Phase 2): extracted clause listing and
deterministic statutory lookups from jurisdiction_statutes.yaml.

lookup_statutes is the only tool without document scope — it reads a
static reference file, never document content.
"""

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from backend.models.agent import ToolParam, ToolResult, ToolSpec

if TYPE_CHECKING:
    from backend.agent.tool_registry import ToolRegistry
    from backend.agent.tools import AgentToolServices

_STATUTES_PATH = Path(__file__).parent.parent.parent / "legal_references" / "jurisdiction_statutes.yaml"

# Token → canonical YAML key. Mirrors _canon_jurisdiction in
# contract_review_service.py; keep in sync when adding jurisdictions.
_JURISDICTION_TOKENS = (
    (("ksa", "saudi"), "KSA"),
    (("uae", "emirates", "dubai", "abu dhabi"), "UAE"),
    (("qatar",), "Qatar"),
    (("bahrain",), "Bahrain"),
    (("oman",), "Oman"),
    (("kuwait",), "Kuwait"),
)

_CLAUSE_LISTING_FIELDS = (
    "clause_id",
    "clause_number",
    "clause_heading",
    "clause_title",
    "legal_category",
    "unit_type",
    "page_start",
    "page_end",
)


@lru_cache(maxsize=1)
def _load_statutes() -> Dict[str, Any]:
    if not _STATUTES_PATH.exists():
        return {}
    with open(_STATUTES_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _canon_jurisdiction_key(raw: str) -> Optional[str]:
    lowered = raw.strip().lower()
    for tokens, canonical in _JURISDICTION_TOKENS:
        if any(token in lowered for token in tokens):
            return canonical
    # Fall back to a direct case-insensitive key match against the YAML.
    for key in _load_statutes():
        if key.lower() == lowered:
            return key
    return None


def register_reference_tools(registry: "ToolRegistry", services: "AgentToolServices") -> None:
    def extract_clauses(
        document_ids: List[str],
        jurisdiction: Optional[str] = None,
        clause_type: Optional[str] = None,
    ) -> ToolResult:
        if services.extracted_clause_store is None:
            return ToolResult(
                tool_name="extract_clauses",
                status="error",
                summary="Clause store is not available in this deployment.",
            )
        clauses_by_document: Dict[str, List[Dict[str, Any]]] = {}
        total = 0
        for doc_id in document_ids:
            raw_clauses = services.extracted_clause_store.get_document_clauses(doc_id) or []
            entries = [
                {field: clause.get(field) for field in _CLAUSE_LISTING_FIELDS}
                for clause in raw_clauses
            ]
            if clause_type:
                wanted = clause_type.strip().lower()
                entries = [
                    e
                    for e in entries
                    if wanted in str(e.get("legal_category") or "").lower()
                    or wanted in str(e.get("clause_heading") or "").lower()
                    or wanted in str(e.get("clause_title") or "").lower()
                ]
            clauses_by_document[doc_id] = entries
            total += len(entries)

        headings = [
            str(e.get("clause_heading") or e.get("clause_title") or "?")
            for entries in clauses_by_document.values()
            for e in entries
        ]
        preview = ", ".join(headings[:5])
        return ToolResult(
            tool_name="extract_clauses",
            status="ok",
            summary=(
                f"{total} clause(s)"
                + (f" matching '{clause_type}'" if clause_type else "")
                + (f": {preview}" if preview else "")
                + ("…" if len(headings) > 5 else "")
            ),
            payload={"clauses": clauses_by_document},
        )

    registry.register(
        ToolSpec(
            name="extract_clauses",
            description=(
                "List the extracted clauses (headings, numbers, categories, pages) of the "
                "documents in scope, optionally filtered by clause type. Use to check which "
                "clauses exist before answering clause-specific questions."
            ),
            params=[
                ToolParam(
                    name="clause_type",
                    type="string",
                    description="Optional filter, e.g. 'termination' or 'liability'.",
                    required=False,
                ),
            ],
        ),
        extract_clauses,
    )

    def lookup_statutes(
        jurisdiction: str,
        clause_type: Optional[str] = None,
    ) -> ToolResult:
        statutes = _load_statutes()
        canonical = _canon_jurisdiction_key(jurisdiction)
        entries = statutes.get(canonical) if canonical else None
        if not entries:
            available = ", ".join(sorted(statutes)) or "(none)"
            return ToolResult(
                tool_name="lookup_statutes",
                status="error",
                summary=f"No statutory references for '{jurisdiction}'. Available: {available}.",
            )
        if clause_type:
            wanted = clause_type.strip().lower()
            entries = {k: v for k, v in entries.items() if wanted in k.lower()}
        return ToolResult(
            tool_name="lookup_statutes",
            status="ok",
            summary=(
                f"{len(entries)} statutory reference(s) for {canonical}: "
                f"{', '.join(sorted(entries)) or '(none matching filter)'}."
            ),
            payload={"jurisdiction": canonical, "statutes": entries},
        )

    registry.register(
        ToolSpec(
            name="lookup_statutes",
            description=(
                "Look up jurisdiction-specific statutory article references (GCC labor law) "
                "from the static reference file. Deterministic; reads no document content. "
                "Use when the user asks what the law of a jurisdiction provides."
            ),
            params=[
                ToolParam(
                    name="jurisdiction",
                    type="string",
                    description="Jurisdiction name or code, e.g. 'KSA', 'UAE', 'Qatar'.",
                ),
                ToolParam(
                    name="clause_type",
                    type="string",
                    description="Optional topic filter, e.g. 'notice' or 'annual_leave'.",
                    required=False,
                ),
            ],
            requires_document_scope=False,
        ),
        lookup_statutes,
    )
