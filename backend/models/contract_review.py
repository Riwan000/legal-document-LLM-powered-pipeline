"""
Pydantic models for Contract Review workflow outputs.

These models describe the deliverable-oriented schema for the Contract Review
workflow as specified in the PRD:
- Risk table with severities and citations
- Clause-level evidence blocks
- Executive summary items
"""

from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, Field

StructureClass = Literal["clause", "provision", "section_non_standard"]


RiskSeverity = Literal["high", "medium", "low"]
RiskStatus = Literal["detected", "not_detected", "uncertain", "implicitly_covered"]


class VerbatimSnippet(BaseModel):
    """Verbatim text excerpt from a matching evidence block, attached to a RiskItem."""

    clause_id: str = Field(..., description="Clause identifier the snippet comes from.")
    display_name: Optional[str] = Field(default=None, description="Human-readable clause name.")
    page_number: int = Field(..., description="Page where the snippet appears.")
    text: str = Field(..., description="clean_text from ClauseEvidenceBlock, truncated to 400 chars.")


class RiskItem(BaseModel):
    """
    Single risk entry in the contract review risk table.

    This is designed to be easily renderable in a tabular UI and exportable.
    """

    description: str = Field(..., description="Plain-language description of the identified risk or issue.")
    severity: RiskSeverity = Field(..., description='Risk severity: "high", "medium", or "low".')
    status: Optional[RiskStatus] = Field(
        default=None,
        description="Clause presence status for expected-clause checks: detected | not_detected | uncertain.",
    )
    clause_types: List[str] = Field(
        default_factory=list,
        description="Canonical clause types associated with this risk (e.g. ['termination', 'liability']).",
    )
    missing_clause: bool = Field(
        default=False,
        description="True if this risk represents a missing expected clause rather than problematic language.",
    )
    clause_ids: List[str] = Field(
        default_factory=list,
        description="IDs of clauses contributing to this risk (if present).",
    )
    page_numbers: List[int] = Field(
        default_factory=list,
        description="List of page numbers where supporting evidence appears.",
    )
    display_names: List[str] = Field(
        default_factory=list,
        description="Human-readable clause names for UI (one per clause_ids entry when available).",
    )
    coverage_note: Optional[str] = Field(
        default=None,
        description="When status is implicitly_covered, optional note describing where related obligations appear.",
    )
    # ── Improvement 1: verbatim evidence per risk ──────────────────────────────
    verbatim_evidence: List[VerbatimSnippet] = Field(
        default_factory=list,
        description="Up to 3 verbatim text snippets from matching evidence blocks (only for detected clauses).",
    )
    # ── Improvement 2: severity reason + recommendation ───────────────────────
    severity_reason: Optional[str] = Field(
        default=None,
        description="1–2 sentences explaining why this severity was assigned (from static templates).",
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Actionable guidance for addressing this risk (from static templates).",
    )


class ClauseEvidenceBlock(BaseModel):
    """
    Evidence block for a clause used in Contract Review.

    Mirrors the PRD's requirement to surface:
    - Original text
    - Cleaned text
    - Page reference
    - Optional human-readable display name (no unknown_* or hashes in UI).
    """

    clause_id: str = Field(..., description="Unique clause identifier.")
    page_number: int = Field(..., description="Page number where this evidence appears.")
    raw_text: str = Field(..., description="Original verbatim text (e.g. OCR output).")
    clean_text: str = Field(..., description="Normalized/cleaned text used for analysis.")
    display_name: Optional[str] = Field(
        default=None,
        description="Human-readable clause name for UI; derived from clause_id at render time.",
    )
    is_non_contractual: bool = Field(
        default=False,
        description="True if evidence looks like a section header or non-contractual text; UI may show 'Section (Non-contractual)'.",
    )
    semantic_label: Optional[str] = Field(
        default=None,
        description="G3: When set, UI uses 'Section: {semantic_label} — Page N'; else 'Section (Non-standard)'.",
    )
    structure_class: Optional[StructureClass] = Field(
        default=None,
        description="Display-level structure confidence: clause (heading + obligations + low OCR noise), provision (obligations only), section_non_standard.",
    )


ExecutiveSummaryCategory = Literal["risk", "finding", "confirmation"]


class ExecutiveSummaryItem(BaseModel):
    """Single bullet in the Contract Review executive summary."""

    text: str = Field(..., description="Summary sentence/paragraph.")
    severity: Optional[RiskSeverity] = Field(
        default=None,
        description="Optional associated severity, if this item is tied to a key risk.",
    )
    category: Optional[ExecutiveSummaryCategory] = Field(
        default=None,
        description="Bucket for UI: risk (missing clause), finding (non-standard/limited), confirmation (present).",
    )
    related_risk_indices: List[int] = Field(
        default_factory=list,
        description="Indices into the risks array that this summary item relates to.",
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Actionable guidance for this summary item (from static templates, mirrors RiskItem.recommendation).",
    )


class ContractReviewResponse(BaseModel):
    """
    Top-level Contract Review workflow output.

    This is what the orchestrator will ultimately expose (possibly wrapped
    in a WorkflowContext envelope at the API layer).
    """

    workflow_id: str = Field(..., description="Workflow run identifier (from WorkflowContext).")
    document_id: str = Field(..., description="Primary contract document ID under review.")
    contract_type: str = Field(..., description="Contract type used for profile lookup (e.g. 'NDA', 'employment').")
    jurisdiction: Optional[str] = Field(
        default=None,
        description="Jurisdiction applied for checks (e.g. 'KSA', 'UAE', 'Generic GCC').",
    )

    risks: List[RiskItem] = Field(
        default_factory=list,
        description="Risk table entries (each with severity and citations).",
    )
    evidence: List[ClauseEvidenceBlock] = Field(
        default_factory=list,
        description="Flattened list of clause-level evidence blocks.",
    )
    executive_summary: List[ExecutiveSummaryItem] = Field(
        default_factory=list,
        description="Executive summary items for senior review.",
    )
    not_detected_clauses: List[str] = Field(
        default_factory=list,
        description="Human-readable names of expected clauses for which no evidence was found (display list only).",
    )
    implicitly_covered_clauses: List[str] = Field(
        default_factory=list,
        description="Human-readable names of expected clauses that appear implicitly across other provisions.",
    )
    implicit_coverage_notes: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional map from clause display name to coverage note for implicitly_covered clauses.",
    )
    document_classification_warning: Optional[str] = Field(
        default=None,
        description="If set, document did not appear to be an operative contract; UI should show a non-blocking banner.",
    )
    used_implicit_or_distributed_logic: bool = Field(
        default=False,
        description="True if governing law was detected implicit, benefits distributed, or any evidence is section_non_standard; UI shows global disclaimer only when True.",
    )
    disclaimer: str = Field(
        default=(
            "This system does not provide legal advice. "
            "All outputs are for review purposes only."
        ),
        description="Mandatory disclaimer text for all Contract Review outputs.",
    )


class ContractReviewRequest(BaseModel):
    """
    Request payload for the Contract Review workflow.

    This is the logical request model; the FastAPI endpoint may still accept
    these fields via form-data for Streamlit compatibility.
    """

    contract_id: str = Field(..., description="Primary contract document ID.")
    contract_type: str = Field(
        ...,
        description="Contract type key used to select a contract profile (e.g. 'nda', 'employment', 'msa').",
    )
    jurisdiction: Optional[str] = Field(
        default=None,
        description="Jurisdiction to apply when interpreting the contract profile (e.g. 'KSA', 'UAE', 'Generic GCC').",
    )
    review_depth: Optional[Literal["quick", "standard"]] = Field(
        default="standard",
        description="Review depth hint; can be used to adjust thresholds or LLM usage.",
    )



