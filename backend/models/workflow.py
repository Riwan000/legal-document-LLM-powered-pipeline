from __future__ import annotations

"""
Workflow-level models and shared context for orchestrated legal workflows.

This module defines:
- WorkflowError: structured failure information
- WorkflowContext: shared, stateful context passed through all workflow steps
- DocumentExplorer request/response models
"""

from typing import Literal, Optional, Dict, List, Any

from pydantic import BaseModel, Field


class WorkflowError(BaseModel):
    """
    Shared error schema for workflows.

    Used to capture structured failures at any step in a workflow while
    preserving intermediate results in the surrounding WorkflowContext.
    """

    code: str = Field(
        ...,
        description='Short machine-readable error code (e.g. "NO_EVIDENCE", "OCR_FAILED")',
    )
    message: str = Field(
        ...,
        description="Human-readable error message describing what went wrong.",
    )
    step: str = Field(
        ...,
        description="Name of the workflow step where the failure occurred.",
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured details to aid debugging or UI display.",
    )


class WorkflowContext(BaseModel):
    """
    Canonical, shared context object for all orchestrated workflows.

    Created at workflow start and passed through every step. It:
    - Carries workflow metadata (type, ids, jurisdiction)
    - Accumulates intermediate_results keyed by step name
    - Tracks current_step during execution
    - Holds final status + optional WorkflowError
    """

    # Identity / type
    workflow_id: str = Field(
        ...,
        description="Unique identifier for this workflow run (e.g. uuid4 hex).",
    )
    workflow_type: Literal[
        "contract_review",
        "contract_comparison",
        "due_diligence",
        "document_explorer",
    ] = Field(
        ...,
        description="Type of workflow being executed.",
    )

    # Inputs
    document_ids: List[str] = Field(
        default_factory=list,
        description="Primary document IDs involved in this workflow (e.g. [contract_id] or [contract_id, template_id]).",
    )
    jurisdiction: Optional[str] = Field(
        default=None,
        description="Jurisdiction code/name (e.g. 'KSA', 'UAE', 'Generic GCC').",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional input parameters and workflow configuration.",
    )

    # Execution state
    intermediate_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-step intermediate outputs, keyed by step name.",
    )
    current_step: Optional[str] = Field(
        default=None,
        description="Name of the step currently executing or most recently executed.",
    )

    # Outcome
    status: Literal["running", "completed", "completed_with_warnings", "failed"] = Field(
        "running",
        description='Execution status: "running", "completed", "completed_with_warnings", or "failed".',
    )
    error: Optional[WorkflowError] = Field(
        default=None,
        description="Populated when status == 'failed' with structured error information.",
    )
    final_output: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Canonical final payload for workflow outputs (None when refused/failed).",
    )

    def add_result(self, step: str, result: Any) -> None:
        """
        Convenience helper to store an intermediate result for a step.
        Does not mutate status or error.
        """
        self.intermediate_results[step] = result
        self.current_step = step

    def fail(self, *, code: str, message: str, step: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Convenience helper to mark the context as failed with a WorkflowError
        while preserving all accumulated intermediate_results.
        """
        self.status = "failed"
        self.current_step = step
        self.error = WorkflowError(code=code, message=message, step=step, details=details or {})


class DocumentExplorerRequest(BaseModel):
    """
    Request model for Document Explorer workflow.

    NOTE: This is used for internal orchestration / validation. The public API
    response shape for the explorer is still the simple JSON with `results`
    as specified in the PRD.
    """

    document_id: str = Field(..., description="Single document ID to search within.")
    query: str = Field(..., description="Query for evidence retrieval.")
    mode: Literal["text", "clauses"] = Field(
        default="text",
        description="Search mode: raw text chunks or extracted clauses.",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Optional limit for number of results (defaults to config).",
    )


class EvidenceExplorerRequest(BaseModel):
    """Request for Evidence Explorer (evidence-only, no LLM)."""

    document_id: str = Field(..., description="Single document ID to search within.")
    query: str = Field(..., description="Query for evidence retrieval.")
    mode: Literal["text", "clauses", "both"] = Field(
        default="text",
        description="Search mode: text chunks, extracted clauses, or both (clauses first, then text fallback).",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Optional limit for number of results (defaults to config).",
    )


class EvidenceExplorerResult(BaseModel):
    """Single evidence hit (chunk or clause)."""

    document_id: str = Field(..., description="Source document ID.")
    page_number: int = Field(..., description="Page number (chunk) or page_start (clause).")
    chunk_index: int = Field(default=0, description="Chunk index (0 for clause results).")
    text_snippet: str = Field(..., description="Relevant text snippet.")
    score: float = Field(..., description="Relevance score.")
    source_type: Literal["chunk", "clause"] = Field(..., description="Whether from chunk or clause.")
    display_name: Optional[str] = Field(None, description="User-friendly document name.")
    citation: Optional[str] = Field(None, description="Formatted citation if available.")
    clause_id: Optional[str] = Field(None, description="Clause ID when source_type is clause.")
    page_end: Optional[int] = Field(None, description="End page for clause span.")


class DocumentExplorerResult(BaseModel):
    """Single search hit for Document Explorer (evidence snippet)."""

    document_id: str = Field(..., description="Source document ID.")
    page_number: int = Field(..., description="Page number where the snippet appears.")
    chunk_index: int = Field(..., description="Chunk index within the document.")
    text_snippet: str = Field(..., description="Relevant text snippet.")
    score: float = Field(..., description="Relevance score from retrieval.")
    display_name: Optional[str] = Field(None, description="User-friendly document display name.")
    citation: Optional[str] = Field(None, description="Formatted citation if available.")


class DocumentExplorerResponse(BaseModel):
    """
    Response model for Document Explorer workflow (success path).

    Matches the PRD shape when serialized:
    {
      \"results\": [{...}]
    }
    """

    results: List[DocumentExplorerResult] = Field(
        default_factory=list,
        description="List of evidence results for the query.",
    )
    answer: Optional[str] = Field(None, description="RAG answer for the query.")
    status: Optional[str] = Field(None, description="RAG status (explicitly_stated, not_specified, refused).")
    confidence: Optional[str] = Field(None, description="RAG confidence level if available.")
    citation: Optional[str] = Field(None, description="Primary citation if available.")
    refusal_reason: Optional[str] = Field(None, description="Refusal reason if status is refused.")
    reason: Optional[str] = Field(None, description="Reason for not_found responses.")
    debug: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Debug metadata for QA and troubleshooting.",
    )


class EvidenceExplorerResponse(BaseModel):
    """
    Evidence-only response (no answer field).
    Used by POST /api/explore-evidence.
    """

    status: Literal["ok", "not_found"] = Field(..., description="ok if any results, not_found otherwise.")
    results: List[EvidenceExplorerResult] = Field(
        default_factory=list,
        description="Evidence snippets (chunks or clauses).",
    )
    reason: Optional[str] = Field(None, description="Reason when status is not_found.")
    debug: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Debug metadata (mode, counts, expanded_keywords, etc.).",
    )


