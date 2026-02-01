"""
Data models package for the Legal Document Intelligence system.

Most code imports concrete models directly from their modules, e.g.:
    from backend.models.document import DocumentUploadResponse

This __init__ is intentionally minimal; it only re-exports a few shared
workflow-level models for convenience where appropriate.
"""

from .workflow import (
    WorkflowContext,
    WorkflowError,
    DocumentExplorerRequest,
    DocumentExplorerResponse,
    DocumentExplorerResult,
)  # noqa: F401
from .contract_review import (
    ContractReviewRequest,
    ContractReviewResponse,
    RiskItem,
    ClauseEvidenceBlock,
    ExecutiveSummaryItem,
)  # noqa: F401
