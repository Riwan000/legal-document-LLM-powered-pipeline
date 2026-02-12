"""
Canonical workflow stages and stage status.

Single source of truth for UI progress: no guessing, only backend state.
"""
from enum import Enum

from pydantic import BaseModel, Field


class WorkflowStage(str, Enum):
    UPLOAD_EXTRACTION = "upload_extraction"
    LEGAL_ANALYSIS = "legal_analysis"
    REPORT_GENERATION = "report_generation"


class StageStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class WorkflowState(BaseModel):
    """Per-stage status for the three workflow stages. UI reads this only."""

    upload_extraction: StageStatus = Field(
        default=StageStatus.PENDING,
        description="Upload & Extraction stage.",
    )
    legal_analysis: StageStatus = Field(
        default=StageStatus.PENDING,
        description="Legal Analysis stage.",
    )
    report_generation: StageStatus = Field(
        default=StageStatus.PENDING,
        description="Report Generation stage.",
    )
