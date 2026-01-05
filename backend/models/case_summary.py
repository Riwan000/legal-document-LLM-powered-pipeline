"""
Pydantic models for case summarization output schema.
Strict JSON schema with mandatory citations.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SourceCitation(BaseModel):
    """Source citation for a claim."""
    document: str = Field(..., description="Document ID")
    page: int = Field(..., description="Page number")
    chunk_id: str = Field(..., description="Stable chunk identifier")


class ExecutiveSummaryItem(BaseModel):
    """Single item in executive summary with citation."""
    text: str = Field(..., description="Summary text")
    source: SourceCitation = Field(..., description="Source citation")


class TimelineEvent(BaseModel):
    """Timeline event with citation."""
    date: str = Field(..., description="Date or time period")
    event: str = Field(..., description="Event description")
    source: SourceCitation = Field(..., description="Source citation")


class ArgumentItem(BaseModel):
    """Single argument with citation."""
    text: str = Field(..., description="Argument text")
    source: SourceCitation = Field(..., description="Source citation")


class KeyArguments(BaseModel):
    """Key arguments organized by party."""
    claimant: List[ArgumentItem] = Field(default_factory=list, description="Claimant/Plaintiff arguments")
    defendant: List[ArgumentItem] = Field(default_factory=list, description="Defendant/Respondent arguments")


class OpenIssue(BaseModel):
    """Open issue with citation."""
    text: str = Field(..., description="Issue description")
    source: SourceCitation = Field(..., description="Source citation")


class CaseSpine(BaseModel):
    """Case spine - foundational structure."""
    case_name: str = Field(..., description="Name of the case")
    court: str = Field(..., description="Court name")
    date: str = Field(..., description="Case date")
    parties: List[str] = Field(..., description="List of parties")
    procedural_posture: str = Field(..., description="Procedural posture")
    core_issues: List[str] = Field(..., description="Core legal issues")


class CitationMetadata(BaseModel):
    """Full citation metadata for reference."""
    document: str = Field(..., description="Document ID")
    page: int = Field(..., description="Page number")
    chunk_id: str = Field(..., description="Chunk identifier")
    chunk_type: str = Field(..., description="Chunk type classification")


class CaseSummary(BaseModel):
    """Complete case summary with strict schema."""
    case_spine: CaseSpine = Field(..., description="Case spine (mandatory)")
    executive_summary: List[ExecutiveSummaryItem] = Field(default_factory=list, description="Executive summary items")
    timeline: List[TimelineEvent] = Field(default_factory=list, description="Timeline of events")
    key_arguments: KeyArguments = Field(default_factory=KeyArguments, description="Key arguments by party")
    open_issues: List[OpenIssue] = Field(default_factory=list, description="Open/unresolved issues")
    citations: List[CitationMetadata] = Field(default_factory=list, description="All citations used")


class CaseSummaryError(BaseModel):
    """Structured error response."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Error details")

