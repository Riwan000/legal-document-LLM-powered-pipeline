"""
Legal clause data models.
Structured representation of legal clauses with authority, hierarchy, and evidence.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class AuthorityLevel(str, Enum):
    """Authority levels for legal clauses."""
    SUPREME = "supreme"  # External law (e.g., Saudi Labour Law)
    REGULATORY = "regulatory"  # Compliance laws (e.g., Emigration Act)
    CONTRACTUAL = "contractual"  # Agreement terms
    ADMINISTRATIVE = "administrative"  # Instructions, dates, notices


class ClauseType(str, Enum):
    """Top-level legal clause categories."""
    GOVERNING_LAW = "governing_law"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    TERMINATION = "termination"
    COMPENSATION_BENEFITS = "compensation_benefits"
    SALARY_WAGES = "salary_wages"
    EMPLOYER_OBLIGATIONS = "employer_obligations"
    EMPLOYEE_OBLIGATIONS = "employee_obligations"
    CONDUCT_DISCIPLINE = "conduct_discipline"
    JURISDICTION = "jurisdiction"
    DISPUTE_RESOLUTION = "dispute_resolution"
    CONFIDENTIALITY = "confidentiality"
    LIABILITY = "liability"
    OTHER = "other"


class TerminationSubtype(str, Enum):
    """Termination clause subtypes."""
    TERMINATION_RIGHTS = "termination_rights"
    TERMINATION_NOTICE = "termination_notice"
    PROBATION_TERMINATION = "probation_termination"
    END_OF_SERVICE_COMPENSATION = "end_of_service_compensation"
    DEATH_DISABILITY = "death_disability"
    OTHER = "other"


class EvidenceBlock(BaseModel):
    """Evidence block representing where a clause appears in the document."""
    page: int = Field(..., description="Page number")
    paragraph: Optional[int] = Field(None, description="Paragraph index")
    line_start: Optional[int] = Field(None, description="Starting line number")
    line_end: Optional[int] = Field(None, description="Ending line number")
    raw_text: str = Field(..., description="Original OCR output")
    clean_text: str = Field(..., description="Normalized text")


class StructuredClause(BaseModel):
    """Structured legal clause with authority, hierarchy, and evidence."""
    clause_id: str = Field(..., description="Unique clause identifier")
    title: str = Field(..., description="Clause title/heading")
    type: ClauseType = Field(..., description="Top-level clause category")
    subtype: Optional[str] = Field(None, description="Subtype (e.g., termination_notice)")
    authority_level: AuthorityLevel = Field(..., description="Authority level")
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction (e.g., Saudi Arabia)")
    can_override_contract: bool = Field(False, description="Whether this clause can override contract terms")
    overrides: List[str] = Field(default_factory=list, description="List of authority levels this overrides")
    evidence: List[EvidenceBlock] = Field(default_factory=list, description="Evidence blocks (page references)")
    explicitly_provided: bool = Field(True, description="Whether explicitly stated (vs implied)")
    linked_clause_id: Optional[str] = Field(None, description="Linked clause ID for bilingual versions")
    consistency_flag: Optional[str] = Field(None, description="Consistency flag if mismatch detected")
    language: Optional[str] = Field(None, description="Language code ('ar', 'en')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True

