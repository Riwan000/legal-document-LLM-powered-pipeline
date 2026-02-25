"""
Pydantic models for conversational RAG session management.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class SessionMode(str, Enum):
    STRICT = "strict"
    CONVERSATIONAL = "conversational"


class RetrievalTrace(BaseModel):
    """Structured audit trace for a single retrieval turn."""
    original_query: str
    rewritten_query: Optional[str] = None
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    similarity_scores: List[float] = Field(default_factory=list)
    guardrail_decision: Optional[str] = None   # pass | weak | fail
    evidence_score: Optional[str] = None       # strong | moderate | weak | none


class SessionMessage(BaseModel):
    """A single message in the chat session."""
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace: Optional[RetrievalTrace] = None


class ChatSession(BaseModel):
    """Full session state stored in SQLite."""
    session_id: str
    document_id: str
    mode: SessionMode = SessionMode.STRICT
    messages: List[SessionMessage] = Field(default_factory=list)
    summary: Optional[str] = None        # compressed summary of older turns
    last_clause_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    # Phase 1 — RAG 5-Layer Upgrade: richer session state
    conversation_goal: Optional[str] = None
    identified_terms: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    # Cross-turn consistency: document-level facts extracted from LLM answers
    # Keys: "document_type", "jurisdiction", "party_names" (List[str])
    established_facts: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    document_id: str
    mode: SessionMode = SessionMode.STRICT


class CreateSessionResponse(BaseModel):
    session_id: str
    document_id: str
    mode: SessionMode
    created_at: datetime


class ChatMessageRequest(BaseModel):
    message: str
    mode: Optional[SessionMode] = None   # override per-message; falls back to session mode


class ChatSource(BaseModel):
    document_id: str
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    text_snippet: Optional[str] = None
    score: Optional[float] = None
    citation: Optional[str] = None


class ChatMessageResponse(BaseModel):
    session_id: str
    answer: str
    status: str                          # ok | refused
    evidence_score: Optional[str] = None
    guardrail_decision: Optional[str] = None
    sources: List[ChatSource] = Field(default_factory=list)
    trace: Optional[RetrievalTrace] = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    document_id: str
    mode: SessionMode
    messages: List[SessionMessage]
    summary: Optional[str] = None
    last_active_at: datetime
