"""
SessionManager — orchestration layer over SessionStore.

Responsibilities:
- Validate document_id consistency (cannot change mid-session).
- Enforce turn count (MAX_SESSION_TURNS) and token budget (MAX_SESSION_TOKENS).
- Return a context window (last N exchanges) for the orchestrator.
- Track last_clause_id when assistant messages contain clause citations.
- Trigger conversation summarization when limits are approached.

This class never calls the LLM directly.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, TYPE_CHECKING

from backend.config import settings
from backend.models.session import ChatSession, SessionMessage, SessionMode
from backend.services.session_store import SessionStore
from backend.utils.token_counter import count_session_tokens

if TYPE_CHECKING:
    from backend.services.conversation_summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)

# Pattern to extract clause IDs from assistant messages (e.g. "Clause 5", "clause_id: abc-123")
_CLAUSE_REF_PATTERN = re.compile(
    r"(?:clause[_\s-]id[:\s]+([a-zA-Z0-9_-]+))",
    re.IGNORECASE,
)


class SessionNotFoundError(Exception):
    pass


class DocumentMismatchError(Exception):
    pass


class SessionManager:
    """High-level session lifecycle manager."""

    def __init__(
        self,
        store: Optional[SessionStore] = None,
        summarizer: Optional["ConversationSummarizer"] = None,
    ):
        self.store = store or SessionStore()
        self.summarizer = summarizer  # injected after ConversationSummarizer is created

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, document_id: str, mode: SessionMode) -> ChatSession:
        return self.store.create_session(document_id, mode)

    def get_session(self, session_id: str) -> ChatSession:
        """Load session; raise SessionNotFoundError if missing or expired."""
        session = self.store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found or has expired.")
        return session

    def delete_session(self, session_id: str) -> bool:
        return self.store.delete_session(session_id)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def append_user_message(self, session_id: str, content: str) -> ChatSession:
        """Append a user message to the session and persist."""
        session = self.get_session(session_id)
        msg = SessionMessage(role="user", content=content)
        session.messages.append(msg)
        self.store.save_session(session)
        return session

    def append_assistant_message(
        self,
        session_id: str,
        content: str,
        trace=None,
    ) -> ChatSession:
        """
        Append an assistant message (with optional RetrievalTrace) and persist.
        Also extracts and updates last_clause_id if a clause reference is found.
        """
        session = self.get_session(session_id)
        msg = SessionMessage(role="assistant", content=content, trace=trace)
        session.messages.append(msg)

        # Update last_clause_id if the answer references one
        clause_match = _CLAUSE_REF_PATTERN.search(content)
        if clause_match:
            session.last_clause_id = clause_match.group(1)

        self.store.save_session(session)
        return session

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def get_context(self, session_id: str) -> List[SessionMessage]:
        """
        Return the context window (last N exchanges) for the LLM.

        If the assembled context exceeds MAX_SESSION_TOKENS, older turns are
        truncated (not the most recent SESSION_CONTEXT_WINDOW turns) until
        the budget is satisfied.
        """
        session = self.get_session(session_id)
        messages = session.messages

        # Keep at most the last SESSION_CONTEXT_WINDOW * 2 messages (user+assistant pairs)
        window_size = settings.SESSION_CONTEXT_WINDOW * 2
        window = messages[-window_size:] if len(messages) > window_size else messages[:]

        # Token budget enforcement: truncate from the front
        while len(window) > 2 and count_session_tokens(window) > settings.MAX_SESSION_TOKENS:
            window = window[2:]  # drop oldest user+assistant pair
            logger.warning(
                "Session %s: context exceeded %d tokens; truncating oldest turn.",
                session_id,
                settings.MAX_SESSION_TOKENS,
            )

        return window

    # ------------------------------------------------------------------
    # Turn / token enforcement
    # ------------------------------------------------------------------

    def enforce_limits(self, session_id: str) -> None:
        """
        Check turn count and token budget; trigger summarization if configured.

        Called before appending a new user message (at the start of each turn).
        """
        session = self.get_session(session_id)
        turn_count = sum(1 for m in session.messages if m.role == "user")
        total_tokens = count_session_tokens(session.messages)

        near_turn_limit = turn_count >= settings.MAX_SESSION_TURNS
        near_token_limit = total_tokens >= int(settings.MAX_SESSION_TOKENS * 0.85)

        if (near_turn_limit or near_token_limit) and self.summarizer is not None:
            logger.info(
                "Session %s approaching limits (turns=%d, tokens=%d). Triggering summarization.",
                session_id, turn_count, total_tokens,
            )
            self.summarizer.summarize(session)
            self.store.save_session(session)

    def validate_document(self, session_id: str, document_id: str) -> None:
        """Raise DocumentMismatchError if document_id differs from session's."""
        session = self.get_session(session_id)
        if session.document_id != document_id:
            raise DocumentMismatchError(
                f"Session '{session_id}' is bound to document '{session.document_id}', "
                f"not '{document_id}'."
            )
