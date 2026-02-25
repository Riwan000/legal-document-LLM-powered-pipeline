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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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

# Cross-turn consistency: extract document-level facts from LLM answers
_DOC_TYPE_PATTERN = re.compile(
    r"this\s+is\s+(?:a|an)\s+(NDA|Non-Disclosure Agreement|Employment Agreement|"
    r"Master Service Agreement|MSA|Professional Services Agreement|"
    r"Service Level Agreement|SLA|Consultancy Agreement|Supply Agreement|"
    r"Framework Agreement|Joint Venture Agreement)[^.]*\.",
    re.IGNORECASE,
)
_JURISDICTION_PATTERN = re.compile(
    r"governed\s+by\s+(?:the\s+)?(?:laws?\s+of\s+)?"
    r"(KSA|Saudi Arabia|UAE|United Arab Emirates|English law|"
    r"laws of England|Bahrain|Qatar|Kuwait|Oman)[^.]*\.",
    re.IGNORECASE,
)
_PARTY_PATTERN = re.compile(
    r'(?:between|by\s+and\s+between)\s+"?([A-Z][A-Za-z\s&.,]+?)"?\s+(?:and|,)',
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

        # Phase 5: infer conversation goal on first user message only
        if session.conversation_goal is None:
            session.conversation_goal = self._infer_conversation_goal(content)

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
        Also extracts and updates last_clause_id, identified_terms, and risk_flags.
        """
        session = self.get_session(session_id)
        msg = SessionMessage(role="assistant", content=content, trace=trace)
        session.messages.append(msg)

        # Update last_clause_id if the answer references one
        clause_match = _CLAUSE_REF_PATTERN.search(content)
        if clause_match:
            session.last_clause_id = clause_match.group(1)

        # Phase 5: extract defined terms and detect risk flags
        new_terms = self._extract_defined_terms_from_answer(content)
        for t in new_terms:
            if t not in session.identified_terms:
                session.identified_terms.append(t)

        new_flags = self._detect_risk_flags(content, session.risk_flags)
        session.risk_flags = list(set(session.risk_flags) | set(new_flags))

        # Cross-turn consistency: extract and cache document-level facts
        new_facts = self._extract_established_facts(content)
        for key, value in new_facts.items():
            if key not in session.established_facts:  # never overwrite earlier confident facts
                session.established_facts[key] = value

        self.store.save_session(session)
        return session

    # ------------------------------------------------------------------
    # Phase 5 private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_conversation_goal(message: str) -> str:
        """Keyword-pattern goal inference (no LLM call). Called on first user message."""
        lower = message.lower()
        if any(kw in lower for kw in ("terminat", "dismiss")):
            return "understand_termination_rights"
        if any(kw in lower for kw in ("confidential", "nda", "disclosure")):
            return "check_confidentiality"
        if any(kw in lower for kw in ("salary", "wage", "compensation", "pay")):
            return "review_compensation"
        if any(kw in lower for kw in ("liable", "liability", "damages")):
            return "understand_liability"
        if any(kw in lower for kw in ("governing law", "jurisdiction")):
            return "check_governing_law"
        if any(kw in lower for kw in ("assign", "transfer", "novation")):
            return "understand_assignment"
        return "general_exploration"

    @staticmethod
    def _extract_defined_terms_from_answer(answer: str) -> List[str]:
        """Extract capitalized quoted terms from assistant answer."""
        pattern = re.compile(r'["\u201c]([A-Z][A-Za-z\s]+)["\u201d]')
        return pattern.findall(answer)

    _RISK_PATTERNS = {
        "unilateral_termination_by_employer": re.compile(
            r"employer\s+(may|can|shall)\s+terminate\s+without\s+(cause|reason|notice)",
            re.IGNORECASE,
        ),
        "no_notice_period": re.compile(
            r"no\s+notice\s+(period|required)", re.IGNORECASE
        ),
        "unlimited_liability": re.compile(r"unlimited\s+liability", re.IGNORECASE),
        "broad_assignment_rights": re.compile(
            r"(employer|company)\s+(may|can)\s+assign", re.IGNORECASE
        ),
        "non_compete": re.compile(r"non[-\s]?compete", re.IGNORECASE),
    }

    @classmethod
    def _detect_risk_flags(cls, answer: str, existing_flags: List[str]) -> List[str]:
        """Detect risk patterns in assistant answer; return list of new flag names."""
        new_flags = []
        for flag_name, pattern in cls._RISK_PATTERNS.items():
            if flag_name not in existing_flags and pattern.search(answer):
                new_flags.append(flag_name)
        return new_flags

    @staticmethod
    def _extract_established_facts(answer: str) -> Dict[str, Any]:
        """
        Extract document-level facts from an LLM answer for cross-turn consistency.

        Matches only high-confidence phrasings (e.g. "this is a/an [X]") to avoid
        caching hallucinations or hedged statements.

        Returns a dict with any of: document_type (str), jurisdiction (str),
        party_names (List[str]).  Only includes keys where a match was found.
        """
        facts: Dict[str, Any] = {}

        doc_match = _DOC_TYPE_PATTERN.search(answer)
        if doc_match:
            facts["document_type"] = doc_match.group(1).strip()

        jur_match = _JURISDICTION_PATTERN.search(answer)
        if jur_match:
            facts["jurisdiction"] = jur_match.group(1).strip()

        party_matches = _PARTY_PATTERN.findall(answer)
        if party_matches:
            facts["party_names"] = [p.strip() for p in party_matches[:3]]

        return facts

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
