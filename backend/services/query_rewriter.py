"""
QueryRewriter — pronoun-expansion rewriter for conversational RAG.

Expands pronouns ("it", "that clause", "the previous one") using the recent
conversation history so that the rewritten query is self-contained and
suitable for document retrieval.

Rules enforced via prompt:
- Expand pronouns only — never add legal interpretation.
- Never introduce external laws or facts not present in the history.
- Keep the query document-scoped.
- If no pronouns are detected, return the original question unchanged
  (avoids unnecessary LLM calls).
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import ollama

from backend.config import settings
from backend.models.session import ChatSession, SessionMessage

logger = logging.getLogger(__name__)

# Pronouns that warrant rewriting when found in the user question
_PRONOUN_PATTERN = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|"
    r"the previous one|the above|the clause|the section|"
    r"the same|the latter|the former)\b",
    re.IGNORECASE,
)

_REWRITE_SYSTEM_PROMPT = """You are a query rewriter for a legal document retrieval system.
Your ONLY job is to expand pronouns in the user's question using the conversation history.

Rules — follow strictly:
1. Replace pronouns ("it", "that clause", "the previous one", etc.) with the explicit noun they refer to.
2. Do NOT add legal interpretations, conclusions, or opinions.
3. Do NOT introduce any laws, regulations, or facts not mentioned in the conversation history.
4. Keep the rewritten question scoped to the document under discussion.
5. If no pronouns need expanding, return the original question UNCHANGED.
6. Output ONLY the rewritten question — no explanation, no prefix, no quotes."""


class QueryRewriter:
    """Rewrites a follow-up question to be self-contained for retrieval."""

    def __init__(self):
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)

    def rewrite(
        self,
        history: List[SessionMessage],
        question: str,
        document_id: str,
        session: Optional["ChatSession"] = None,
    ) -> str:
        """
        Rewrite `question` using `history` to expand pronouns.

        Args:
            history:     Recent SessionMessage list (role + content).
            question:    The current user question.
            document_id: The document being discussed (for logging only).
            session:     Optional ChatSession for state-aware context injection (Phase 5).

        Returns:
            The rewritten query, or the original question if no pronouns found
            or if the LLM call fails.
        """
        # Fast-path: skip LLM call if no pronouns present
        if not _PRONOUN_PATTERN.search(question):
            return question

        if not history:
            return question

        # Build history context (last 6 messages max to stay compact)
        recent = history[-6:]
        history_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in recent
        )

        # Phase 5: prepend session state hint if available
        session_hint = ""
        if session is not None:
            hint_parts = []
            if getattr(session, "conversation_goal", None):
                hint_parts.append(f"Conversation goal: {session.conversation_goal}")
            identified = getattr(session, "identified_terms", [])
            if identified:
                hint_parts.append(f"Previously identified terms: {', '.join(identified[:5])}")
            if hint_parts:
                session_hint = "[Session context] " + "; ".join(hint_parts) + "\n\n"

        user_prompt = (
            f"{session_hint}"
            f"Conversation history:\n{history_text}\n\n"
            f"Current question: {question}\n\n"
            f"Rewritten question:"
        )

        try:
            response = self.client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.0, "seed": 42},
            )
            rewritten = response["message"]["content"].strip()

            # Safety: if the rewriter returns something very short or empty, fall back
            if not rewritten or len(rewritten) < 5:
                return question

            logger.debug(
                "QueryRewriter [%s]: %r → %r", document_id, question[:80], rewritten[:80]
            )
            return rewritten

        except Exception as exc:
            logger.warning("QueryRewriter failed, using original question: %s", exc)
            return question
