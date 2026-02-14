"""
ConversationSummarizer — compresses older session turns into a summary.

Triggered by SessionManager when:
  - Turn count exceeds MAX_SESSION_TURNS, OR
  - Token count approaches MAX_SESSION_TOKENS (≥ 85% of budget)

Behaviour:
  - Preserves the most recent SESSION_CONTEXT_WINDOW * 2 messages intact.
  - Summarizes all older messages into session.summary via a local Ollama call.
  - Never invents facts — instructed to only summarise what was said.
  - Preserves all clause references (clause_id, page numbers) verbatim.

Rules enforced via prompt:
  1. Do not add legal opinions or interpretations.
  2. Preserve all clause and page number references exactly as mentioned.
  3. Use past tense ("The user asked…", "The assistant explained…").
  4. If prior summary exists, merge it with new older turns rather than discarding it.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ollama

from backend.config import settings

if TYPE_CHECKING:
    from backend.models.session import ChatSession

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM_PROMPT = """You are a conversation summarizer for a legal document Q&A system.
Your task is to compress a sequence of conversation turns into a concise summary.

Rules — follow strictly:
1. Preserve all references to specific clauses, page numbers, and document sections exactly as mentioned.
2. Do NOT add legal interpretations, opinions, or conclusions beyond what was stated.
3. Use past tense: "The user asked about...", "The assistant explained...".
4. Be concise: one or two sentences per key topic discussed.
5. If a prior summary is provided, merge it with the new turns — do not discard prior context.
6. Output ONLY the summary text — no headings, no bullet points, no explanation."""


class ConversationSummarizer:
    """Summarizes older session turns into session.summary, keeping recent turns intact."""

    def __init__(self):
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        # Number of recent messages to keep intact (untouched by summarization)
        self._keep_recent = settings.SESSION_CONTEXT_WINDOW * 2

    def summarize(self, session: "ChatSession") -> None:
        """
        Mutate session in-place:
        - Summarize messages older than the keep_recent window into session.summary.
        - Replace those older messages with a synthetic "summary" message marker.
        - Retain the most recent _keep_recent messages unchanged.

        The caller (SessionManager) is responsible for persisting the session afterward.
        """
        messages = session.messages
        if len(messages) <= self._keep_recent:
            return  # Nothing to summarize yet

        older = messages[: -self._keep_recent]
        recent = messages[-self._keep_recent :]

        # Build the text to summarize
        turns_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in older
        )

        prior_summary = session.summary or ""
        if prior_summary:
            user_prompt = (
                f"Prior summary:\n{prior_summary}\n\n"
                f"New conversation turns to merge into the summary:\n{turns_text}\n\n"
                f"Updated summary:"
            )
        else:
            user_prompt = (
                f"Conversation turns to summarize:\n{turns_text}\n\n"
                f"Summary:"
            )

        try:
            response = self.client.chat(
                model=settings.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.0, "seed": 42},
            )
            new_summary = response["message"]["content"].strip()
            session.summary = new_summary
            session.messages = list(recent)
            logger.info(
                "ConversationSummarizer: compressed %d older messages for session %s",
                len(older),
                session.session_id,
            )
        except Exception as exc:
            logger.warning(
                "ConversationSummarizer: LLM call failed for session %s: %s — keeping original messages",
                session.session_id,
                exc,
            )
            # On failure, keep all messages and do not update summary
