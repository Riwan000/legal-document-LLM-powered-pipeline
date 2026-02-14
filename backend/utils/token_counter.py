"""
Token counting utility using tiktoken (cl100k_base encoding).

Qwen2.5 uses a tiktoken-based tokenizer compatible with cl100k_base,
so this gives accurate token counts without requiring the full model download.

Requires: pip install tiktoken
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from backend.models.session import SessionMessage

logger = logging.getLogger(__name__)

# Lazy-loaded encoder — initialized once on first use.
_encoder = None


def _get_encoder():
    """Return the tiktoken encoder, initializing it on first call."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning(
                "tiktoken not installed. Falling back to char-based approximation. "
                "Run: pip install tiktoken"
            )
            _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """
    Count tokens in a string using tiktoken (cl100k_base / Qwen2.5).

    Falls back to len(text) // 4 if tiktoken is unavailable.
    """
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    # Fallback: character-based approximation
    return max(1, len(text) // 4)


def count_session_tokens(messages: "List[SessionMessage]") -> int:
    """
    Count total tokens across all messages in a session.

    Adds 4 tokens per message for role/structure overhead
    (matches OpenAI's per-message overhead convention).
    """
    total = 0
    for msg in messages:
        total += 4  # role + structural overhead
        total += count_tokens(msg.content)
    return total
