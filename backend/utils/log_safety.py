"""
Logging safety utilities to prevent log injection / forging
(CWE-117 / SonarQube S5145).

User-controlled values (document ids, file paths, session ids, queries) must be
neutralized before they are written to a log record. Without this, a value
containing CR/LF or other control characters could split a log line and forge
additional entries, or corrupt structured (JSON) logs.
"""
import re

# Control characters (incl. CR, LF, TAB, NUL, DEL) that can forge/split log lines.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# Default cap so a hostile or oversized value cannot flood the logs.
DEFAULT_MAX_LOG_LENGTH = 200


def sanitize_for_log(value: object, max_length: int = DEFAULT_MAX_LOG_LENGTH) -> str:
    """
    Return a single-line, length-bounded, control-character-free representation
    of ``value`` that is safe to embed in a log record.

    Args:
        value: Any value (coerced to ``str``); typically user-controlled.
        max_length: Maximum length before truncation.

    Returns:
        A sanitized string with control characters replaced by spaces.
    """
    text = _CONTROL_CHARS.sub(" ", str(value))
    if len(text) > max_length:
        text = text[:max_length] + "…"
    return text
