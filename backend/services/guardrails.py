"""
Post-generation guardrail enforcement (FIX 3).

Scans generated text for forbidden prescriptive language and fails closed.
Runs only on LLM-generated output (executive summaries, risk descriptions,
due diligence memo sections), never on raw extracted text.
"""
from __future__ import annotations

from typing import Any, List

# Mandatory disclaimer required by PRD v0.2 guardrails.
# IMPORTANT: keep this non-prescriptive (no "should/must/recommend").
WORKFLOW_DISCLAIMER = (
    "This system does not provide legal advice. "
    "All outputs are for review purposes only."
)

# Terms that indicate prescriptive / advisory language.
# Keep this list conservative and apply only to generated text (never raw evidence).
FORBIDDEN_TERMS = [
    "should",
    "must",
    "recommend",
    "recommended",
    "recommendation",
    "we recommend",
    "you should",
    "you must",
    "legally required",
    "advised to",
]


class GuardrailViolation(Exception):
    """Raised when generated text contains prescriptive language."""

    def __init__(self, code: str, step: str, message: str, details: dict):
        self.code = code
        self.step = step
        self.message = message
        self.details = details
        super().__init__(message)


def enforce_non_prescriptive_language(text: str, step: str = "generation") -> str:
    """
    Scan text for forbidden prescriptive terms. If any are found, raise
    GuardrailViolation so the orchestrator can fail the workflow with
    ctx.error (code=PRESCRIPTIVE_LANGUAGE).

    Use only on generated text (executive summaries, risk descriptions,
    due diligence sections). Never run on raw extracted clauses.
    """
    if not text or not text.strip():
        return text
    lower = text.lower()
    violations = [t for t in FORBIDDEN_TERMS if t in lower]
    if violations:
        raise GuardrailViolation(
            code="PRESCRIPTIVE_LANGUAGE",
            step=step,
            message="Generated output contained prescriptive legal language",
            details={"terms": violations},
        )
    return text


def enforce_non_prescriptive_language_in_obj(obj: Any, *, step: str) -> Any:
    """
    Recursively enforce non-prescriptive language on all strings inside obj.

    Intended for structured LLM outputs (dict/list/pydantic model_dump output).
    """
    if obj is None:
        return obj
    if isinstance(obj, str):
        return enforce_non_prescriptive_language(obj, step=step)
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = enforce_non_prescriptive_language_in_obj(obj[i], step=step)
        return obj
    if isinstance(obj, tuple):
        return tuple(enforce_non_prescriptive_language_in_obj(x, step=step) for x in obj)
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = enforce_non_prescriptive_language_in_obj(v, step=step)
        return obj
    # Unknown type: leave as-is (caller should apply to model_dump if needed).
    return obj
