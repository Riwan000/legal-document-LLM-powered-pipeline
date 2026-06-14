from __future__ import annotations

"""
Deterministic rule-based planner (Phase 1).

Routes a natural-language request to exactly one core tool via an ordered
keyword table, then finalizes. No LLM involved; the LLM planner (Phase 2)
implements the same next_action interface and is a drop-in replacement.

Route order is deliberate: workflow routes are checked BEFORE the
QueryClassifier out-of-scope check, because OUT_OF_SCOPE_PATTERNS
("court judgment", "case law", ...) legitimately appear in litigation
summarization requests (pillar 3).
"""

from typing import Any, List, Optional

from backend.models.agent import (
    AgentScope,
    AgentStep,
    PlannerDecision,
    PlannerFinalAnswer,
    PlannerRefusal,
    PlannerToolCall,
    ToolCall,
)

_DD_KEYWORDS = ("due diligence", "dd memo")
_COMPARE_KEYWORDS = ("compare", "deviat", "redline", "against the template")
_REVIEW_KEYWORDS = ("review", "risk", "missing clause")
_SUMMARIZE_KEYWORDS = ("summar", "chronolog", "timeline", "case file")

_UNSUPPORTED_REASON = (
    "This request does not match a supported workflow. Supported: contract review, "
    "document comparison, due diligence memo, and case-file summarization."
)
_OUT_OF_SCOPE_REASON = (
    "This request is outside the scope of document-grounded legal workflows."
)


class RuleBasedPlanner:
    """One-shot deterministic router. First matching route wins."""

    def __init__(self, query_classifier: Optional[Any] = None):
        self._query_classifier = query_classifier

    def next_action(
        self, request: str, scope: AgentScope, history: List[AgentStep]
    ) -> PlannerDecision:
        if history:
            # One tool per run: finalize on success; refuse otherwise.
            last = history[-1]
            if last.result_status == "ok":
                return PlannerFinalAnswer(use_results=[last.step_index])
            return PlannerRefusal(reason="The selected workflow did not complete successfully.")

        text = request.lower()

        if any(keyword in text for keyword in _DD_KEYWORDS):
            return PlannerToolCall(call=ToolCall(tool_name="due_diligence_memo", arguments={}))

        if any(keyword in text for keyword in _COMPARE_KEYWORDS) and len(scope.document_ids) >= 2:
            return PlannerToolCall(call=ToolCall(tool_name="compare_documents", arguments={}))

        if any(keyword in text for keyword in _REVIEW_KEYWORDS):
            return PlannerToolCall(call=ToolCall(tool_name="review_contract", arguments={}))

        if any(keyword in text for keyword in _SUMMARIZE_KEYWORDS):
            return PlannerToolCall(call=ToolCall(tool_name="summarize_document", arguments={}))

        if self._query_classifier is not None and self._query_classifier.is_out_of_scope(request):
            return PlannerRefusal(reason=_OUT_OF_SCOPE_REASON)

        return PlannerRefusal(reason=_UNSUPPORTED_REASON)
