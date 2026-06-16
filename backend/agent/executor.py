from __future__ import annotations

"""
Agent executor: the loop between a planner and the tool registry.

Semantics:
- scope is validated against the document registry before any planner call
- a successful terminal tool finalizes the run immediately
- fail_fast=True (rule planner): any tool error fails the run, preserving
  the partial trace (mirrors WorkflowContext partial-results-on-failure)
- fail_fast=False (LLM planner): a tool error is fed back to the planner as
  an error summary so it can correct arguments, pick another tool, or refuse
- identical repeated tool calls are short-circuited, never re-executed
- max_steps is capped server-side regardless of the request
- planner exceptions (PlannerError, transport failures) fail the run cleanly

The executor assembles all user-facing output from tool payloads; planner
prose is never returned to the user. Multi-tool assembled answers are
re-verified by EvidenceGuardrailService before release (verifier pass);
single-tool answers inherit the guardrails of the wrapped service.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Protocol

from backend.agent.run_store import put_run
from backend.agent.tool_registry import ToolRegistry
from backend.agent.trace import AgentTraceLogger
from backend.models.agent import (
    AgentRunRequest,
    AgentRunResponse,
    AgentScope,
    AgentStep,
    PlannerDecision,
    ToolResult,
)
from backend.services.guardrails import WORKFLOW_DISCLAIMER

logger = logging.getLogger(__name__)

_MAX_STEPS_CAP = 10
_VERIFIER_DOWNGRADE_NOTICE = (
    "The combined narrative could not be verified against the cited evidence and "
    "was removed. The structured per-tool results below remain available."
)


class Planner(Protocol):
    def next_action(
        self, request: str, scope: AgentScope, history: List[AgentStep]
    ) -> PlannerDecision: ...


class AgentExecutor:
    """Runs one agent request to completion. Synchronous."""

    def __init__(
        self,
        registry: ToolRegistry,
        planner: Planner,
        document_registry: Any,
        trace_logger: Optional[AgentTraceLogger] = None,
        guardrail_service: Optional[Any] = None,
        fail_fast: bool = True,
    ):
        self._registry = registry
        self._planner = planner
        self._document_registry = document_registry
        self._trace = trace_logger or AgentTraceLogger()
        self._guardrail = guardrail_service
        self._fail_fast = fail_fast

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        run_id = uuid.uuid4().hex
        scope = AgentScope(
            document_ids=request.document_ids, jurisdiction=request.jurisdiction
        )
        self._trace.run_started(run_id, request.document_ids)

        unknown = [
            doc_id
            for doc_id in request.document_ids
            if self._document_registry.get_document(doc_id) is None
        ]
        if unknown:
            return self._finalize(
                run_id,
                status="refused",
                steps=[],
                refusal_reason=f"Unknown document(s) in scope: {', '.join(unknown)}.",
            )

        max_steps = min(request.max_steps, _MAX_STEPS_CAP)
        steps: List[AgentStep] = []
        results: List[ToolResult] = []
        seen_calls: Dict[str, ToolResult] = {}

        while len(steps) < max_steps:
            try:
                decision = self._planner.next_action(
                    request=request.request, scope=scope, history=steps
                )
            except Exception as exc:
                # PlannerError (invalid output after retry) or transport failure.
                # Sanitized: log the class only, never the message.
                logger.error(
                    "agent planner failed", extra={"error_type": type(exc).__name__}
                )
                return self._finalize(run_id, status="failed", steps=steps)

            if decision.kind == "refusal":
                return self._finalize(
                    run_id, status="refused", steps=steps, refusal_reason=decision.reason
                )

            if decision.kind == "final_answer":
                if any(i < 0 or i >= len(results) for i in decision.use_results):
                    return self._finalize(run_id, status="failed", steps=steps)
                return self._finalize(
                    run_id,
                    status="completed",
                    steps=steps,
                    used_results=[results[i] for i in decision.use_results],
                    request_text=request.request,
                )

            call = decision.call
            call_key = f"{call.tool_name}:{sorted(call.arguments.items())!r}"
            if call_key in seen_calls:
                result = seen_calls[call_key]
                steps.append(
                    AgentStep(
                        step_index=len(steps),
                        tool_call=call,
                        result_status=result.status,
                        result_summary=f"(duplicate call — reused) {result.summary}",
                        duration_ms=0,
                    )
                )
                results.append(result)
                continue

            result = self._registry.execute(call, scope)
            seen_calls[call_key] = result
            step = AgentStep(
                step_index=len(steps),
                tool_call=call,
                result_status=result.status,
                result_summary=result.summary,
                duration_ms=result.duration_ms,
            )
            steps.append(step)
            results.append(result)
            self._trace.step(run_id, step)

            if result.status != "ok":
                if self._fail_fast:
                    return self._finalize(run_id, status="failed", steps=steps)
                # Error-recovery turn: the corrective summary is now in the
                # history; let the planner react (fix args, reroute, refuse).
                continue

            spec = self._registry.get_spec(call.tool_name)
            if spec is not None and spec.is_terminal:
                return self._finalize(
                    run_id,
                    status="completed",
                    steps=steps,
                    used_results=[result],
                    request_text=request.request,
                )

        return self._finalize(run_id, status="max_steps_reached", steps=steps)

    def _finalize(
        self,
        run_id: str,
        status: str,
        steps: List[AgentStep],
        used_results: Optional[List[ToolResult]] = None,
        refusal_reason: Optional[str] = None,
        request_text: Optional[str] = None,
    ) -> AgentRunResponse:
        answer: Optional[str] = None
        structured_results: Dict[str, Any] = {}
        citations: List[Dict[str, Any]] = []
        guardrail_status: Optional[str] = None
        verifier_notice: Optional[str] = None

        if used_results:
            # Tools that produce user-facing prose put it in payload["answer_text"];
            # for report-style tools the planner summary doubles as the answer line.
            answer = " ".join(
                result.payload.get("answer_text") or result.summary
                for result in used_results
            )
            for result in used_results:
                structured_results[result.tool_name] = result.payload
                citations.extend(result.citations)

            if len(used_results) == 1:
                guardrail_status = "inherited"
            else:
                answer, guardrail_status, verifier_notice = self._verify_assembled(
                    answer, used_results, request_text or ""
                )

        response = AgentRunResponse(
            run_id=run_id,
            status=status,  # type: ignore[arg-type]
            answer=answer,
            structured_results=structured_results,
            citations=citations,
            steps=steps,
            disclaimer=WORKFLOW_DISCLAIMER,
            refusal_reason=refusal_reason,
            guardrail_status=guardrail_status,
            verifier_notice=verifier_notice,
        )
        put_run(run_id, response.model_dump())
        self._trace.run_finished(run_id, status, len(steps))
        return response

    def _verify_assembled(
        self, answer: str, used_results: List[ToolResult], request_text: str
    ) -> tuple[Optional[str], str, Optional[str]]:
        """
        Verifier pass over a multi-tool assembled answer.

        Returns (answer, guardrail_status, verifier_notice). Never returns
        unverified synthesis: with no verifier or no text-bearing citations,
        the narrative is downgraded and only structured results remain.
        """
        if self._guardrail is None:
            return None, "downgraded", _VERIFIER_DOWNGRADE_NOTICE

        chunks = [
            citation
            for result in used_results
            for citation in result.citations
            if citation.get("text")
        ]
        if not chunks:
            return None, "downgraded", _VERIFIER_DOWNGRADE_NOTICE

        try:
            verdict = self._guardrail.check(answer=answer, chunks=chunks, query=request_text)
        except Exception as exc:
            logger.error(
                "agent verifier failed", extra={"error_type": type(exc).__name__}
            )
            return None, "downgraded", _VERIFIER_DOWNGRADE_NOTICE

        if verdict.decision == "fail":
            return None, "downgraded", _VERIFIER_DOWNGRADE_NOTICE
        if verdict.decision == "weak":
            return answer, "weak", None
        return answer, "verified", None
