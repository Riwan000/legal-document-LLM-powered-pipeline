from __future__ import annotations

"""
LLM planner (Phase 2): routes a request to tools via Ollama native
tool calling (/api/chat with tools=). The planner only ever selects
tools or finalizes from prior tool results — its prose is never shown
to the user (the executor assembles answers from tool payloads).

Robustness rules for small local models:
- temperature=0 + fixed seed (determinism)
- malformed output → ONE retry with the validation error appended;
  a second failure raises PlannerError and the run fails cleanly
  (never guess a tool call).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.models.agent import (
    AgentScope,
    AgentStep,
    PlannerDecision,
    PlannerFinalAnswer,
    PlannerRefusal,
    PlannerToolCall,
    ToolCall,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the routing planner of a legal document analysis system.

Hard rules:
- You NEVER answer from your own knowledge. You only pick tools or finalize from prior tool results.
- Call at most ONE tool per turn, using the provided tool definitions.
- Document scope is fixed by the system; you cannot add documents.
- When the prior tool results already answer the request, reply with JSON only:
  {"action": "final_answer", "use_results": [<indices of the steps to use>]}
- To refuse, reply with JSON only:
  {"action": "refuse", "reason": "<short user-facing reason>"}
- Refuse when the request asks for legal advice or opinion, when no tool fits,
  or when the request is unrelated to the documents in scope.
Reply with either one tool call or one JSON object. Nothing else."""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class PlannerError(Exception):
    """Planner produced unusable output after retry; the run must fail cleanly."""


class _ParseError(Exception):
    """Internal: one malformed planner response (retried once)."""


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an attribute-style response object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class LLMPlanner:
    """Ollama tool-calling planner. Implements the same protocol as RuleBasedPlanner."""

    def __init__(
        self,
        registry: Any,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        seed: Optional[int] = None,
    ):
        if client is None:
            import ollama

            client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self._client = client
        self._registry = registry
        self._model = model or settings.AGENT_MODEL
        self._seed = settings.AGENT_SEED if seed is None else seed

    def next_action(
        self, request: str, scope: AgentScope, history: List[AgentStep]
    ) -> PlannerDecision:
        messages = self._build_messages(request, scope, history)
        tools = self._registry.to_ollama_tools()

        last_error: Optional[str] = None
        for _attempt in range(2):
            attempt_messages = list(messages)
            if last_error:
                attempt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid: {last_error} "
                            "Respond again following the required format exactly."
                        ),
                    }
                )
            response = self._client.chat(
                model=self._model,
                messages=attempt_messages,
                tools=tools,
                options={"temperature": 0, "seed": self._seed},
            )
            try:
                return self._parse(response, history)
            except _ParseError as exc:
                last_error = str(exc)
                logger.warning(
                    "planner produced malformed output", extra={"error": last_error}
                )

        raise PlannerError(f"Planner output invalid after retry: {last_error}")

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    def _build_messages(
        self, request: str, scope: AgentScope, history: List[AgentStep]
    ) -> List[Dict[str, Any]]:
        scope_line = (
            f"Documents in scope: {len(scope.document_ids)}. "
            f"Jurisdiction: {scope.jurisdiction or 'not specified'}."
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"{scope_line}\n\nRequest: {request}"},
        ]
        for step in history:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": step.tool_call.tool_name,
                                "arguments": step.tool_call.arguments,
                            }
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "content": f"[step {step.step_index}, {step.result_status}] {step.result_summary}",
                }
            )
        return messages

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse(self, response: Any, history: List[AgentStep]) -> PlannerDecision:
        message = _field(response, "message", {}) or {}
        tool_calls = _field(message, "tool_calls") or []

        if tool_calls:
            first = tool_calls[0]
            function = _field(first, "function", {}) or {}
            name = _field(function, "name")
            arguments = _field(function, "arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    raise _ParseError(f"Tool arguments for '{name}' are not valid JSON.")
            if not name or not isinstance(arguments, dict):
                raise _ParseError("Tool call is missing a name or has non-object arguments.")
            if self._registry.get_spec(name) is None:
                known = ", ".join(spec.name for spec in self._registry.specs())
                raise _ParseError(f"Unknown tool '{name}'. Known tools: {known}.")
            return PlannerToolCall(call=ToolCall(tool_name=name, arguments=arguments))

        content = (_field(message, "content") or "").strip()
        match = _JSON_BLOCK.search(content)
        if not match:
            raise _ParseError("Response contained neither a tool call nor a JSON decision.")
        try:
            decision = json.loads(match.group(0))
        except json.JSONDecodeError:
            raise _ParseError("Decision JSON could not be parsed.")

        action = decision.get("action")
        if action == "refuse":
            reason = str(decision.get("reason") or "Request refused by the planner.")
            return PlannerRefusal(reason=reason)
        if action == "final_answer":
            use_results = decision.get("use_results")
            if not isinstance(use_results, list) or not use_results:
                raise _ParseError("final_answer requires a non-empty 'use_results' list.")
            try:
                indices = [int(i) for i in use_results]
            except (TypeError, ValueError):
                raise _ParseError("'use_results' must contain integer step indices.")
            invalid = [i for i in indices if i < 0 or i >= len(history)]
            if invalid:
                raise _ParseError(
                    f"'use_results' references unknown step indices: {invalid}. "
                    f"Valid range: 0..{len(history) - 1}."
                )
            return PlannerFinalAnswer(use_results=indices)
        raise _ParseError(f"Unknown action '{action}'. Use 'final_answer' or 'refuse'.")
