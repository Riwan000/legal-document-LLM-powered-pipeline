from __future__ import annotations

"""
Tool registry for the agent layer.

The registry is the security boundary between the planner (LLM or rules)
and the underlying services:
- validates tool names and arguments against declared ToolSpecs
- injects the run's document scope, overriding planner-supplied values
- sanitizes handler exceptions (no stack traces or document text leak
  back into planner context)
- timeboxes every handler call

Handlers receive validated arguments as keyword args plus, for scoped tools,
`document_ids` and `jurisdiction` from the AgentScope.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, List

from backend.models.agent import AgentScope, ToolCall, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

# Type coercion map for declared param types.
_PARAM_TYPE_CHECKS: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
}

ToolHandler = Callable[..., ToolResult]


class ToolRegistry:
    """Holds ToolSpecs and their handlers; executes validated, scoped tool calls."""

    def __init__(self, tool_timeout_s: int = 120):
        self._tools: Dict[str, tuple[ToolSpec, ToolHandler]] = {}
        self._timeout_s = tool_timeout_s

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a tool. Raises ValueError on duplicate names."""
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = (spec, handler)

    def specs(self) -> List[ToolSpec]:
        """All registered specs, in registration order."""
        return [spec for spec, _ in self._tools.values()]

    def get_spec(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def to_ollama_tools(self) -> List[Dict[str, Any]]:
        """Export specs as Ollama /api/chat `tools` JSON schema."""
        tools: List[Dict[str, Any]] = []
        for spec, _ in self._tools.values():
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for param in spec.params:
                prop: Dict[str, Any] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.enum:
                    prop["enum"] = param.enum
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return tools

    def execute(self, call: ToolCall, scope: AgentScope) -> ToolResult:
        """
        Validate and execute one tool call within the given scope.

        Never raises: every failure mode returns a ToolResult with
        status="error" and a planner-readable corrective summary.
        """
        entry = self._tools.get(call.tool_name)
        if entry is None:
            return ToolResult(
                tool_name=call.tool_name,
                status="error",
                summary=f"Unknown tool: {call.tool_name}. Use one of the declared tools.",
            )
        spec, handler = entry

        validation_error = self._validate_arguments(spec, call.arguments)
        if validation_error:
            return ToolResult(
                tool_name=call.tool_name, status="error", summary=validation_error
            )

        kwargs = dict(call.arguments)
        if spec.requires_document_scope:
            # Scope injection: the planner can never reach outside the run's
            # document set, even if it supplies its own document_ids.
            kwargs["document_ids"] = list(scope.document_ids)
            kwargs["jurisdiction"] = scope.jurisdiction

        start = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(handler, **kwargs)
                # Worker thread is abandoned on timeout; acceptable for v1
                # because all registered tools are read-only.
                result = future.result(timeout=self._timeout_s)
        except FutureTimeoutError:
            logger.warning(
                "agent tool timed out", extra={"tool": call.tool_name, "timeout_s": self._timeout_s}
            )
            return ToolResult(
                tool_name=call.tool_name,
                status="error",
                summary=f"Tool {call.tool_name} timed out after {self._timeout_s}s.",
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as exc:
            # Sanitized: log the class for diagnosis, never the message,
            # which may contain document content.
            logger.error(
                "agent tool failed", extra={"tool": call.tool_name, "error_type": type(exc).__name__}
            )
            return ToolResult(
                tool_name=call.tool_name,
                status="error",
                summary=f"Tool {call.tool_name} failed with an internal error.",
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        result.duration_ms = int((time.monotonic() - start) * 1000)
        return result

    def _validate_arguments(self, spec: ToolSpec, arguments: Dict[str, Any]) -> str | None:
        """Return a corrective error message, or None if arguments are valid."""
        declared = {param.name: param for param in spec.params}

        # document_ids/jurisdiction are injected, not planner-supplied; tolerate
        # the planner naming them but never trust the values (overwritten later).
        injected = {"document_ids", "jurisdiction"} if spec.requires_document_scope else set()

        unknown = [key for key in arguments if key not in declared and key not in injected]
        if unknown:
            return (
                f"Unknown argument(s) for {spec.name}: {', '.join(sorted(unknown))}. "
                f"Allowed: {', '.join(sorted(declared)) or '(none)'}."
            )

        missing = [
            name for name, param in declared.items() if param.required and name not in arguments
        ]
        if missing:
            return f"Missing required argument(s) for {spec.name}: {', '.join(sorted(missing))}."

        for name, param in declared.items():
            if name not in arguments:
                continue
            expected = _PARAM_TYPE_CHECKS[param.type]
            value = arguments[name]
            if param.type == "number" and isinstance(value, int) and not isinstance(value, bool):
                continue  # ints are acceptable numbers
            if not isinstance(value, expected) or (
                expected is int and isinstance(value, bool)
            ):
                return (
                    f"Argument '{name}' of {spec.name} must be of type {param.type}."
                )
            if param.enum and value not in param.enum:
                return (
                    f"Argument '{name}' of {spec.name} must be one of: {', '.join(param.enum)}."
                )

        return None
