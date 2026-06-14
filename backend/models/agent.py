from __future__ import annotations

"""
Agent-layer models: tool specs, tool calls/results, planner decisions,
and agent run request/response schemas.

These models are pure data — no logic. The agent layer (backend/agent/)
consumes them. See plans/agentic-orchestrator.md for the design.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ToolParam(BaseModel):
    """Single parameter definition in a ToolSpec."""

    name: str = Field(..., description="Parameter name (snake_case).")
    type: Literal["string", "integer", "number", "boolean", "array"] = Field(
        ..., description="JSON-schema primitive type of the parameter."
    )
    description: str = Field(
        ..., description="What the parameter means — shown to the planner."
    )
    required: bool = Field(default=True, description="Whether the parameter is required.")
    enum: Optional[List[str]] = Field(
        default=None, description="Allowed values, if the parameter is an enum."
    )


class ToolSpec(BaseModel):
    """
    Declaration of a callable tool.

    The description is what the planner sees when choosing tools, so it must
    be written like a prompt: state what the tool does and when to use it.
    """

    name: str = Field(..., description="Unique tool name (snake_case).")
    description: str = Field(
        ..., description="Planner-facing description of what the tool does and when to use it."
    )
    params: List[ToolParam] = Field(default_factory=list, description="Parameter definitions.")
    requires_document_scope: bool = Field(
        default=True,
        description=(
            "If True, the registry injects document_ids from the run scope, "
            "overriding any planner-supplied values (document isolation boundary)."
        ),
    )
    is_terminal: bool = Field(
        default=False,
        description="If True, the tool's output is a complete answer and the run can finalize on it.",
    )
    requires_human_approval: bool = Field(
        default=False,
        description="Reserved for future write-tools; all v1 tools are read-only and never set this.",
    )


class ToolCall(BaseModel):
    """A planner's request to execute one tool."""

    tool_name: str = Field(..., description="Name of the tool to execute.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool, keyed by param name."
    )


class ToolResult(BaseModel):
    """
    Result of one tool execution.

    Two channels by design:
    - summary: short text fed back to the planner (kept small for local models)
    - payload: full structured result held by the executor, never sent to the LLM
    """

    tool_name: str = Field(..., description="Name of the executed tool.")
    status: Literal["ok", "error", "refused"] = Field(..., description="Execution outcome.")
    summary: str = Field(
        ..., description="Short planner-facing summary of the result (<= ~400 chars)."
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full structured result; used to assemble the final response.",
    )
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chunk/citation references accumulated for guardrail verification.",
    )
    duration_ms: int = Field(default=0, description="Wall-clock execution time in milliseconds.")


class AgentScope(BaseModel):
    """Document scope a run is confined to. Injected into every scoped tool call."""

    document_ids: List[str] = Field(
        ..., description="Documents this run may touch. Tools cannot reach outside this set."
    )
    jurisdiction: Optional[str] = Field(
        default=None, description="Jurisdiction code/name (e.g. 'KSA', 'UAE')."
    )


class AgentStep(BaseModel):
    """One entry in the run's audit trace. Contains no document text."""

    step_index: int = Field(..., description="0-based position of this step in the run.")
    tool_call: ToolCall = Field(..., description="The tool call that was executed.")
    result_status: str = Field(..., description="Status of the tool result (ok/error/refused).")
    result_summary: str = Field(..., description="The planner-facing summary of the result.")
    duration_ms: int = Field(default=0, description="Execution time in milliseconds.")


class PlannerToolCall(BaseModel):
    """Planner decision: execute a tool next."""

    kind: Literal["tool_call"] = "tool_call"
    call: ToolCall = Field(..., description="The tool call to execute.")


class PlannerFinalAnswer(BaseModel):
    """
    Planner decision: finalize the run from prior tool results.

    The executor assembles the actual answer from the referenced step payloads;
    the planner's own prose is never returned to the user.
    """

    kind: Literal["final_answer"] = "final_answer"
    use_results: List[int] = Field(
        ..., description="Indices of prior steps whose payloads form the final answer."
    )


class PlannerRefusal(BaseModel):
    """Planner decision: refuse the request."""

    kind: Literal["refusal"] = "refusal"
    reason: str = Field(..., description="User-facing reason for refusing the request.")


PlannerDecision = Union[PlannerToolCall, PlannerFinalAnswer, PlannerRefusal]


class AgentRunRequest(BaseModel):
    """Request body for POST /api/agent/run."""

    request: str = Field(..., min_length=1, description="Natural-language task description.")
    document_ids: List[str] = Field(
        ..., min_length=1, description="Explicit document scope for the run (mandatory)."
    )
    jurisdiction: Optional[str] = Field(default=None, description="Optional jurisdiction hint.")
    max_steps: int = Field(
        default=6, ge=1, le=10, description="Maximum planner/tool iterations (server cap 10)."
    )


class AgentRunResponse(BaseModel):
    """Response body for POST /api/agent/run and GET /api/agent/run/{run_id}."""

    run_id: str = Field(..., description="Unique identifier of this agent run.")
    status: Literal["completed", "refused", "failed", "max_steps_reached"] = Field(
        ..., description="Terminal status of the run."
    )
    answer: Optional[str] = Field(
        default=None, description="Assembled answer text (from tool payloads only)."
    )
    structured_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Terminal tool payloads keyed by tool name (e.g. full contract review report).",
    )
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="All citations supporting the answer."
    )
    steps: List[AgentStep] = Field(
        default_factory=list, description="Audit trace of executed steps."
    )
    disclaimer: str = Field(..., description="Workflow disclaimer; always present.")
    refusal_reason: Optional[str] = Field(
        default=None, description="Populated when status == 'refused'."
    )
