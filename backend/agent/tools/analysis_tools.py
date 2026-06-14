from __future__ import annotations

"""
Pillar-2 and pillar-3 tools: due diligence memo and case-file summarization.
"""

from typing import TYPE_CHECKING, List, Optional

from backend.models.agent import ToolParam, ToolResult, ToolSpec

if TYPE_CHECKING:
    from backend.agent.tool_registry import ToolRegistry
    from backend.agent.tools import AgentToolServices


def register_analysis_tools(registry: "ToolRegistry", services: "AgentToolServices") -> None:
    def due_diligence_memo(
        document_ids: List[str],
        jurisdiction: Optional[str] = None,
    ) -> ToolResult:
        ctx = services.workflow_orchestrator.run_due_diligence_memo(document_id=document_ids[0])
        ctx_dict = ctx.model_dump()
        if ctx_dict.get("status") == "failed":
            error = ctx_dict.get("error") or {}
            return ToolResult(
                tool_name="due_diligence_memo",
                status="error",
                summary=(
                    f"due_diligence_memo failed at step '{error.get('step', 'unknown')}' "
                    f"with code {error.get('code', 'WORKFLOW_FAILED')}."
                ),
                payload=ctx_dict,
            )
        return ToolResult(
            tool_name="due_diligence_memo",
            status="ok",
            summary="Due diligence memo generated for the document in scope.",
            payload=ctx_dict,
        )

    registry.register(
        ToolSpec(
            name="due_diligence_memo",
            description=(
                "Generate a structured due diligence memo for the document in scope. "
                "Use when the user asks for due diligence, a DD memo, or a deal-readiness review."
            ),
            params=[],
            is_terminal=True,
        ),
        due_diligence_memo,
    )

    def summarize_document(
        document_ids: List[str],
        jurisdiction: Optional[str] = None,
        top_k: int = 10,
    ) -> ToolResult:
        result = services.summarization_service.summarize_case_file_prd(
            document_id=document_ids[0], top_k=top_k
        )
        if isinstance(result, dict) and "error" in result:
            # Surface only the machine-readable code — error messages from the
            # summarizer may quote document text.
            code = (result.get("error") or {}).get("code", "SUMMARIZATION_FAILED")
            return ToolResult(
                tool_name="summarize_document",
                status="error",
                summary=f"summarize_document failed with code {code}.",
            )
        n_sections = len(result.get("sections") or {}) if isinstance(result, dict) else 0
        return ToolResult(
            tool_name="summarize_document",
            status="ok",
            summary=f"Case summary generated with {n_sections} section(s).",
            payload={"summary": result},
        )

    registry.register(
        ToolSpec(
            name="summarize_document",
            description=(
                "Generate a cited, multi-section summary of the case file in scope "
                "(facts, issues, chronology). Use when the user asks to summarize a "
                "document, build a chronology, or get a case overview."
            ),
            params=[
                ToolParam(
                    name="top_k",
                    type="integer",
                    description="Number of chunks to retrieve per section.",
                    required=False,
                ),
            ],
            is_terminal=True,
        ),
        summarize_document,
    )
