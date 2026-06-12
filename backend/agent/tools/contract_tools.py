from __future__ import annotations

"""
Core pillar-1 tools: contract review and document comparison.

Handlers translate service results into the ToolResult two-channel shape:
short planner-facing summary + full payload for the executor.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.config import settings
from backend.models.agent import ToolParam, ToolResult, ToolSpec

if TYPE_CHECKING:
    from backend.agent.tool_registry import ToolRegistry
    from backend.agent.tools import AgentToolServices

_DOC_EXTENSIONS = (".pdf", ".docx", ".doc")


def _resolve_document_path(document_id: str) -> Optional[str]:
    """Find the source file for a document id, mirroring /api/compare resolution."""
    for base in (settings.DOCUMENTS_PATH, settings.TEMPLATES_PATH):
        for ext in _DOC_EXTENSIONS:
            candidate = Path(base) / f"{document_id}{ext}"
            if candidate.exists():
                return str(candidate)
    return None


def _workflow_error_result(tool_name: str, ctx_dict: Dict[str, Any]) -> ToolResult:
    error = ctx_dict.get("error") or {}
    code = error.get("code", "WORKFLOW_FAILED")
    step = error.get("step", "unknown")
    return ToolResult(
        tool_name=tool_name,
        status="error",
        summary=f"{tool_name} failed at step '{step}' with code {code}.",
        payload=ctx_dict,
    )


def register_contract_tools(registry: "ToolRegistry", services: "AgentToolServices") -> None:
    def review_contract(
        document_ids: List[str],
        jurisdiction: Optional[str] = None,
        contract_type: Optional[str] = None,
        review_depth: str = "standard",
    ) -> ToolResult:
        contract_id = document_ids[0]

        # Auto-resolve contract_type/jurisdiction from the classification
        # registry when not explicitly provided (F-01 pattern).
        cls_info = services.document_registry.get_classification(contract_id) or {}
        if not contract_type:
            detected = cls_info.get("detected_contract_type")
            contract_type = detected if detected and detected != "other" else "other"
        if not jurisdiction:
            jurisdiction = cls_info.get("detected_jurisdiction")

        ctx = services.workflow_orchestrator.run_contract_review(
            contract_id=contract_id,
            contract_type=contract_type,
            jurisdiction=jurisdiction,
            review_depth=review_depth,
        )
        ctx_dict = ctx.model_dump()
        if ctx_dict.get("status") == "failed":
            return _workflow_error_result("review_contract", ctx_dict)

        response = (
            (ctx_dict.get("intermediate_results") or {}).get("contract_review", {}).get("response")
            or {}
        )
        n_risks = len(response.get("risks") or [])
        risk_label = response.get("risk_label", "n/a")
        risk_score = response.get("risk_score", "n/a")
        return ToolResult(
            tool_name="review_contract",
            status="ok",
            summary=(
                f"Review complete ({contract_type}): {n_risks} risk(s), "
                f"overall {risk_label} (score {risk_score})."
            ),
            payload=ctx_dict,
        )

    registry.register(
        ToolSpec(
            name="review_contract",
            description=(
                "Run a full contract review on the document in scope: detects risks, "
                "missing clauses, contradictions, and statutory notes. Use when the user "
                "asks to review a contract, find risks, or check completeness."
            ),
            params=[
                ToolParam(
                    name="contract_type",
                    type="string",
                    description="Contract type; omit to auto-detect from classification.",
                    required=False,
                ),
                ToolParam(
                    name="review_depth",
                    type="string",
                    description="Review depth.",
                    required=False,
                    enum=["quick", "standard"],
                ),
            ],
            is_terminal=True,
        ),
        review_contract,
    )

    def compare_documents(
        document_ids: List[str],
        jurisdiction: Optional[str] = None,
    ) -> ToolResult:
        if len(document_ids) < 2:
            return ToolResult(
                tool_name="compare_documents",
                status="error",
                summary="Comparison needs two documents in scope: a contract and a template.",
            )
        contract_id, template_id = document_ids[0], document_ids[1]
        contract_path = _resolve_document_path(contract_id)
        template_path = _resolve_document_path(template_id)
        if not contract_path or not template_path:
            missing = contract_id if not contract_path else template_id
            return ToolResult(
                tool_name="compare_documents",
                status="error",
                summary=f"Source file not found for document {missing}.",
            )

        comparison = services.comparison_service.compare_contracts(
            contract_path=contract_path,
            template_path=template_path,
            contract_id=contract_id,
            template_id=template_id,
        )
        report = services.comparison_service.generate_comparison_report(
            comparison, format="markdown"
        )
        n_diff = len(comparison.get("differences") or []) if isinstance(comparison, dict) else 0
        return ToolResult(
            tool_name="compare_documents",
            status="ok",
            summary=f"Comparison complete: {n_diff} difference(s) between contract and template.",
            payload={"comparison": comparison, "report": report},
        )

    registry.register(
        ToolSpec(
            name="compare_documents",
            description=(
                "Compare the first document in scope (contract) against the second "
                "(template/standard) and report deviations. Use when the user asks to "
                "compare documents, check against a template, or find deviations."
            ),
            params=[],
            is_terminal=True,
        ),
        compare_documents,
    )
