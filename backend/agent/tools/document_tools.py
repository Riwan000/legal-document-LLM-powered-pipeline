from __future__ import annotations

"""
Supporting tools: scoped document listing and classification lookup.
These exist so the planner can orient itself before picking a core tool.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.models.agent import ToolResult, ToolSpec

if TYPE_CHECKING:
    from backend.agent.tool_registry import ToolRegistry
    from backend.agent.tools import AgentToolServices

_LISTING_FIELDS = (
    "document_id",
    "display_name",
    "document_type",
    "total_pages",
    "version",
    "is_latest",
)


def register_document_tools(registry: "ToolRegistry", services: "AgentToolServices") -> None:
    def list_documents(
        document_ids: List[str],
        jurisdiction: Optional[str] = None,
    ) -> ToolResult:
        allowed = set(document_ids)
        records = services.document_registry.list_documents(include_versions=False)
        documents: List[Dict[str, Any]] = [
            {field: record.get(field) for field in _LISTING_FIELDS}
            for record in records
            if record.get("document_id") in allowed
        ]
        names = ", ".join(str(doc.get("display_name")) for doc in documents) or "(none)"
        return ToolResult(
            tool_name="list_documents",
            status="ok",
            summary=f"{len(documents)} document(s) in scope: {names}.",
            payload={"documents": documents},
        )

    registry.register(
        ToolSpec(
            name="list_documents",
            description=(
                "List the documents available in this run's scope with their names, "
                "types, and page counts. Use to orient before picking another tool."
            ),
            params=[],
        ),
        list_documents,
    )

    def get_classification(
        document_ids: List[str],
        jurisdiction: Optional[str] = None,
    ) -> ToolResult:
        classifications: Dict[str, Any] = {}
        parts: List[str] = []
        for doc_id in document_ids:
            cls_info = services.document_registry.get_classification(doc_id)
            classifications[doc_id] = cls_info
            if cls_info:
                parts.append(
                    f"{doc_id}: {cls_info.get('detected_contract_type', 'unknown')}"
                    f" ({cls_info.get('detected_jurisdiction') or 'no jurisdiction'})"
                )
            else:
                parts.append(f"{doc_id}: unclassified")
        return ToolResult(
            tool_name="get_classification",
            status="ok",
            summary="Classification — " + "; ".join(parts),
            payload={"classifications": classifications},
        )

    registry.register(
        ToolSpec(
            name="get_classification",
            description=(
                "Return the stored document/contract type and detected jurisdiction "
                "for each document in scope. Use to decide contract_type before review."
            ),
            params=[],
        ),
        get_classification,
    )
