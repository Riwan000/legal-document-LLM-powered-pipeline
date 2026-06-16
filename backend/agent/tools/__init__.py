from __future__ import annotations

"""
Tool registration for the agent layer.

AgentToolServices is the dependency bundle built in main.py's DI block;
register_all_tools wires every Phase 1 tool into a ToolRegistry.
"""

from dataclasses import dataclass
from typing import Any

from backend.agent.tool_registry import ToolRegistry
from backend.agent.tools.analysis_tools import register_analysis_tools
from backend.agent.tools.contract_tools import register_contract_tools
from backend.agent.tools.document_tools import register_document_tools
from backend.agent.tools.reference_tools import register_reference_tools
from backend.agent.tools.retrieval_tools import register_retrieval_tools


@dataclass(frozen=True)
class AgentToolServices:
    """Injected service dependencies for the agent tools."""

    workflow_orchestrator: Any
    comparison_service: Any
    summarization_service: Any
    document_registry: Any
    rag_service: Any = None
    extracted_clause_store: Any = None


def register_all_tools(registry: ToolRegistry, services: AgentToolServices) -> None:
    """Register the full v1 tool catalog. Order defines presentation order to the planner."""
    register_contract_tools(registry, services)
    register_analysis_tools(registry, services)
    register_document_tools(registry, services)
    if services.rag_service is not None:
        register_retrieval_tools(registry, services)
    register_reference_tools(registry, services)
