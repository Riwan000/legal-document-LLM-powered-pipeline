"""
In-memory store for workflow state keyed by workflow_id.

Used by GET /api/workflow/{workflow_id}/state. Updated at workflow start and
after stage transitions. Fast, non-blocking reads.
"""
from typing import Any, Dict, Optional

from backend.workflow_stages import WorkflowState


_store: Dict[str, Dict[str, Any]] = {}


def put(workflow_id: str, workflow_state: WorkflowState) -> None:
    """Write workflow state for workflow_id (serialized for JSON)."""
    _store[workflow_id] = workflow_state.model_dump(mode="json")


def get(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Read workflow state; None if not found."""
    return _store.get(workflow_id)


def remove(workflow_id: str) -> None:
    """Remove state (optional cleanup)."""
    _store.pop(workflow_id, None)
