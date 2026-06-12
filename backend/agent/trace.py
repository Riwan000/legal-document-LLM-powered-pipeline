from __future__ import annotations

"""
Privacy-safe audit trace for agent runs — logs/agent.log (structured JSON lines).

Rules (CLAUDE.md §8): hashed document identifiers, argument KEYS only
(values may contain user content), no chunk text, no answers.
The full run response lives in run_store, never in logs.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from backend.models.agent import AgentStep

_agent_log_path = Path("logs/agent.log")
_agent_log_path.parent.mkdir(parents=True, exist_ok=True)
_audit_logger = logging.getLogger("agent.audit")
if not _audit_logger.handlers:
    _fh = logging.FileHandler(str(_agent_log_path), encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(message)s"))
    _audit_logger.addHandler(_fh)
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False


def _hash_doc_id(document_id: str) -> str:
    return "sha256:" + hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:16]


class AgentTraceLogger:
    """Writes one JSON line per run event. Logger injectable for tests."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or _audit_logger

    def _emit(self, event: dict) -> None:
        event["ts"] = int(time.time() * 1000)
        self._logger.info(json.dumps(event, ensure_ascii=False))

    def run_started(self, run_id: str, document_ids: List[str]) -> None:
        self._emit(
            {
                "event": "agent_run_started",
                "run_id": run_id,
                "doc_hashes": [_hash_doc_id(d) for d in document_ids],
                "n_documents": len(document_ids),
            }
        )

    def step(self, run_id: str, step: AgentStep) -> None:
        self._emit(
            {
                "event": "agent_step",
                "run_id": run_id,
                "step": step.step_index,
                "tool": step.tool_call.tool_name,
                "args_keys": sorted(step.tool_call.arguments.keys()),
                "status": step.result_status,
                "duration_ms": step.duration_ms,
            }
        )

    def run_finished(self, run_id: str, status: str, n_steps: int) -> None:
        self._emit(
            {
                "event": "agent_run_finished",
                "run_id": run_id,
                "status": status,
                "n_steps": n_steps,
            }
        )
