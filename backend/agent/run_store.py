from __future__ import annotations

"""
In-process store for completed agent runs, keyed by run_id.

Mirrors backend/workflow_store.py but holds AgentRunResponse dumps.
Bounded: oldest runs are evicted past _MAX_RUNS to avoid unbounded growth.
"""

from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

_MAX_RUNS = 100

_runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_lock = Lock()


def put_run(run_id: str, response: Dict[str, Any]) -> None:
    """Store a completed run, evicting the oldest entry past capacity."""
    with _lock:
        _runs[run_id] = response
        _runs.move_to_end(run_id)
        while len(_runs) > _MAX_RUNS:
            _runs.popitem(last=False)


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Return a stored run, or None."""
    with _lock:
        return _runs.get(run_id)
