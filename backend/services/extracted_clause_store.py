"""
Extracted Clause Store.

Persistent storage for deterministic extracted clauses (verbatim evidence only).
This store is the source of truth for Document Explorer.

Payload includes extraction_version (EVIDENCE_EXPLORER_SPEC §5) for explainability
and regression debugging when extraction logic changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.config import settings

# Pinned extraction schema version for stored clauses and API responses
EXTRACTION_VERSION = "v1.0"


class ExtractedClauseStore:
    """Store and load extracted clauses for a document."""

    def __init__(self, store_path: Optional[Path] = None):
        """
        Args:
            store_path: Base directory for extracted clauses (defaults to config).
        """
        self.store_path = store_path or settings.EXTRACTED_CLAUSES_PATH
        self.store_path.mkdir(parents=True, exist_ok=True)

    def save_document_clauses(self, document_id: str, clauses: List[Dict[str, Any]]) -> None:
        """
        Save extracted clauses for a document.

        Args:
            document_id: Document identifier
            clauses: List of extraction dicts (verbatim evidence only)
        """
        if not isinstance(clauses, list):
            raise ValueError("Extracted clauses must be a list")

        payload = {
            "schema_version": 1,
            "document_id": document_id,
            "extraction_version": EXTRACTION_VERSION,
            "clauses": clauses,
        }
        path = self._document_path(document_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def get_document_clauses(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Load extracted clauses for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of extracted clause dicts (empty if not found)
        """
        path = self._document_path(document_id)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        clauses = payload.get("clauses", [])
        if not isinstance(clauses, list):
            return []
        return clauses

    def delete_document_clauses(self, document_id: str) -> int:
        """
        Delete extracted clauses for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of payload files deleted (0 or 1).
        """
        path = self._document_path(document_id)
        if not path.exists():
            return 0
        try:
            path.unlink()
            return 1
        except OSError:
            # Best-effort delete; the caller should not fail the entire workflow
            # because of a missing or locked JSON file.
            return 0

    def _document_path(self, document_id: str) -> Path:
        return self.store_path / f"{document_id}.json"
