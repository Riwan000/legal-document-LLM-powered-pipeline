"""
Persist and query ingestion metadata per document/version.
Stores chunking strategy, structure detection, and embedding model version.
"""
import sqlite3
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager
from datetime import datetime

from backend.config import settings
from backend.models.document_structure import IngestionMetadata


class IngestionMetadataStore:
    """Store for ingestion run metadata (strategy, structure, embedding version)."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.DOCUMENTS_PATH.parent / "documents.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_metadata (
                    document_id TEXT NOT NULL,
                    ingestion_version INTEGER NOT NULL,
                    chunking_strategy TEXT NOT NULL,
                    structure_detected INTEGER NOT NULL,
                    estimated_clause_count INTEGER NOT NULL,
                    embedding_model_version TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (document_id, ingestion_version)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ingestion_document
                ON ingestion_metadata(document_id)
            """)

    def save(self, metadata: IngestionMetadata) -> None:
        """Persist one ingestion metadata record (upsert by document_id + version)."""
        with self._get_connection() as conn:
            created = metadata.created_at
            created_str = created.isoformat() if isinstance(created, datetime) else (created or datetime.now().isoformat())
            conn.execute("""
                INSERT OR REPLACE INTO ingestion_metadata (
                    document_id, ingestion_version, chunking_strategy,
                    structure_detected, estimated_clause_count, embedding_model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                metadata.document_id,
                metadata.ingestion_version,
                metadata.chunking_strategy,
                1 if metadata.structure_detected else 0,
                metadata.estimated_clause_count,
                metadata.embedding_model_version,
                created_str,
            ])

    def get_by_document(
        self, document_id: str, version: Optional[int] = None
    ) -> Optional[IngestionMetadata]:
        """Return ingestion metadata for a document (latest version if version is None)."""
        with self._get_connection() as conn:
            if version is not None:
                row = conn.execute("""
                    SELECT * FROM ingestion_metadata
                    WHERE document_id = ? AND ingestion_version = ?
                """, [document_id, version]).fetchone()
            else:
                row = conn.execute("""
                    SELECT * FROM ingestion_metadata
                    WHERE document_id = ?
                    ORDER BY ingestion_version DESC LIMIT 1
                """, [document_id]).fetchone()
            if not row:
                return None
            d = dict(row)
            d["structure_detected"] = bool(d.get("structure_detected", 0))
            ca = d.get("created_at")
            if isinstance(ca, str):
                created_at = datetime.fromisoformat(ca)
            else:
                created_at = ca if ca is not None else datetime.now()
            return IngestionMetadata(
                document_id=d["document_id"],
                ingestion_version=d["ingestion_version"],
                chunking_strategy=d["chunking_strategy"],
                structure_detected=d["structure_detected"],
                estimated_clause_count=d["estimated_clause_count"],
                embedding_model_version=d["embedding_model_version"],
                created_at=created_at,
            )

    def list_for_document(self, document_id: str) -> List[IngestionMetadata]:
        """Return all ingestion metadata records for a document (all versions)."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM ingestion_metadata
                WHERE document_id = ?
                ORDER BY ingestion_version DESC
            """, [document_id]).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["structure_detected"] = bool(d.get("structure_detected", 0))
            ca = d.get("created_at")
            created_at = datetime.fromisoformat(ca) if isinstance(ca, str) else (ca if ca is not None else datetime.now())
            out.append(IngestionMetadata(
                document_id=d["document_id"],
                ingestion_version=d["ingestion_version"],
                chunking_strategy=d["chunking_strategy"],
                structure_detected=d["structure_detected"],
                estimated_clause_count=d["estimated_clause_count"],
                embedding_model_version=d["embedding_model_version"],
                created_at=created_at,
            ))
        return out
