"""
SQLite-backed session store for conversational RAG sessions.

Sessions are stored in data/sessions.db (separate from documents.db).
Each session is serialized as a JSON blob in the `data` column.
TTL expiry is evaluated lazily on read (no background purge process needed).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from backend.config import settings
from backend.models.session import ChatSession, SessionMode

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/sessions.db")


class SessionStore:
    """Persistent SQLite store for ChatSession objects."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
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

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    document_id  TEXT NOT NULL,
                    mode         TEXT NOT NULL,
                    data         TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    last_active_at TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_doc ON sessions(document_id)"
            )

    def _is_expired(self, last_active_at: str) -> bool:
        """Return True if the session has exceeded SESSION_TTL_HOURS."""
        if settings.SESSION_PERSIST_FOREVER:
            return False
        if settings.SESSION_TTL_HOURS <= 0:
            return False
        last = datetime.fromisoformat(last_active_at)
        cutoff = datetime.utcnow() - timedelta(hours=settings.SESSION_TTL_HOURS)
        return last < cutoff

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, document_id: str, mode: SessionMode) -> ChatSession:
        """Create and persist a new session. Returns the ChatSession."""
        from uuid import uuid4
        session = ChatSession(
            session_id=str(uuid4()),
            document_id=document_id,
            mode=mode,
        )
        now_iso = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO sessions
                   (session_id, document_id, mode, data, created_at, last_active_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session.session_id,
                    document_id,
                    mode.value,
                    session.model_dump_json(),
                    now_iso,
                    now_iso,
                ),
            )
        logger.info("Created session %s for document %s", session.session_id, document_id)
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Load a session by ID.

        Returns None if not found or if TTL has expired (expired session is
        also deleted from the store on the spot).
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

        if row is None:
            return None

        if self._is_expired(row["last_active_at"]):
            logger.info("Session %s has expired; deleting.", session_id)
            self.delete_session(session_id)
            return None

        return ChatSession.model_validate_json(row["data"])

    def save_session(self, session: ChatSession) -> None:
        """Persist the full session state (after mutations by SessionManager)."""
        now_iso = datetime.utcnow().isoformat()
        session.last_active_at = datetime.utcnow()
        with self._conn() as conn:
            conn.execute(
                """UPDATE sessions
                   SET data = ?, last_active_at = ?
                   WHERE session_id = ?""",
                (session.model_dump_json(), now_iso, session.session_id),
            )

    def append_message(self, session_id: str, message) -> Optional[ChatSession]:
        """
        Append a SessionMessage to the session and persist.

        Returns the updated session or None if not found / expired.
        """
        session = self.get_session(session_id)
        if session is None:
            return None
        session.messages.append(message)
        self.save_session(session)
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if a row was deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
        return cursor.rowcount > 0
