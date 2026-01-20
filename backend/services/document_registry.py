"""
Document registry service for managing document identity, versions, and metadata.
Uses SQLite to persist document information with proper separation of internal hashes
from user-facing display names.
"""
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager

from backend.config import settings


class DocumentRegistry:
    """Registry for document identity, versions, and metadata."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize document registry.
        
        Args:
            db_path: Path to SQLite database (defaults to data/documents.db)
        """
        self.db_path = db_path or settings.DOCUMENTS_PATH.parent / "documents.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper transaction handling."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    document_hash TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    is_latest BOOLEAN NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    uploaded_by TEXT,
                    file_path TEXT,
                    total_pages INTEGER,
                    total_chunks INTEGER,
                    UNIQUE(document_hash, version)
                )
            """)
            
            # Index for fast hash lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_hash 
                ON documents(document_hash)
            """)
            
            # Index for latest version lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_latest 
                ON documents(is_latest, document_id)
            """)
    
    def compute_hash(self, file_content: bytes) -> str:
        """
        Compute SHA-256 hash of file content.
        
        Args:
            file_content: File content as bytes
            
        Returns:
            SHA-256 hash as hex string
        """
        return hashlib.sha256(file_content).hexdigest()
    
    def get_next_document_id(self) -> str:
        """
        Generate next incremental document ID (DOC-####).
        
        Returns:
            Document ID like DOC-0001
        """
        with self._get_connection() as conn:
            # Get max document number
            result = conn.execute("""
                SELECT MAX(CAST(SUBSTR(document_id, 5) AS INTEGER)) as max_num
                FROM documents
                WHERE document_id LIKE 'DOC-%'
            """).fetchone()
            
            max_num = result['max_num'] if result['max_num'] is not None else 0
            next_num = max_num + 1
            
            return f"DOC-{next_num:04d}"
    
    def find_by_hash(self, document_hash: str, is_latest: bool = True) -> Optional[Dict[str, Any]]:
        """
        Find document by hash.
        
        Args:
            document_hash: SHA-256 hash of document content
            is_latest: If True, only return latest version
            
        Returns:
            Document record as dict or None
        """
        with self._get_connection() as conn:
            query = """
                SELECT * FROM documents 
                WHERE document_hash = ? 
            """
            params = [document_hash]
            
            if is_latest:
                query += " AND is_latest = 1"
            
            query += " ORDER BY version DESC LIMIT 1"
            
            row = conn.execute(query, params).fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_document(self, document_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get document by ID and optional version.
        
        Args:
            document_id: Document ID (DOC-####)
            version: Optional version number (defaults to latest)
            
        Returns:
            Document record as dict or None
        """
        with self._get_connection() as conn:
            if version:
                query = """
                    SELECT * FROM documents 
                    WHERE document_id = ? AND version = ?
                """
                params = [document_id, version]
            else:
                query = """
                    SELECT * FROM documents 
                    WHERE document_id = ? AND is_latest = 1
                """
                params = [document_id]
            
            row = conn.execute(query, params).fetchone()
            
            if row:
                return dict(row)
            return None
    
    def register_document(
        self,
        document_hash: str,
        original_filename: str,
        document_type: str = "document",
        display_name: Optional[str] = None,
        uploaded_by: Optional[str] = None,
        file_path: Optional[str] = None,
        total_pages: Optional[int] = None,
        total_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Register a new document or create a new version.
        
        Args:
            document_hash: SHA-256 hash of document content
            original_filename: Original filename from upload
            document_type: Type of document ("document" or "template")
            display_name: User-friendly display name (defaults to filename without extension)
            uploaded_by: Optional user identifier
            file_path: Optional path to stored file
            total_pages: Optional total page count
            total_chunks: Optional total chunk count
            
        Returns:
            Document record as dict
        """
        # Check if document with this hash already exists
        existing = self.find_by_hash(document_hash, is_latest=True)
        
        if existing:
            # Same content, return existing document
            return existing
        
        # Generate display name if not provided
        if not display_name:
            display_name = Path(original_filename).stem
        
        # Determine document_id
        # If this is an update to an existing document (same display_name), reuse document_id
        # Otherwise, generate new document_id
        document_id = None
        version = 1
        
        # Check if there's an existing document with same display_name
        with self._get_connection() as conn:
            existing_by_name = conn.execute("""
                SELECT document_id, MAX(version) as max_version
                FROM documents
                WHERE display_name = ? AND document_type = ?
                GROUP BY document_id
            """, [display_name, document_type]).fetchone()
            
            if existing_by_name:
                document_id = existing_by_name['document_id']
                version = (existing_by_name['max_version'] or 0) + 1
                
                # Mark old versions as not latest
                conn.execute("""
                    UPDATE documents 
                    SET is_latest = 0 
                    WHERE document_id = ?
                """, [document_id])
            else:
                # New document, generate ID
                document_id = self.get_next_document_id()
        
        # Insert new document record
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO documents (
                    document_id, document_hash, display_name, original_filename,
                    document_type, version, is_latest, uploaded_by, file_path,
                    total_pages, total_chunks
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                document_id, document_hash, display_name, original_filename,
                document_type, version, True, uploaded_by, file_path,
                total_pages, total_chunks
            ])
        
        # Return the new document record
        return self.get_document(document_id, version)
    
    def update_display_name(self, document_id: str, new_display_name: str) -> bool:
        """
        Update display name for a document (all versions).
        
        Args:
            document_id: Document ID
            new_display_name: New display name
            
        Returns:
            True if updated, False if document not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                UPDATE documents 
                SET display_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
            """, [new_display_name, document_id])
            
            return cursor.rowcount > 0
    
    def list_documents(
        self,
        document_type: Optional[str] = None,
        include_versions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all documents.
        
        Args:
            document_type: Optional filter by type ("document" or "template")
            include_versions: If True, include all versions; if False, only latest
            
        Returns:
            List of document records
        """
        with self._get_connection() as conn:
            query = "SELECT * FROM documents WHERE 1=1"
            params = []
            
            if document_type:
                query += " AND document_type = ?"
                params.append(document_type)
            
            if not include_versions:
                query += " AND is_latest = 1"
            
            query += " ORDER BY document_id, version DESC"
            
            rows = conn.execute(query, params).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_versions(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of version records
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM documents 
                WHERE document_id = ?
                ORDER BY version DESC
            """, [document_id]).fetchall()
            
            return [dict(row) for row in rows]
    
    def delete_document(self, document_id: str, version: Optional[int] = None) -> int:
        """
        Delete document(s).
        
        Args:
            document_id: Document ID
            version: Optional version number (if None, deletes all versions)
            
        Returns:
            Number of records deleted
        """
        with self._get_connection() as conn:
            if version:
                cursor = conn.execute("""
                    DELETE FROM documents 
                    WHERE document_id = ? AND version = ?
                """, [document_id, version])
            else:
                cursor = conn.execute("""
                    DELETE FROM documents 
                    WHERE document_id = ?
                """, [document_id])
            
            return cursor.rowcount
