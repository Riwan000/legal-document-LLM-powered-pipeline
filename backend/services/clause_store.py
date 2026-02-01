"""
Legal-Grade Clause Store.
Persistent storage for structured legal clauses with query capabilities.

ClauseStore guarantees:
- No ranking logic
- Deterministic clause retrieval
- Normalized legal metadata
- In-memory cache for query path

Memory Usage:
- In-memory cache: ~1-5 MB per 1,000 clauses (typical legal documents)
- Each StructuredClause: ~1-5 KB (includes evidence blocks, metadata)
- For 10,000 clauses: ~10-50 MB total (acceptable for production)
- Cache is loaded once at startup, persists for application lifetime
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.models.clause import StructuredClause, ClauseType
from backend.config import settings


class ClauseStore:
    """Service for storing and retrieving structured legal clauses."""
    
    def __init__(self, store_path: Optional[Path] = None):
        """
        Initialize clause store.
        
        Args:
            store_path: Path to clause store directory (defaults to config)
        """
        self.store_path = store_path or settings.CLAUSE_STORE_PATH
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory index: document_id -> list of clause_ids
        self.document_index: Dict[str, List[str]] = {}
        
        # In-memory index: clause_id -> document_id
        self.clause_to_document: Dict[str, str] = {}

        # In-memory cache: clause_id -> StructuredClause
        # Loaded once at startup and updated on save/delete operations.
        self.clause_cache: Dict[str, StructuredClause] = {}

        # Canonical clause types for integration with weighted ranking.
        # Aligned with ClauseType enum (underscore form) so stored clauses match.
        self.CANONICAL_CLAUSE_TYPES = {ct.value for ct in ClauseType}
        
        # Load existing data (index + cache)
        self.load()
    
    def save_clauses(
        self,
        document_id: str,
        clauses: List[StructuredClause]
    ) -> None:
        """
        Save structured clauses for a document.
        
        Args:
            document_id: Document identifier
            clauses: List of StructuredClause objects
        """
        # Update document index
        clause_ids = [clause.clause_id for clause in clauses]
        self.document_index[document_id] = clause_ids
        
        for clause in clauses:
            # Normalize metadata once at save time
            if getattr(clause, "normalized_clause_type", None) is None:
                clause.normalized_clause_type = str(clause.type).lower()
            if getattr(clause, "normalized_authority_level", None) is None:
                clause.normalized_authority_level = str(clause.authority_level).lower()

            # Validate canonical clause type (best-effort; log warning only)
            n_type = clause.normalized_clause_type
            if n_type and n_type not in self.CANONICAL_CLAUSE_TYPES:
                print(
                    f"[ClauseStore] Warning: normalized_clause_type '{n_type}' "
                    f"not in CANONICAL_CLAUSE_TYPES for clause {clause.clause_id}"
                )

            # Update clause-to-document mapping
            self.clause_to_document[clause.clause_id] = document_id

            # Update in-memory cache
            self.clause_cache[clause.clause_id] = clause
        
        # Save each clause to file
        for clause in clauses:
            clause_file = self.store_path / f"{clause.clause_id}.json"
            with open(clause_file, 'w', encoding='utf-8') as f:
                # Convert to dict for JSON serialization
                clause_dict = clause.model_dump()
                # Add schema version for safe future migrations
                clause_dict["schema_version"] = 1
                json.dump(clause_dict, f, indent=2, ensure_ascii=False)
        
        # Save document index
        self._save_index()
    
    def get_clauses_by_document(self, document_id: str) -> List[StructuredClause]:
        """
        Get all clauses for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of StructuredClause objects
        """
        if document_id not in self.document_index:
            return []
        
        clause_ids = self.document_index[document_id]
        clauses = []
        
        for clause_id in clause_ids:
            clause = self.get_clause(clause_id)
            if clause:
                clauses.append(clause)
        
        return clauses
    
    def get_clause(self, clause_id: str) -> Optional[StructuredClause]:
        """
        Get a specific clause by ID.
        
        Args:
            clause_id: Clause identifier
            
        Returns:
            StructuredClause or None if not found
        """
        # Fast path: serve from in-memory cache
        clause = self.clause_cache.get(clause_id)
        if clause is not None:
            return clause

        # Fallback: attempt to load from disk (should be rare in steady state)
        clause_file = self.store_path / f"{clause_id}.json"
        if not clause_file.exists():
            return None
        
        try:
            with open(clause_file, 'r', encoding='utf-8') as f:
                clause_dict = json.load(f)

            schema_version = clause_dict.pop("schema_version", 0)
            clause = StructuredClause(**clause_dict)

            # Best-effort: store schema version in metadata for inspection
            if isinstance(clause.metadata, dict):
                clause.metadata.setdefault("schema_version", schema_version)

            # Normalize metadata if missing
            if getattr(clause, "normalized_clause_type", None) is None:
                clause.normalized_clause_type = str(clause.type).lower()
            if getattr(clause, "normalized_authority_level", None) is None:
                clause.normalized_authority_level = str(clause.authority_level).lower()

            # Cache for future fast access
            self.clause_cache[clause_id] = clause
            return clause
        except Exception as e:
            print(f"Error loading clause {clause_id}: {str(e)}")
            return None
    
    def query_clauses(
        self,
        document_id: Optional[str] = None,
        clause_type: Optional[str] = None,
        authority_level: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        can_override: Optional[bool] = None
    ) -> List[StructuredClause]:
        """
        Query clauses by filters.

        NOTE:
        - query_clauses() is intended for administrative and debugging use.
        - RAG retrieval should use get_candidate_clauses() for ranking-ready structures.
        
        Args:
            document_id: Filter by document ID
            clause_type: Filter by clause type
            authority_level: Filter by authority level
            jurisdiction: Filter by jurisdiction
            can_override: Filter by override capability
            
        Returns:
            List of matching StructuredClause objects
        """
        # Get clauses to search
        if document_id:
            clauses = self.get_clauses_by_document(document_id)
        else:
            # Search all clauses
            clauses = []
            for doc_id in self.document_index.keys():
                clauses.extend(self.get_clauses_by_document(doc_id))
        
        # Apply filters using normalized fields where available
        filtered: List[StructuredClause] = []
        for clause in clauses:
            norm_type = getattr(clause, "normalized_clause_type", None) or str(clause.type).lower()
            norm_auth = getattr(clause, "normalized_authority_level", None) or str(clause.authority_level).lower()

            if clause_type and norm_type != clause_type.lower():
                continue
            if authority_level and norm_auth != authority_level.lower():
                continue
            if jurisdiction and clause.jurisdiction != jurisdiction:
                continue
            if can_override is not None and clause.can_override_contract != can_override:
                continue
            
            filtered.append(clause)
        
        return filtered
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all clauses for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Number of clauses deleted
        """
        if document_id not in self.document_index:
            return 0
        
        clause_ids = self.document_index[document_id]
        deleted_count = 0
        
        for clause_id in clause_ids:
            clause_file = self.store_path / f"{clause_id}.json"
            if clause_file.exists():
                clause_file.unlink()
                deleted_count += 1
            
            # Remove from clause-to-document mapping
            if clause_id in self.clause_to_document:
                del self.clause_to_document[clause_id]

            # Remove from in-memory cache (Fix G: deletion consistency)
            self.clause_cache.pop(clause_id, None)
        
        # Remove from document index
        del self.document_index[document_id]
        
        # Save updated index
        self._save_index()
        
        return deleted_count
    
    def _save_index(self) -> None:
        """Save document index to disk."""
        index_file = self.store_path / "document_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_index, f, indent=2)
    
    def load(self) -> None:
        """
        Load document index and populate in-memory cache from disk.

        This runs once at startup and primes clause_cache for fast lookups.
        """
        index_file = self.store_path / "document_index.json"

        self.document_index = {}
        self.clause_to_document = {}
        self.clause_cache = {}
        
        if not index_file.exists():
            return

        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                self.document_index = json.load(f)
            
            # Rebuild clause-to-document mapping and populate cache
            for doc_id, clause_ids in self.document_index.items():
                for clause_id in clause_ids:
                    self.clause_to_document[clause_id] = doc_id

                    clause_file = self.store_path / f"{clause_id}.json"
                    if not clause_file.exists():
                        continue

                    try:
                        with open(clause_file, 'r', encoding='utf-8') as cf:
                            clause_dict = json.load(cf)

                        schema_version = clause_dict.pop("schema_version", 0)
                        clause = StructuredClause(**clause_dict)

                        # Best-effort: set schema_version in metadata
                        if isinstance(clause.metadata, dict):
                            clause.metadata.setdefault("schema_version", schema_version)

                        # Normalize metadata if not already present
                        if getattr(clause, "normalized_clause_type", None) is None:
                            clause.normalized_clause_type = str(clause.type).lower()
                        if getattr(clause, "normalized_authority_level", None) is None:
                            clause.normalized_authority_level = str(clause.authority_level).lower()

                        # Validate canonical type
                        n_type = clause.normalized_clause_type
                        if n_type and n_type not in self.CANONICAL_CLAUSE_TYPES:
                            print(
                                f"[ClauseStore] Warning: normalized_clause_type '{n_type}' "
                                f"not in CANONICAL_CLAUSE_TYPES for clause {clause.clause_id}"
                            )

                        self.clause_cache[clause_id] = clause
                    except Exception as ce:
                        print(f"Error loading clause {clause_id} during cache warm-up: {str(ce)}")
        except Exception as e:
            print(f"Error loading clause store index: {str(e)}")
            self.document_index = {}
            self.clause_to_document = {}
            self.clause_cache = {}

    def get_candidate_clauses(
        self,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Return ranking-ready clause structures (data only, no scores).

        Each result is a dict:
        {
          "clause": StructuredClause,
          "clause_id": str,
          "clause_type": str,
          "authority_level": str,
          "jurisdiction": str,
          "can_override": bool,
        }

        This is intended for integration with ranking components (e.g., RAG),
        while keeping ClauseStore itself ranking-agnostic.
        """
        results: List[Dict[str, Any]] = []

        if document_ids is None:
            doc_ids = list(self.document_index.keys())
        else:
            doc_ids = document_ids

        for doc_id in doc_ids:
            clause_ids = self.document_index.get(doc_id, [])
            for clause_id in clause_ids:
                clause = self.clause_cache.get(clause_id)
                if not clause:
                    continue

                norm_type = getattr(clause, "normalized_clause_type", None) or str(clause.type).lower()
                norm_auth = getattr(clause, "normalized_authority_level", None) or str(clause.authority_level).lower()

                results.append({
                    "clause": clause,
                    "clause_id": clause.clause_id,
                    "clause_type": norm_type,
                    "authority_level": norm_auth,
                    "jurisdiction": clause.jurisdiction,
                    "can_override": clause.can_override_contract,
                })

        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored clauses."""
        total_clauses = sum(len(clause_ids) for clause_ids in self.document_index.values())
        total_documents = len(self.document_index)
        
        return {
            'total_clauses': total_clauses,
            'total_documents': total_documents,
            'store_path': str(self.store_path)
        }

