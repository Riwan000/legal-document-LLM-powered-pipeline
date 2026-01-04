"""
Legal-Grade Clause Store.
Persistent storage for structured legal clauses with query capabilities.
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.models.clause import StructuredClause
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
        
        # Load existing data
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
        
        # Update clause-to-document mapping
        for clause in clauses:
            self.clause_to_document[clause.clause_id] = document_id
        
        # Save each clause to file
        for clause in clauses:
            clause_file = self.store_path / f"{clause.clause_id}.json"
            with open(clause_file, 'w', encoding='utf-8') as f:
                # Convert to dict for JSON serialization
                clause_dict = clause.model_dump()
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
        clause_file = self.store_path / f"{clause_id}.json"
        
        if not clause_file.exists():
            return None
        
        try:
            with open(clause_file, 'r', encoding='utf-8') as f:
                clause_dict = json.load(f)
                return StructuredClause(**clause_dict)
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
        
        # Apply filters
        filtered = []
        for clause in clauses:
            if clause_type and clause.type.value != clause_type:
                continue
            if authority_level and clause.authority_level.value != authority_level:
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
        """Load document index from disk."""
        index_file = self.store_path / "document_index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.document_index = json.load(f)
                
                # Rebuild clause-to-document mapping
                self.clause_to_document = {}
                for doc_id, clause_ids in self.document_index.items():
                    for clause_id in clause_ids:
                        self.clause_to_document[clause_id] = doc_id
            except Exception as e:
                print(f"Error loading clause store index: {str(e)}")
                self.document_index = {}
                self.clause_to_document = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored clauses."""
        total_clauses = sum(len(clause_ids) for clause_ids in self.document_index.values())
        total_documents = len(self.document_index)
        
        return {
            'total_clauses': total_clauses,
            'total_documents': total_documents,
            'store_path': str(self.store_path)
        }

