"""
Legal reasoning service.
Implements rule-based reasoning for legal hierarchy, citation requirements, and not-specified detection.
"""
from typing import List, Dict, Any, Optional
from backend.models.document import LegalHierarchyLevel, QueryClassification
from backend.services.legal_hierarchy_service import LegalHierarchyService


class LegalReasoningService:
    """Service for rule-based legal reasoning."""
    
    def __init__(self):
        """Initialize the legal reasoning service."""
        self.hierarchy_service = LegalHierarchyService()
    
    def analyze_legal_hierarchy(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze legal hierarchy in retrieved chunks.
        
        Args:
            query: User query
            chunks: List of retrieved chunks with metadata
            
        Returns:
            Dictionary with hierarchy analysis:
            - has_governing_law: bool
            - has_contract_clauses: bool
            - has_conflict: bool
            - precedence: str (law/contract/policy)
            - law_clauses: List[Dict]
            - contract_clauses: List[Dict]
        """
        law_clauses = []
        contract_clauses = []
        policy_clauses = []
        
        for chunk in chunks:
            # Metadata is already spread into chunk dict, not nested
            hierarchy_level = chunk.get('hierarchy_level', 'contract')
            
            try:
                level = LegalHierarchyLevel(hierarchy_level)
            except ValueError:
                level = LegalHierarchyLevel.CONTRACT
            
            # Categorize chunks by hierarchy level
            if level == LegalHierarchyLevel.LAW:
                law_clauses.append(chunk)
            elif level == LegalHierarchyLevel.CONTRACT:
                contract_clauses.append(chunk)
            elif level == LegalHierarchyLevel.POLICY:
                policy_clauses.append(chunk)
            else:
                # Default to contract if unknown
                contract_clauses.append(chunk)
        
        # Check for conflicts
        has_conflict = False
        precedence = "contract"  # Default
        
        if law_clauses and contract_clauses:
            # Check if contract clauses conflict with law
            # This is a simplified check - in production, would need semantic analysis
            has_conflict = True
            precedence = "law"  # Law always prevails
        
        return {
            'has_governing_law': len(law_clauses) > 0,
            'has_contract_clauses': len(contract_clauses) > 0,
            'has_policy_clauses': len(policy_clauses) > 0,
            'has_conflict': has_conflict,
            'precedence': precedence,
            'law_clauses': law_clauses,
            'contract_clauses': contract_clauses,
            'policy_clauses': policy_clauses
        }
    
    def check_citation_requirement(self, query_classification: QueryClassification) -> bool:
        """
        Check if query requires citations.
        
        Args:
            query_classification: QueryClassification object
            
        Returns:
            True if citations are required
        """
        # Legal/compliance queries MUST have citations
        if query_classification.is_legal_query:
            return True
        
        # Queries requiring hierarchy check should have citations
        if query_classification.requires_legal_hierarchy:
            return True
        
        # Specific query types that require citations
        citation_required_types = ["legality", "compliance", "governing law"]
        if query_classification.query_type in citation_required_types:
            return True
        
        return False
    
    def detect_not_specified(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        query_classification: QueryClassification
    ) -> bool:
        """
        Detect if query topic is not specified in documents.
        
        Args:
            chunks: Retrieved chunks
            query: User query
            query_classification: QueryClassification object
            
        Returns:
            True if topic is not specified
        """
        if not chunks:
            return True
        
        # If chunks were found via keyword matching, they're likely relevant
        # Don't mark as "not specified" if keyword matches exist
        keyword_matches = [c for c in chunks if c.get('keyword_match', False)]
        if keyword_matches:
            # Keyword matches indicate content exists, even if semantic similarity is low
            return False
        
        query_lower = query.lower()
        query_type = query_classification.query_type
        
        # Extract key terms from query
        key_terms = []
        
        # Add query type terms
        if query_type != "general":
            key_terms.append(query_type)
        
        # Add scope topics
        key_terms.extend(query_classification.scope_topics)
        
        # Extract important keywords from query itself (more flexible matching)
        import re
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'who', 'what', 'when', 'where', 'why', 'how', 'which', 'this', 'that', 'these', 'those'}
        query_words = [w for w in re.findall(r'\b\w+\b', query_lower) if w not in stop_words and len(w) > 3]
        key_terms.extend(query_words[:5])  # Add top 5 query words
        
        # Check if any chunks contain these terms
        found_terms = set()
        for chunk in chunks:
            text_lower = chunk.get('text', '').lower()
            for term in key_terms:
                if term.lower() in text_lower:
                    found_terms.add(term)
        
        # If we found at least 2 key terms, content is likely relevant
        if len(found_terms) >= 2:
            return False
        
        # If no key terms found, topic might not be specified
        # But be more lenient - if we have chunks, they might still be relevant
        if key_terms and not found_terms:
            # Check if chunks have reasonable scores (even if terms don't match exactly)
            if chunks:
                avg_score = sum(c.get('score', 0.0) for c in chunks) / len(chunks)
                # If average score is reasonable, don't mark as not specified
                if avg_score >= 0.4:  # More lenient threshold
                    return False
            return True
        
        # Additional check: if chunks are very low similarity AND no terms match, might not be relevant
        if chunks:
            avg_score = sum(c.get('score', 0.0) for c in chunks) / len(chunks)
            # Only mark as not specified if score is very low AND no terms found
            if avg_score < 0.25 and not found_terms:  # Very low threshold
                return True
        
        return False
    
    def calculate_confidence(
        self,
        chunks: List[Dict[str, Any]],
        hierarchy_analysis: Dict[str, Any],
        has_explicit_clause: bool
    ) -> str:
        """
        Calculate confidence level for answer.
        
        Args:
            chunks: Retrieved chunks
            hierarchy_analysis: Result from analyze_legal_hierarchy
            has_explicit_clause: Whether explicit clause was found
            
        Returns:
            Confidence level: "high", "medium", or "low"
        """
        if not chunks:
            return "low"
        
        # High confidence: explicit clause found with good similarity
        if has_explicit_clause:
            top_score = chunks[0].get('score', 0.0) if chunks else 0.0
            if top_score >= 0.6:
                return "high"
            elif top_score >= 0.4:
                return "medium"
        
        # Medium confidence: governed by law reference
        if hierarchy_analysis.get('has_governing_law', False):
            return "medium"
        
        # Low confidence: ambiguous or missing info
        return "low"
    
    def format_citation(self, chunk: Dict[str, Any]) -> str:
        """
        Format citation for a chunk.
        Uses display_name (never document_hash or internal IDs).
        
        Args:
            chunk: Chunk dictionary with metadata
            
        Returns:
            Formatted citation string (e.g., "Employment Contract (2023), Page 4 (Governing Law)")
        """
        metadata = chunk.get('metadata', {}) if isinstance(chunk.get('metadata'), dict) else {}
        # Use display_name for citation (never expose document_hash)
        display_name = chunk.get('display_name', chunk.get('document_id', 'Document'))
        page_number = chunk.get('page_number', 'N/A')
        # Preserve original clause_id logic
        clause_id = metadata.get('clause_id') or chunk.get('clause_id')
        clause_type = metadata.get('type') or (metadata.get('clause_types', [])[0] if metadata.get('clause_types') else None)
        hierarchy_level = metadata.get('hierarchy_level', 'contract')
        
        # Build citation with display_name (never document_hash)
        citation_parts = []
        
        # Start with document display name
        citation_parts.append(display_name)
        
        if clause_id:
            citation_parts.append(f"Clause {clause_id}")
        elif clause_type:
            citation_parts.append(f"Clause ({clause_type})")
        else:
            citation_parts.append("Clause")
        
        citation_parts.append(f"Page {page_number}")
        
        # Add hierarchy indicator if relevant
        if hierarchy_level == "law":
            citation_parts.append("(Governing Law)")
        elif clause_type:
            citation_parts.append(f"({clause_type})")
        
        return ", ".join(citation_parts)

