"""
Legal reasoning service.

Implements rule-based reasoning for legal hierarchy, citation requirements,
and not-specified detection.

IMPORTANT DISCLAIMER:
This service provides heuristic, best-effort legal reasoning. It does NOT
provide legal advice or guaranteed correctness. All outputs should be treated
as assistive signals for RAG systems, not as final legal decisions.

This service:
- Reasons about pre-computed hierarchy levels (does not detect them)
- Calculates confidence using rule-based heuristics
- Determines citation requirements based on query classification
- Formats citations safely (never exposes internal IDs or hashes)

This service does NOT:
- Search documents or generate embeddings
- Decide legal truth or provide legal advice
- Guarantee correctness of any reasoning
"""
from typing import List, Dict, Any, Optional
import re
from backend.models.document import LegalHierarchyLevel, QueryClassification

# Similarity thresholds for relevance detection
MIN_REASONABLE_SIMILARITY = 0.4  # Threshold for considering chunks relevant
VERY_LOW_SIMILARITY = 0.25  # Threshold for marking as not specified
HIGH_CONFIDENCE_SCORE = 0.6  # Score threshold for high confidence
MEDIUM_CONFIDENCE_SCORE = 0.4  # Score threshold for medium confidence
MEDIUM_CONFIDENCE_AVG_SCORE = 0.45  # Average score threshold for medium confidence with multiple clauses
MIN_CLAUSES_FOR_MEDIUM_CONFIDENCE = 3  # Minimum number of clauses for medium confidence boost

# Text overlap detection thresholds
MIN_KEY_TERMS_FOUND = 2  # Minimum terms needed to consider content relevant
MAX_QUERY_WORDS = 5  # Top N query words to extract

# Stop words for query term extraction
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'who', 'what', 'when',
    'where', 'why', 'how', 'which', 'this', 'that', 'these', 'those'
}


class LegalReasoningService:
    """
    Service for rule-based legal reasoning.
    
    Provides heuristic reasoning about retrieved legal content, including:
    - Hierarchy analysis (LAW > CONTRACT > POLICY)
    - Citation requirement detection
    - Not-specified detection
    - Confidence calculation
    - Citation formatting
    
    All methods use conservative, rule-based heuristics and may return uncertain
    results for ambiguous cases. Intended to guide RAG response generation,
    not as final legal decisions.
    """
    
    def __init__(self):
        """Initialize the legal reasoning service."""
        pass
    
    def analyze_legal_hierarchy(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze legal hierarchy in retrieved chunks using heuristic rules.
        
        IMPORTANT: Hierarchy levels are assumed to be precomputed during document ingestion
        and stored in chunk metadata. This service does not re-detect hierarchy; it only
        reasons about pre-existing hierarchy_level values.
        
        Uses conservative heuristics to identify potential conflicts. Multiple authority
        levels present does not guarantee a real conflict - this is a best-effort signal.
        
        Args:
            query: User query (currently unused, reserved for future semantic analysis)
            chunks: List of retrieved chunks with hierarchy_level in metadata
            
        Returns:
            Dictionary with hierarchy analysis:
            - has_governing_law: bool - True if any LAW clauses found
            - has_contract_clauses: bool - True if any CONTRACT clauses found
            - has_policy_clauses: bool - True if any POLICY clauses found
            - has_potential_conflict: bool - True if multiple authority levels present
              (indicates possible, not guaranteed, conflict requiring human review)
            - precedence: str - "law", "contract", or "policy" (default: "contract")
            - law_clauses: List[Dict] - Chunks classified as LAW
            - contract_clauses: List[Dict] - Chunks classified as CONTRACT
            - policy_clauses: List[Dict] - Chunks classified as POLICY
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
        
        # Check for potential conflicts (conservative heuristic)
        has_potential_conflict = False
        precedence = "contract"  # Default
        
        if law_clauses and contract_clauses:
            # Multiple authority levels present - possible conflict
            # This is a simplified check - in production, would need semantic analysis
            has_potential_conflict = True
            precedence = "law"  # Law always prevails
        
        return {
            'has_governing_law': len(law_clauses) > 0,
            'has_contract_clauses': len(contract_clauses) > 0,
            'has_policy_clauses': len(policy_clauses) > 0,
            'has_potential_conflict': has_potential_conflict,
            'precedence': precedence,
            'law_clauses': law_clauses,
            'contract_clauses': contract_clauses,
            'policy_clauses': policy_clauses
        }
    
    def check_citation_requirement(self, query_classification: QueryClassification) -> bool:
        """
        Check if query requires citations using heuristic rules.
        
        Uses conservative rules to determine citation requirements. Legal/compliance
        queries and queries requiring hierarchy analysis always require citations.
        
        Args:
            query_classification: QueryClassification object with query_types (multi-label)
            
        Returns:
            True if citations are required, False otherwise
        """
        # Legal/compliance queries MUST have citations
        if query_classification.is_legal_query:
            return True
        
        # Queries requiring hierarchy check should have citations
        if query_classification.requires_legal_hierarchy:
            return True
        
        # Specific query types that require citations (check all query_types)
        citation_required_types = ["legality", "compliance", "governing law"]
        if any(qt in citation_required_types for qt in query_classification.query_types):
            return True
        
        return False
    
    def _extract_query_key_terms(
        self,
        query: str,
        query_classification: QueryClassification
    ) -> List[str]:
        """
        Extract key terms from query and classification for relevance checking.
        
        Args:
            query: User query text
            query_classification: QueryClassification object
            
        Returns:
            List of key terms extracted from query types, scope topics, and query text
        """
        key_terms = []
        
        # Add query type terms (exclude "general")
        for query_type in query_classification.query_types:
            if query_type != "general":
                key_terms.append(query_type)
        
        # Add scope topics
        key_terms.extend(query_classification.scope_topics)
        
        # Extract important keywords from query itself (more flexible matching)
        query_lower = query.lower()
        query_words = [
            w for w in re.findall(r'\b\w+\b', query_lower)
            if w not in STOP_WORDS and len(w) > 3
        ]
        key_terms.extend(query_words[:MAX_QUERY_WORDS])
        
        return key_terms
    
    def _has_text_overlap(
        self,
        chunks: List[Dict[str, Any]],
        terms: List[str]
    ) -> bool:
        """
        Check if chunks contain any of the given terms.
        
        Args:
            chunks: List of chunk dictionaries
            terms: List of terms to search for
            
        Returns:
            True if at least MIN_KEY_TERMS_FOUND terms are found across chunks
        """
        found_terms = set()
        for chunk in chunks:
            text_lower = chunk.get('text', '').lower()
            for term in terms:
                if term.lower() in text_lower:
                    found_terms.add(term)
        
        return len(found_terms) >= MIN_KEY_TERMS_FOUND
    
    def _has_reasonable_similarity(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Check if chunks have reasonable average similarity scores.
        
        Args:
            chunks: List of chunk dictionaries with score fields
            
        Returns:
            True if average score >= MIN_REASONABLE_SIMILARITY
        """
        if not chunks:
            return False
        
        avg_score = sum(c.get('score', 0.0) for c in chunks) / len(chunks)
        return avg_score >= MIN_REASONABLE_SIMILARITY
    
    def detect_not_specified(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        query_classification: QueryClassification
    ) -> bool:
        """
        Detect if query topic is not specified in documents using heuristic rules.
        
        Uses conservative, multi-stage checks to determine if retrieved chunks are
        relevant to the query. Returns True only if all checks indicate the topic
        is not specified.
        
        This is best-effort detection and may return false positives/negatives for
        ambiguous cases. Intended to guide RAG response generation, not as a final
        legal decision.
        
        Args:
            chunks: Retrieved chunks with similarity scores and metadata
            query: User query text
            query_classification: QueryClassification object
            
        Returns:
            True if topic appears not to be specified in documents, False otherwise
        """
        if not chunks:
            return True
        
        # Keyword matches indicate content exists, even if semantic similarity is low
        if any(c.get('keyword_match', False) for c in chunks):
            return False
        
        # Extract and check for text overlap
        key_terms = self._extract_query_key_terms(query, query_classification)
        if self._has_text_overlap(chunks, key_terms):
            return False
        
        # Check similarity as fallback (even if terms don't match exactly)
        if self._has_reasonable_similarity(chunks):
            return False
        
        # Additional conservative check: very low similarity AND no terms found
        if chunks:
            avg_score = sum(c.get('score', 0.0) for c in chunks) / len(chunks)
            if avg_score < VERY_LOW_SIMILARITY and not key_terms:
                return True
        
        return False
    
    def calculate_confidence(
        self,
        chunks: List[Dict[str, Any]],
        hierarchy_analysis: Dict[str, Any],
        has_explicit_clause: bool
    ) -> str:
        """
        Calculate confidence level for answer using heuristic rules.
        
        Uses conservative, rule-based heuristics to estimate confidence.
        May return uncertain results for ambiguous cases. Intended to guide
        RAG response generation, not as a final legal decision.
        
        Confidence levels:
        - "high": Explicit clause found with strong similarity (score >= HIGH_CONFIDENCE_SCORE)
        - "medium": Multiple consistent clauses OR governing law present OR explicit clause with moderate similarity
        - "low": Ambiguous, missing info, or very low similarity
        
        Args:
            chunks: Retrieved chunks with similarity scores
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
            if top_score >= HIGH_CONFIDENCE_SCORE:
                return "high"
            elif top_score >= MEDIUM_CONFIDENCE_SCORE:
                return "medium"
        
        # Medium confidence: multiple consistent clauses with reasonable average score
        if len(chunks) >= MIN_CLAUSES_FOR_MEDIUM_CONFIDENCE:
            avg_score = sum(c.get('score', 0.0) for c in chunks) / len(chunks)
            if avg_score >= MEDIUM_CONFIDENCE_AVG_SCORE:
                return "medium"
        
        # Medium confidence: governed by law reference
        if hierarchy_analysis.get('has_governing_law', False):
            return "medium"
        
        # Low confidence: ambiguous or missing info
        return "low"
    
    def format_citation(self, chunk: Dict[str, Any]) -> str:
        """
        Format citation for a chunk using safe, normalized metadata access.
        
        Uses display_name for citations (never exposes document_hash or internal IDs).
        Normalizes metadata access to handle both nested and top-level fields consistently.
        
        This is a formatting utility - it does not validate legal correctness of citations.
        
        Args:
            chunk: Chunk dictionary with metadata (may have nested or top-level fields)
            
        Returns:
            Formatted citation string (e.g., "Employment Contract (2023), Clause 11, Page 4 (Governing Law)")
        """
        # Normalize metadata access - prefer nested metadata, fallback to top-level
        metadata = chunk.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Extract all fields from normalized metadata first, with fallbacks
        clause_id = metadata.get('clause_id') or chunk.get('clause_id')
        clause_types = metadata.get('clause_types', [])
        clause_type = metadata.get('type') or (clause_types[0] if clause_types else None)
        hierarchy_level = metadata.get('hierarchy_level', chunk.get('hierarchy_level', 'contract'))
        display_name = chunk.get('display_name', chunk.get('document_id', 'Document'))
        page_number = chunk.get('page_number', 'N/A')
        
        # Build citation with display_name (never expose document_hash)
        citation_parts = []
        
        # Start with document display name
        citation_parts.append(display_name)
        
        # Add clause identifier
        if clause_id:
            citation_parts.append(f"Clause {clause_id}")
        elif clause_type:
            citation_parts.append(f"Clause ({clause_type})")
        else:
            citation_parts.append("Clause")
        
        # Add page number
        citation_parts.append(f"Page {page_number}")
        
        # Add hierarchy indicator if relevant
        if hierarchy_level == "law":
            citation_parts.append("(Governing Law)")
        elif clause_type:
            citation_parts.append(f"({clause_type})")
        
        return ", ".join(citation_parts)

