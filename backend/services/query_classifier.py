"""
Query classification service.
Classifies user queries to determine processing requirements.
"""
from typing import List, Dict, Any
from backend.models.document import QueryClassification
from backend.config import settings


class QueryClassifier:
    """Service for classifying user queries."""
    
    def __init__(self):
        """Initialize the query classifier."""
        self.legal_query_keywords = [
            "legal", "law", "statute", "regulation", "compliance", "governed by",
            "legality", "legal requirement", "legal obligation"
        ]
        
        self.hierarchy_query_keywords = [
            "override", "supersede", "prevail", "precedence", "hierarchy",
            "which takes precedence", "which applies", "law vs contract"
        ]
        
        self.query_type_patterns = {
            "termination": ["terminate", "termination", "dismiss", "dismissal", "end employment", "fire"],
            "legality": ["legal", "lawful", "legal requirement", "compliance", "governed by"],
            "benefits": ["benefit", "allowance", "housing", "medical", "insurance", "compensation"],
            "compensation": ["compensation", "salary", "wage", "payment", "pay", "remuneration"],
            "compliance": ["compliance", "comply", "requirement", "obligation", "must", "shall"],
            "notice": ["notice", "notification", "advance notice", "written notice"],
            "probation": ["probation", "probationary", "trial period", "trial"]
        }
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify user query to determine processing requirements.
        
        Args:
            query: User query string
            
        Returns:
            QueryClassification object
        """
        query_lower = query.lower()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Check if requires legal hierarchy
        requires_legal_hierarchy = self._requires_hierarchy_check(query_lower)
        
        # Extract scope topics
        scope_topics = self._extract_scope_topics(query_lower)
        
        # Check if legal query (requires citations)
        is_legal_query = self._is_legal_query(query_lower)
        
        return QueryClassification(
            query_type=query_type,
            requires_legal_hierarchy=requires_legal_hierarchy,
            scope_topics=scope_topics,
            is_legal_query=is_legal_query
        )
    
    def _detect_query_type(self, query_lower: str) -> str:
        """Detect the type of query."""
        # Check each query type pattern
        for query_type, patterns in self.query_type_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return query_type
        
        # Default to "general" if no specific type detected
        return "general"
    
    def _requires_hierarchy_check(self, query_lower: str) -> bool:
        """Determine if query requires legal hierarchy analysis."""
        # Check for hierarchy-related keywords
        if any(keyword in query_lower for keyword in self.hierarchy_query_keywords):
            return True
        
        # Check for legality questions
        if any(keyword in query_lower for keyword in self.legal_query_keywords):
            return True
        
        # Check for override/precedence questions
        override_patterns = [
            "can override", "override", "supersede", "which applies",
            "law vs", "contract vs", "precedence"
        ]
        
        return any(pattern in query_lower for pattern in override_patterns)
    
    def _extract_scope_topics(self, query_lower: str) -> List[str]:
        """Extract topics from query for scope checking."""
        topics = []
        
        # Check against covered topics
        for topic in settings.COVERED_TOPICS:
            if topic.lower() in query_lower:
                topics.append(topic)
        
        # Also check query type patterns
        for query_type, patterns in self.query_type_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                if query_type not in topics:
                    topics.append(query_type)
        
        return topics
    
    def _is_legal_query(self, query_lower: str) -> bool:
        """Determine if query is a legal/compliance query requiring citations."""
        # Check for legal keywords
        if any(keyword in query_lower for keyword in self.legal_query_keywords):
            return True
        
        # Check for compliance-related terms
        compliance_patterns = [
            "compliance", "legal requirement", "must", "shall", "obligation",
            "governed by", "in accordance with", "pursuant to"
        ]
        
        return any(pattern in query_lower for pattern in compliance_patterns)
    
    def is_out_of_scope(self, query: str, covered_topics: List[str] = None) -> bool:
        """
        Check if query is out of scope.
        
        Args:
            query: User query
            covered_topics: List of topics covered by documents (defaults to config)
            
        Returns:
            True if query is out of scope
        """
        if covered_topics is None:
            covered_topics = settings.COVERED_TOPICS
        
        query_lower = query.lower()
        
        # Out-of-scope patterns
        out_of_scope_patterns = [
            "vision 2030", "court precedent", "case law", "judicial",
            "court decision", "legal opinion", "attorney", "lawyer advice",
            "tax", "immigration", "visa", "residency permit"
        ]
        
        # Check for out-of-scope patterns
        if any(pattern in query_lower for pattern in out_of_scope_patterns):
            return True
        
        # Check if query topics match covered topics
        classification = self.classify_query(query)
        query_topics = classification.scope_topics
        
        # If no topics match covered topics, might be out of scope
        if query_topics:
            # Check if any query topic is in covered topics
            matches = [topic for topic in query_topics if topic.lower() in [ct.lower() for ct in covered_topics]]
            if not matches and query_topics:
                # Query has topics but none match covered topics
                return True
        
        return False

