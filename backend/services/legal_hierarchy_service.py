"""
Legal hierarchy detection service.
Detects hierarchy levels (law > contract > policy) and identifies supremacy clauses.
"""
from typing import List, Dict, Any, Optional
from backend.models.document import LegalHierarchyLevel
from backend.config import settings


class LegalHierarchyService:
    """Service for detecting and ranking legal hierarchy in clauses."""
    
    def __init__(self):
        """Initialize the legal hierarchy service."""
        self.law_keywords = settings.LEGAL_HIERARCHY_KEYWORDS.get("law", [])
        self.supremacy_keywords = settings.LEGAL_HIERARCHY_KEYWORDS.get("supremacy", [])
        self.contract_keywords = settings.LEGAL_HIERARCHY_KEYWORDS.get("contract", [])
    
    def detect_hierarchy_level(self, clause_text: str) -> LegalHierarchyLevel:
        """
        Auto-detect hierarchy level from clause text.
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            LegalHierarchyLevel enum value
        """
        text_lower = clause_text.lower()
        
        # Check for law references first (highest priority)
        for keyword in self.law_keywords:
            if keyword.lower() in text_lower:
                return LegalHierarchyLevel.LAW
        
        # Check for contract keywords
        for keyword in self.contract_keywords:
            if keyword.lower() in text_lower:
                return LegalHierarchyLevel.CONTRACT
        
        # Default to contract if no clear indicators
        return LegalHierarchyLevel.CONTRACT
    
    def is_supremacy_clause(self, clause_text: str) -> bool:
        """
        Detect if clause is a supremacy clause (overrides or is overridden by law).
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            True if clause has supremacy implications
        """
        text_lower = clause_text.lower()
        
        # Check for supremacy keywords
        for keyword in self.supremacy_keywords:
            if keyword.lower() in text_lower:
                return True
        
        # Check for patterns like "subject to law", "in compliance with law"
        supremacy_patterns = [
            "subject to",
            "in compliance with",
            "notwithstanding",
            "override",
            "supersede",
            "prevail"
        ]
        
        for pattern in supremacy_patterns:
            if pattern in text_lower:
                # Check if it's followed by law-related terms
                pattern_idx = text_lower.find(pattern)
                if pattern_idx != -1:
                    # Look for law keywords in surrounding context
                    context = text_lower[max(0, pattern_idx - 50):pattern_idx + 100]
                    if any(law_kw in context for law_kw in self.law_keywords):
                        return True
        
        return False
    
    def extract_jurisdiction(self, clause_text: str) -> Optional[str]:
        """
        Extract jurisdiction from clause text.
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            Jurisdiction string or None
        """
        text_lower = clause_text.lower()
        
        # Common jurisdiction patterns
        jurisdictions = {
            "saudi arabia": ["saudi arabia", "saudi", "kingdom of saudi arabia", "ksa"],
            "uae": ["united arab emirates", "uae", "emirates"],
            "qatar": ["qatar"],
            "kuwait": ["kuwait"],
            "bahrain": ["bahrain"],
            "oman": ["oman"]
        }
        
        for jurisdiction, patterns in jurisdictions.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return jurisdiction.title()
        
        return None
    
    def extract_topics(self, clause_text: str) -> List[str]:
        """
        Extract topics/keywords from clause text.
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            List of topic keywords
        """
        text_lower = clause_text.lower()
        topics = []
        
        # Common legal topics
        topic_keywords = {
            "termination": ["termination", "terminate", "dismissal", "dismiss"],
            "compensation": ["compensation", "salary", "wage", "payment", "remuneration"],
            "benefits": ["benefits", "allowance", "housing", "medical", "insurance"],
            "notice": ["notice", "notification", "advance notice"],
            "probation": ["probation", "probationary", "trial period"],
            "governing law": ["governing law", "applicable law", "law", "statute"],
            "dispute resolution": ["dispute", "arbitration", "mediation", "court"],
            "confidentiality": ["confidential", "non-disclosure", "nda"],
            "liability": ["liability", "responsible", "responsibility"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def rank_by_authority(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank clauses by legal authority (LAW > CONTRACT > POLICY).
        
        Args:
            clauses: List of clause dictionaries with hierarchy_level in metadata
            
        Returns:
            Sorted list with LAW clauses first, then CONTRACT, then POLICY
        """
        # Define hierarchy order (lower number = higher priority)
        hierarchy_order = {
            LegalHierarchyLevel.LAW: 0,
            LegalHierarchyLevel.CONTRACT: 1,
            LegalHierarchyLevel.POLICY: 2
        }
        
        def get_priority(clause: Dict[str, Any]) -> tuple:
            """Get sorting priority for clause."""
            # Extract hierarchy level from metadata
            metadata = clause.get('metadata', {})
            hierarchy_str = metadata.get('hierarchy_level', LegalHierarchyLevel.CONTRACT.value)
            
            try:
                hierarchy_level = LegalHierarchyLevel(hierarchy_str)
            except ValueError:
                hierarchy_level = LegalHierarchyLevel.CONTRACT
            
            priority = hierarchy_order.get(hierarchy_level, 1)
            
            # Boost supremacy clauses (within same hierarchy level)
            is_supremacy = metadata.get('legal_supremacy', False)
            supremacy_boost = -0.5 if is_supremacy else 0
            
            # Use similarity score as tiebreaker (higher is better)
            score = clause.get('score', 0.0)
            
            return (priority + supremacy_boost, -score)  # Negative score for descending order
        
        # Sort by priority
        sorted_clauses = sorted(clauses, key=get_priority)
        
        return sorted_clauses

