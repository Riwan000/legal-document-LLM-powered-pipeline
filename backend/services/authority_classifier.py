"""
Authority level classification service.
Classifies clauses by authority level (supreme, regulatory, contractual, administrative).
"""
from typing import Dict, List, Optional, Any
from backend.models.clause import AuthorityLevel, StructuredClause
from backend.config import settings


class AuthorityClassifier:
    """Service for classifying clause authority levels."""
    
    def __init__(self):
        """Initialize authority classifier with detection patterns."""
        self.supreme_keywords = [
            "law", "statute", "regulation", "governed by", "pursuant to",
            "in accordance with", "Saudi Labor Law", "Labor Law", "Labour Law",
            "Saudi Labour Law", "Kingdom of Saudi Arabia", "Ministry of Labor"
        ]
        
        self.regulatory_keywords = [
            "compliance", "regulatory", "emigration", "visa", "residency",
            "work permit", "iqama", "Ministry of Interior"
        ]
        
        self.administrative_keywords = [
            "instruction", "notice", "date", "effective date", "commencement",
            "schedule", "appendix", "annex"
        ]
    
    def classify_authority(
        self,
        clause_text: str,
        clause_type: str,
        jurisdiction: Optional[str] = None
    ) -> AuthorityLevel:
        """
        Classify clause authority level.
        
        Args:
            clause_text: Clause text content
            clause_type: Clause type category
            jurisdiction: Optional jurisdiction
            
        Returns:
            AuthorityLevel enum value
        """
        text_lower = clause_text.lower()
        
        # Check for supreme (external law) indicators
        if any(keyword in text_lower for keyword in self.supreme_keywords):
            return AuthorityLevel.SUPREME
        
        # Check for regulatory indicators
        if any(keyword in text_lower for keyword in self.regulatory_keywords):
            return AuthorityLevel.REGULATORY
        
        # Check for administrative indicators
        if any(keyword in text_lower for keyword in self.administrative_keywords):
            return AuthorityLevel.ADMINISTRATIVE
        
        # Default to contractual
        return AuthorityLevel.CONTRACTUAL
    
    def determine_override_capability(
        self,
        authority_level: AuthorityLevel,
        clause_text: str
    ) -> Dict[str, Any]:
        """
        Determine if clause can override contract and what it overrides.
        
        Args:
            authority_level: Classified authority level
            clause_text: Clause text
            
        Returns:
            Dict with 'can_override_contract' and 'overrides' list
        """
        text_lower = clause_text.lower()
        
        # Supreme law always overrides contract
        if authority_level == AuthorityLevel.SUPREME:
            return {
                'can_override_contract': True,
                'overrides': ['contractual', 'administrative']
            }
        
        # Regulatory may override contract
        if authority_level == AuthorityLevel.REGULATORY:
            # Check for explicit override language
            override_keywords = [
                "override", "supersede", "prevail", "notwithstanding",
                "subject to", "in compliance with"
            ]
            if any(keyword in text_lower for keyword in override_keywords):
                return {
                    'can_override_contract': True,
                    'overrides': ['contractual', 'administrative']
                }
            return {
                'can_override_contract': False,
                'overrides': []
            }
        
        # Contractual and administrative cannot override
        return {
            'can_override_contract': False,
            'overrides': []
        }
    
    def extract_jurisdiction(self, clause_text: str) -> Optional[str]:
        """Extract jurisdiction from clause text."""
        text_lower = clause_text.lower()
        
        jurisdictions = {
            "Saudi Arabia": ["saudi arabia", "saudi", "kingdom of saudi arabia", "ksa"],
            "UAE": ["united arab emirates", "uae", "emirates"],
            "Qatar": ["qatar"],
            "Kuwait": ["kuwait"],
            "Bahrain": ["bahrain"],
            "Oman": ["oman"]
        }
        
        for jurisdiction, patterns in jurisdictions.items():
            if any(pattern in text_lower for pattern in patterns):
                return jurisdiction
        
        return None

