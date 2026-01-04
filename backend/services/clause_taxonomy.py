"""
Legal taxonomy enforcement service.
Ensures clauses are properly categorized and prevents incompatible category assignments.
"""
from typing import List, Optional, Dict
from backend.models.clause import ClauseType, TerminationSubtype


class ClauseTaxonomyService:
    """Service for enforcing legal taxonomy rules."""
    
    def __init__(self):
        """Initialize taxonomy service with category rules."""
        # Incompatible category pairs
        self.incompatible_pairs = [
            (ClauseType.CONDUCT_DISCIPLINE, ClauseType.CONFIDENTIALITY),
            (ClauseType.TERMINATION, ClauseType.CONFIDENTIALITY),
            # Add more incompatible pairs as needed
        ]
        
        # Category keywords mapping
        self.category_keywords = {
            ClauseType.GOVERNING_LAW: [
                "governing law", "applicable law", "law", "statute",
                "jurisdiction", "legal framework"
            ],
            ClauseType.REGULATORY_COMPLIANCE: [
                "compliance", "regulatory", "emigration", "visa",
                "work permit", "iqama", "residency"
            ],
            ClauseType.TERMINATION: [
                "termination", "terminate", "dismissal", "dismiss",
                "end of employment", "end of service"
            ],
            ClauseType.COMPENSATION_BENEFITS: [
                "compensation", "benefits", "allowance", "housing",
                "medical", "insurance", "end of service"
            ],
            ClauseType.SALARY_WAGES: [
                "salary", "wage", "remuneration", "payment",
                "pay", "compensation"
            ],
            ClauseType.EMPLOYER_OBLIGATIONS: [
                "employer shall", "employer must", "employer will",
                "employer obligations", "employer responsibilities"
            ],
            ClauseType.EMPLOYEE_OBLIGATIONS: [
                "employee shall", "employee must", "employee will",
                "employee obligations", "employee responsibilities"
            ],
            ClauseType.CONDUCT_DISCIPLINE: [
                "conduct", "discipline", "behavior", "misconduct",
                "violation", "breach"
            ],
            ClauseType.JURISDICTION: [
                "jurisdiction", "courts", "legal venue", "competent court"
            ],
            ClauseType.DISPUTE_RESOLUTION: [
                "dispute", "arbitration", "mediation", "settlement",
                "dispute resolution"
            ],
            ClauseType.CONFIDENTIALITY: [
                "confidential", "non-disclosure", "nda", "secrecy",
                "proprietary information"
            ],
            ClauseType.LIABILITY: [
                "liability", "responsible", "responsibility", "damages",
                "indemnification"
            ]
        }
    
    def classify_clause_type(self, clause_text: str) -> ClauseType:
        """
        Classify clause into top-level category.
        
        Args:
            clause_text: Clause text content
            
        Returns:
            ClauseType enum value
        """
        text_lower = clause_text.lower()
        
        # Score each category
        category_scores = {}
        for clause_type, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[clause_type] = score
        
        if not category_scores:
            return ClauseType.OTHER
        
        # Return highest scoring category
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    def validate_category_compatibility(
        self,
        primary_type: ClauseType,
        secondary_types: List[ClauseType]
    ) -> bool:
        """
        Validate that categories are compatible.
        
        Args:
            primary_type: Primary clause type
            secondary_types: List of secondary types
            
        Returns:
            True if compatible, False otherwise
        """
        all_types = [primary_type] + secondary_types
        
        # Check for incompatible pairs
        for type1, type2 in self.incompatible_pairs:
            if type1 in all_types and type2 in all_types:
                return False
        
        return True
    
    def classify_termination_subtype(self, clause_text: str) -> Optional[TerminationSubtype]:
        """
        Classify termination clause subtype.
        
        Args:
            clause_text: Termination clause text
            
        Returns:
            TerminationSubtype or None if not a termination clause
        """
        text_lower = clause_text.lower()
        
        subtype_keywords = {
            TerminationSubtype.TERMINATION_RIGHTS: [
                "right to terminate", "may terminate", "entitled to terminate"
            ],
            TerminationSubtype.TERMINATION_NOTICE: [
                "notice", "notification", "advance notice", "written notice",
                "days notice", "weeks notice", "months notice"
            ],
            TerminationSubtype.PROBATION_TERMINATION: [
                "probation", "probationary", "trial period", "during probation"
            ],
            TerminationSubtype.END_OF_SERVICE_COMPENSATION: [
                "end of service", "gratuity", "severance", "terminal benefits"
            ],
            TerminationSubtype.DEATH_DISABILITY: [
                "death", "disability", "incapacity", "unable to work"
            ]
        }
        
        for subtype, keywords in subtype_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return subtype
        
        return TerminationSubtype.OTHER
    
    def separate_payment_categories(
        self,
        clause_text: str
    ) -> Dict[str, bool]:
        """
        Determine if clause relates to salary, allowances, or costs.
        
        Args:
            clause_text: Clause text
            
        Returns:
            Dict indicating which categories apply
        """
        text_lower = clause_text.lower()
        
        salary_keywords = ["salary", "wage", "remuneration", "basic pay"]
        allowance_keywords = ["allowance", "housing", "transport", "medical", "benefit"]
        cost_keywords = ["visa", "iqama", "work permit", "travel", "repatriation", "cost"]
        
        return {
            'is_salary': any(kw in text_lower for kw in salary_keywords),
            'is_allowance': any(kw in text_lower for kw in allowance_keywords),
            'is_employer_cost': any(kw in text_lower for kw in cost_keywords),
            'explicitly_provided': True  # Default, can be overridden
        }

