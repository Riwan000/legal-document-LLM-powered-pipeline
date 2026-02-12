"""
Legal hierarchy detection service.

This service provides best-effort heuristic detection of legal hierarchy levels and supremacy
clauses, not legal certainty. It uses keyword matching and pattern recognition to classify
clauses as LAW, CONTRACT, or POLICY, and to identify when one authority overrides another.

IMPORTANT: This service provides heuristic detection only. Outputs may be incorrect for
ambiguous or poorly drafted text. Use as a signal in RAG systems, not as a final legal
decision-maker.
"""
from typing import List, Dict, Any, Optional
from backend.models.document import LegalHierarchyLevel
from backend.config import settings


class LegalHierarchyService:
    """
    Service for detecting and ranking legal hierarchy in clauses.
    
    Provides heuristic keyword-based detection of:
    - Hierarchy levels (LAW > CONTRACT > POLICY)
    - Supremacy clause types (law over contract, contract over policy)
    - Jurisdiction extraction
    - Topic extraction
    
    All methods use best-effort keyword matching and may produce incorrect results
    for ambiguous or poorly drafted text.
    """
    
    def __init__(self):
        """Initialize the legal hierarchy service."""
        self.law_keywords = settings.LEGAL_HIERARCHY_KEYWORDS.get("law", [])
        self.supremacy_keywords = settings.LEGAL_HIERARCHY_KEYWORDS.get("supremacy", [])
        self.contract_keywords = settings.LEGAL_HIERARCHY_KEYWORDS.get("contract", [])
        self.policy_keywords = settings.LEGAL_HIERARCHY_KEYWORDS.get("policy", [])
    
    def detect_hierarchy_level(self, clause_text: str) -> LegalHierarchyLevel:
        """
        Auto-detect hierarchy level from clause text using heuristic keyword matching.
        
        Uses keyword matching to classify clauses as LAW, POLICY, or CONTRACT.
        This is best-effort detection and may be incorrect for ambiguous text.
        
        Detection order: LAW → POLICY → CONTRACT (default)
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            LegalHierarchyLevel enum value (LAW, POLICY, or CONTRACT)
        """
        # Normalize text once for all checks
        text = clause_text.lower()
        
        # Check for law references first (highest priority)
        for keyword in self.law_keywords:
            if keyword.lower() in text:
                return LegalHierarchyLevel.LAW
        
        # Check for policy keywords (company policies, handbooks, internal rules)
        for keyword in self.policy_keywords:
            if keyword.lower() in text:
                return LegalHierarchyLevel.POLICY
        
        # Check for contract keywords
        for keyword in self.contract_keywords:
            if keyword.lower() in text:
                return LegalHierarchyLevel.CONTRACT
        
        # Default to contract if no clear indicators
        return LegalHierarchyLevel.CONTRACT
    
    def supremacy_type(self, clause_text: str) -> Optional[str]:
        """
        Detect supremacy type from clause text using heuristic pattern matching.
        
        Identifies the direction of supremacy: law over contract, or contract over policy.
        This is best-effort detection and may return None for ambiguous text.
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            - "law_over_contract" if law overrides contract
            - "contract_over_policy" if contract overrides policy
            - None if no supremacy pattern detected
        """
        # Normalize text once for all checks
        text = clause_text.lower()
        
        # Check for law over contract patterns
        law_over_patterns = [
            "subject to law",
            "in compliance with law",
            "pursuant to law",
            "in accordance with law",
            "governed by law"
        ]
        
        for pattern in law_over_patterns:
            if pattern in text:
                return "law_over_contract"
        
        # Check for contract over policy patterns
        contract_over_patterns = [
            "notwithstanding this policy",
            "this contract shall prevail over",
            "contract overrides policy",
            "agreement supersedes policy"
        ]
        
        for pattern in contract_over_patterns:
            if pattern in text:
                return "contract_over_policy"
        
        # Check for generic supremacy keywords with context
        for keyword in self.supremacy_keywords:
            if keyword in text:
                # Check if it's followed by law-related terms (law over contract)
                keyword_idx = text.find(keyword)
                if keyword_idx != -1:
                    context = text[max(0, keyword_idx - 50):keyword_idx + 100]
                    if any(law_kw in context for law_kw in self.law_keywords):
                        return "law_over_contract"
                    # Check for policy-related terms (contract over policy)
                    policy_indicators = ["policy", "handbook", "company rules"]
                    if any(policy_kw in context for policy_kw in policy_indicators):
                        return "contract_over_policy"
        
        return None
    
    def is_supremacy_clause(self, clause_text: str) -> bool:
        """
        Detect if clause is a supremacy clause (overrides or is overridden).
        
        This is a convenience wrapper around supremacy_type() that returns a boolean.
        Use supremacy_type() if you need to know the direction of supremacy.
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            True if clause has supremacy implications, False otherwise
        """
        return self.supremacy_type(clause_text) is not None
    
    def extract_jurisdiction(self, clause_text: str) -> Optional[str]:
        """
        Extract jurisdiction from clause text using heuristic keyword matching.
        
        Uses centralized jurisdiction patterns from settings. This is best-effort
        detection and may return None or incorrect results for ambiguous text.
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            Title-cased jurisdiction string (e.g., "Saudi Arabia") or None
        """
        # Normalize text once for all checks
        text = clause_text.lower()
        
        # Use centralized jurisdiction patterns from settings
        for jurisdiction, patterns in settings.JURISDICTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in text:
                    return jurisdiction.title()
        
        return None
    
    def extract_topics(self, clause_text: str) -> List[str]:
        """
        Extract topics/keywords from clause text using heuristic keyword matching.
        
        Uses centralized topic keywords from settings for consistency across services.
        This is best-effort detection and may miss topics or return false positives
        for ambiguous text.
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            List of canonical topic names (e.g., ["termination", "compensation"])
        """
        # Normalize text once for all checks
        text = clause_text.lower()
        topics = []
        
        # Use centralized topic keywords from settings
        for topic, keywords in settings.CLAUSE_TOPIC_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def rank_by_authority(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank clauses by legal authority order only (LAW > CONTRACT > POLICY).
        
        This method provides pure authority-based ordering as a deterministic tie-breaker.
        It does NOT perform scoring, boosting, or similarity ranking. Those responsibilities
        belong to VectorStore and RAG services.
        
        This is a pure sorting function based solely on hierarchy level.
        
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
        
        def get_priority(clause: Dict[str, Any]) -> int:
            """
            Get sorting priority for clause based solely on hierarchy level.
            
            Returns:
                Integer priority (0 = LAW, 1 = CONTRACT, 2 = POLICY)
            """
            # Extract hierarchy level from metadata
            metadata = clause.get('metadata', {})
            hierarchy_str = metadata.get('hierarchy_level', LegalHierarchyLevel.CONTRACT.value)
            
            try:
                hierarchy_level = LegalHierarchyLevel(hierarchy_str)
            except ValueError:
                hierarchy_level = LegalHierarchyLevel.CONTRACT
            
            return hierarchy_order.get(hierarchy_level, 1)
        
        # Sort by priority (pure authority ordering, no scoring)
        sorted_clauses = sorted(clauses, key=get_priority)
        
        return sorted_clauses

