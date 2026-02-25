"""
Query classification service.
Classifies user queries to determine processing requirements.
"""
from typing import List, Dict, Any
from backend.models.document import QueryClassification
from backend.config import settings


class QueryClassifier:
    """Service for classifying user queries."""

    # First words that indicate a binary (yes/no) question
    _BINARY_FIRST_WORDS = frozenset([
        "is", "are", "does", "can", "has", "will", "did", "do", "was", "were"
    ])

    def __init__(self):
        """Initialize the query classifier."""
        # Legal authority keywords (for hierarchy detection)
        self.LEGAL_AUTHORITIES = [
            "law", "act", "statute", "regulation",
            "contract", "policy", "company policy",
            "agreement", "labor law", "saudi labor law"
        ]
        
        # Hierarchy conflict patterns (explicit conflict language)
        self.HIERARCHY_PATTERNS = [
            "override", "supersede", "prevail",
            "which applies", "takes precedence",
            "law vs", "contract vs", "conflict between"
        ]
        
        # Hard out-of-scope patterns (immediate exclusion)
        self.OUT_OF_SCOPE_PATTERNS = [
            "immigration", "visa", "tax filing",
            "criminal case", "court judgment", "vision 2030",
            "court precedent", "case law", "judicial",
            "court decision", "legal opinion", "attorney", "lawyer advice",
            "residency permit"
        ]
        
        # Topic synonyms for retrieval filtering
        # scope_topics are used for retrieval filtering, not intent classification
        self.TOPIC_SYNONYMS = {
            "termination": ["terminate", "dismiss", "end employment", "cancel contract", "fire"],
            "notice": ["notice", "inform", "intimation", "advance warning", "notification"],
            "probation": ["probation", "trial period"],
            "compensation": ["salary", "wage", "pay", "remuneration", "compensation"],
            "benefits": ["allowance", "insurance", "medical", "housing", "benefit"]
        }
        
        # Required context map (topic → required context fields)
        self.REQUIRED_CONTEXT = {
            "termination": ["jurisdiction", "contract_type"],
            "compensation": ["jurisdiction"],
            "probation": ["jurisdiction"]
        }
        
        # Legacy keywords (kept for backward compatibility)
        self.legal_query_keywords = [
            "legal", "law", "statute", "regulation", "compliance", "governed by",
            "legality", "legal requirement", "legal obligation"
        ]
        
        self.hierarchy_query_keywords = [
            "override", "supersede", "prevail", "precedence", "hierarchy",
            "which takes precedence", "which applies", "law vs contract"
        ]
        
        # Query type patterns (for intent detection)
        # query_types represent user intent, used for answer generation
        self.query_type_patterns = {
            "termination": ["terminate", "termination", "dismiss", "dismissal", "end employment", "fire"],
            "legality": ["legal", "lawful", "legal requirement", "compliance", "governed by"],
            "benefits": ["benefit", "allowance", "housing", "medical", "insurance", "compensation"],
            "compensation": ["compensation", "salary", "wage", "payment", "pay", "remuneration"],
            "compliance": ["compliance", "comply", "requirement", "obligation", "must", "shall"],
            "notice": ["notice", "notification", "advance notice", "written notice"],
            "probation": ["probation", "probationary", "trial period", "trial"],
            # Phase 3 — definition lookup intent
            "definition_lookup": [
                "what does", "definition of", "defined as", "what is", "meaning of",
                "define", "what are", "means",
            ],
            # Summary intent — overview/high-level questions
            "summary": [
                "summarize", "summary", "overview", "what is this about",
                "what does this contract cover", "give me a summary",
                "briefly describe", "outline", "high-level",
            ],
            # Classification intent — document type identification
            "classification": [
                "what type of", "what kind of contract", "what kind of agreement",
                "is this a", "classify", "identify the document type",
                "what is this contract", "what agreement is this",
            ],
            # Clause lookup intent — direct structural clause reference
            "clause_lookup": [
                "what does clause", "what does article", "what does section",
                "clause number", "article number", "show me clause",
                "content of clause", "what is in clause",
            ],
        }
        
        # Normalization map for word variations
        self._normalization_map = {
            # Verb forms
            "terminated": "terminate",
            "dismissed": "dismiss",
            "cancelled": "cancel",
            # Noun forms
            "termination": "terminate",
            "dismissal": "dismiss",
            "cancellation": "cancel",
            # Plural forms
            "benefits": "benefit",
            "wages": "wage",
            "salaries": "salary"
        }
    
    def _normalize(self, text: str) -> str:
        """
        Normalize text by replacing word variations with base forms.
        
        Handles:
        - Verb forms: terminated → terminate
        - Noun forms: termination → terminate
        - Plural forms: benefits → benefit
        """
        normalized = text
        for variant, base in self._normalization_map.items():
            normalized = normalized.replace(variant, base)
        return normalized
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify user query to determine processing requirements.
        
        Args:
            query: User query string
            
        Returns:
            QueryClassification object with query_types (intent), scope_topics (retrieval filters),
            risk_level, and clarification flags
        """
        # Normalize query first (handles word variations)
        query_normalized = self._normalize(query.lower())
        
        # Detect query types (intent - what user wants to know)
        query_types = self._detect_query_types(query_normalized)
        
        # Check if requires legal hierarchy
        requires_legal_hierarchy = self._requires_hierarchy_check(query_normalized)
        
        # Extract scope topics (retrieval filters - what to search for)
        scope_topics = self._extract_scope_topics(query_normalized)
        
        # Check if legal query (requires citations)
        is_legal_query = self._is_legal_query(query_normalized)
        
        # Calculate risk level
        risk_level = self._risk_level(query_types)
        
        # Detect missing context
        missing_context = self._missing_context(query_types, query_normalized)
        requires_clarification = len(missing_context) > 0
        
        return QueryClassification(
            query_types=query_types,
            requires_legal_hierarchy=requires_legal_hierarchy,
            scope_topics=scope_topics,
            is_legal_query=is_legal_query,
            risk_level=risk_level,
            requires_clarification=requires_clarification,
            missing_context=missing_context
        )
    
    def _detect_query_types(self, query_lower: str) -> List[str]:
        """
        Detect all query intent types (multi-label detection).

        query_types represent user intent - what they want to know.
        Returns list of all matching types, not just the first match.
        """
        detected = []

        # Binary (yes/no) question detection: first word is a question auxiliary + ends with "?"
        first_word = query_lower.split()[0] if query_lower.split() else ""
        if first_word in self._BINARY_FIRST_WORDS and query_lower.rstrip().endswith("?"):
            detected.append("binary")

        # Check each query type pattern
        for query_type, patterns in self.query_type_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected.append(query_type)

        # Default to ["general"] if no specific type detected
        return detected if detected else ["general"]
    
    def _requires_hierarchy_check(self, query_lower: str) -> bool:
        """
        Determine if query requires legal hierarchy analysis.
        
        Hierarchy should be detected only if:
        - Multiple authorities are mentioned (>= 2), OR
        - Explicit conflict language exists
        
        Legal queries without multiple authorities or conflict language do NOT require hierarchy check.
        """
        # Count authority mentions
        authority_mentions = sum(
            1 for authority in self.LEGAL_AUTHORITIES 
            if authority in query_lower
        )
        
        # Check for explicit conflict phrases
        explicit_conflict = any(
            pattern in query_lower for pattern in self.HIERARCHY_PATTERNS
        )
        
        # Require hierarchy check only if multiple authorities OR explicit conflict
        return authority_mentions >= 2 or explicit_conflict
    
    def _extract_scope_topics(self, query_lower: str) -> List[str]:
        """
        Extract scope topics from query for retrieval filtering.
        
        scope_topics are used for retrieval filtering (what to search for in documents),
        not intent classification. They help identify which document sections to search.
        Uses synonym mapping instead of literal matching.
        """
        topics = []
        
        # Check topic synonyms (not literal matches)
        for topic, synonyms in self.TOPIC_SYNONYMS.items():
            if any(synonym in query_lower for synonym in synonyms):
                if topic not in topics:
                    topics.append(topic)
        
        # Also check against covered topics from config
        for topic in settings.COVERED_TOPICS:
            topic_lower = topic.lower()
            if topic_lower in query_lower:
                # Normalize topic name
                normalized_topic = topic_lower
                if normalized_topic not in topics:
                    topics.append(normalized_topic)
        
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
    
    def _missing_context(self, query_types: List[str], query_lower: str) -> List[str]:
        """
        Detect missing context required for the query.
        
        Args:
            query_types: List of detected query intent types
            query_lower: Normalized query text
            
        Returns:
            List of missing context fields (e.g., "jurisdiction", "contract_type")
        """
        missing = []
        
        # Check for jurisdiction
        jurisdiction_keywords = ["jurisdiction", "saudi", "saudi arabia", "ksa", "country", "law"]
        if not any(kw in query_lower for kw in jurisdiction_keywords):
            # Check if any query type requires jurisdiction
            for query_type in query_types:
                if query_type in self.REQUIRED_CONTEXT:
                    if "jurisdiction" in self.REQUIRED_CONTEXT[query_type]:
                        missing.append("jurisdiction")
                        break
        
        # Check for contract type (if termination-related)
        if "termination" in query_types:
            contract_type_keywords = ["contract type", "contract", "agreement type", "employment type"]
            if not any(kw in query_lower for kw in contract_type_keywords):
                if "contract_type" in self.REQUIRED_CONTEXT.get("termination", []):
                    missing.append("contract_type")
        
        return missing
    
    def _risk_level(self, query_types: List[str]) -> str:
        """
        Calculate risk level for the query.
        
        Risk level controls answer strictness, citation requirements, and disclaimers.
        
        Args:
            query_types: List of detected query intent types
            
        Returns:
            "high", "medium", or "low"
        """
        # High risk: legal/compliance queries
        if "legality" in query_types or "compliance" in query_types:
            return "high"
        
        # Medium risk: termination, compensation queries
        if any(t in query_types for t in ["termination", "compensation"]):
            return "medium"
        
        # Low risk: everything else
        return "low"
    
    def is_out_of_scope(self, query: str, covered_topics: List[str] = None) -> bool:
        """
        Check if query is out of scope.
        
        Bias toward answering, not rejecting. Only hard exclusions are enforced.
        
        Args:
            query: User query
            covered_topics: List of topics covered by documents (defaults to config)
            
        Returns:
            True if query is out of scope (hard exclusion only)
        """
        if covered_topics is None:
            covered_topics = settings.COVERED_TOPICS
        
        query_lower = query.lower()
        
        # Step 1: Hard out-of-scope only (immediate exclusion)
        if any(pattern in query_lower for pattern in self.OUT_OF_SCOPE_PATTERNS):
            return True
        
        # Step 2: Soft relevance check (bias toward answering)
        classification = self.classify_query(query)
        
        # If legal query and has recognizable employment intent → in scope
        if classification.is_legal_query and classification.scope_topics:
            return False
        
        # Default: allow query (bias toward answering)
        return False

