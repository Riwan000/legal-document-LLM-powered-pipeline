"""
Legal taxonomy enforcement service.
Ensures clauses are properly categorized and prevents incompatible category assignments.
"""
import logging
import re
import yaml
from pathlib import Path
from typing import List, Optional, Dict
from backend.models.clause import ClauseType, TerminationSubtype

logger = logging.getLogger(__name__)

# Tie-break priority order for classify_legal_category() (lower index = higher priority)
_LEGAL_CATEGORY_PRIORITY = [
    "governing_law",
    "termination",
    "indemnity",
    "liability",
    "limitation_of_liability",
    "confidentiality",
    "arbitration",
    "dispute_resolution",
    "assignment",
    "payment",
    "compensation_benefits",
    "ip_ownership",
    "sla_obligations",
    "indemnification",
    "compliance",
    "non_compete",
    "non_solicitation",
    "annual_leave",
    "end_of_service_gratuity",
    "probation_period",
    "conduct_discipline",
    "carve_outs",
    "mutual_vs_unilateral",
    "remedies_injunction",
    "return_of_information",
    "remedies",
    "employer_obligations",
    "employee_obligations",
    "other",
]

_CATEGORY_SLUG_KEYWORDS: Dict[str, List[str]] = {
    "assignment": ["assign", "transfer of rights", "novation", "delegate"],
    "indemnity": ["indemnify", "indemnification", "hold harmless", "indemnitor"],
    "governing_law": ["governing law", "applicable law", "law", "statute", "jurisdiction", "legal framework"],
    "termination": ["termination", "terminate", "dismissal", "dismiss", "end of employment", "end of service"],
    "liability": ["liability", "responsible", "responsibility", "damages", "indemnification"],
    "limitation_of_liability": ["limitation of liability", "shall not exceed", "aggregate liability", "maximum liability", "liability cap"],
    "confidentiality": ["confidential", "non-disclosure", "nda", "secrecy", "proprietary information"],
    "arbitration": ["dispute", "arbitration", "mediation", "settlement", "dispute resolution"],
    "dispute_resolution": ["dispute resolution", "arbitration", "mediation", "conciliation", "amicable settlement"],
    "payment": ["salary", "wage", "remuneration", "payment", "pay", "compensation"],
    "compensation_benefits": ["compensation", "benefits", "allowance", "housing", "medical", "insurance", "end of service"],
    "compliance": ["compliance", "regulatory", "emigration", "visa", "work permit", "iqama", "residency"],
    "conduct_discipline": ["discipline", "misconduct", "code of conduct", "disciplinary", "disciplinary action", "conduct", "behavior", "violation", "breach"],
    "employer_obligations": ["employer shall", "employer must", "employer will", "employer obligations", "employer responsibilities"],
    "employee_obligations": ["employee shall", "employee must", "employee will", "employee obligations", "employee responsibilities"],
    "ip_ownership": ["intellectual property", "proprietary rights", "ip rights", "work product", "ip ownership", "vests in", "deliverables"],
    "sla_obligations": ["service level", "sla", "uptime", "availability", "kpi", "performance monitoring", "liquidated damages"],
    "indemnification": ["indemnif", "hold harmless", "indemnitor"],
    "non_compete": ["non-compete", "non compete", "not compete", "restrain from competition"],
    "non_solicitation": ["non solicitation", "not solicit", "solicit employees"],
    "annual_leave": ["annual leave", "paid leave", "vacation", "leave entitlement"],
    "end_of_service_gratuity": ["gratuity", "end of service", "terminal benefits", "severance", "eosb"],
    "probation_period": ["probation", "probationary period", "probationary", "trial period"],
    "carve_outs": ["public domain", "prior knowledge", "independently developed", "carve out", "exclusion", "permitted disclosure"],
    "mutual_vs_unilateral": ["each party", "both parties", "mutual", "reciprocal"],
    "remedies_injunction": ["injunction", "injunctive relief", "specific performance", "equitable relief"],
    "return_of_information": ["return", "destroy", "deletion", "certification of destruction"],
    "remedies": ["remedy", "specific performance", "equitable relief", "damages", "seek relief"],
}

_DEFINITIONS_HEADING_RE = re.compile(r"^definitions?\b", re.IGNORECASE)


def _load_clause_keywords() -> Dict[str, List[str]]:
    """Load clause keyword lists from clause_keywords.yaml at startup."""
    path = Path(__file__).parent.parent / "contract_profiles" / "clause_keywords.yaml"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning(
                "Failed to load clause_keywords.yaml — profile-driven taxonomy disabled: %s", exc
            )
    return {}


_CLAUSE_KEYWORDS: Dict[str, List[str]] = _load_clause_keywords()
if not _CLAUSE_KEYWORDS:
    logger.error(
        "clause_keywords.yaml not loaded — ip_ownership, sla_obligations taxonomy disabled. "
        "Check backend/contract_profiles/clause_keywords.yaml exists and is valid YAML."
    )
else:
    logger.info("clause_keywords.yaml loaded: %d string-type clause keys", len(_CLAUSE_KEYWORDS))


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

        # String-keyed keyword dict for all profile types (driven by YAML).
        # Covers clause types that don't have a ClauseType enum value.
        self.string_type_keywords: Dict[str, List[str]] = _CLAUSE_KEYWORDS

        # Enum-keyed dict kept for backward compat with ClauseType enum values.
        self.category_keywords = {
            ClauseType.GOVERNING_LAW: _CLAUSE_KEYWORDS.get("governing_law", [
                "governing law", "applicable law", "law", "statute",
                "jurisdiction", "legal framework"
            ]),
            ClauseType.REGULATORY_COMPLIANCE: _CLAUSE_KEYWORDS.get("compliance", [
                "compliance", "regulatory", "emigration", "visa",
                "work permit", "iqama", "residency"
            ]),
            ClauseType.TERMINATION: _CLAUSE_KEYWORDS.get("termination", [
                "termination", "terminate", "dismissal", "dismiss",
                "end of employment", "end of service"
            ]),
            ClauseType.COMPENSATION_BENEFITS: _CLAUSE_KEYWORDS.get("compensation_benefits", [
                "compensation", "benefits", "allowance", "housing",
                "medical", "insurance", "end of service"
            ]),
            ClauseType.SALARY_WAGES: _CLAUSE_KEYWORDS.get("salary_wages", [
                "salary", "wage", "remuneration", "payment",
                "pay", "compensation"
            ]),
            ClauseType.EMPLOYER_OBLIGATIONS: _CLAUSE_KEYWORDS.get("employer_obligations", [
                "employer shall", "employer must", "employer will",
                "employer obligations", "employer responsibilities"
            ]),
            ClauseType.EMPLOYEE_OBLIGATIONS: _CLAUSE_KEYWORDS.get("employee_obligations", [
                "employee shall", "employee must", "employee will",
                "employee obligations", "employee responsibilities"
            ]),
            ClauseType.CONDUCT_DISCIPLINE: _CLAUSE_KEYWORDS.get("conduct_discipline", [
                "conduct", "discipline", "behavior", "misconduct",
                "violation", "breach"
            ]),
            ClauseType.JURISDICTION: _CLAUSE_KEYWORDS.get("jurisdiction", [
                "jurisdiction", "courts", "legal venue", "competent court"
            ]),
            ClauseType.DISPUTE_RESOLUTION: _CLAUSE_KEYWORDS.get("dispute_resolution", [
                "dispute", "arbitration", "mediation", "settlement",
                "dispute resolution"
            ]),
            ClauseType.CONFIDENTIALITY: _CLAUSE_KEYWORDS.get("confidentiality", [
                "confidential", "non-disclosure", "nda", "secrecy",
                "proprietary information"
            ]),
            ClauseType.LIABILITY: _CLAUSE_KEYWORDS.get("liability", [
                "liability", "responsible", "responsibility", "damages",
                "indemnification"
            ]),
        }

    def classify_clause_type(self, clause_text: str) -> ClauseType:
        """
        Classify clause into top-level category.

        First checks enum-backed keywords, then falls back to string_type_keywords.
        For non-enum matches, returns ClauseType.OTHER but also sets
        normalized_clause_type on the clause externally if needed.

        Args:
            clause_text: Clause text content

        Returns:
            ClauseType enum value
        """
        text_lower = clause_text.lower()

        # Score enum-backed categories
        category_scores = {}
        for clause_type, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[clause_type] = score

        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]

        # Fall back to string-keyed keywords (covers ip_ownership, sla_obligations, etc.)
        string_scores: Dict[str, int] = {}
        for slug, keywords in self.string_type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                string_scores[slug] = score

        # If a string-type match found, return OTHER (caller can read normalized_clause_type)
        if string_scores:
            return ClauseType.OTHER

        return ClauseType.OTHER

    def classify_clause_type_string(self, clause_text: str) -> str:
        """
        Like classify_clause_type but returns the raw string slug.

        Returns the highest-scoring slug from both enum and string_type_keywords.
        Use this when you need the exact profile clause type (e.g. 'ip_ownership').
        """
        text_lower = clause_text.lower()

        all_scores: Dict[str, int] = {}

        # Enum-backed
        for clause_type, keywords in self.category_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                all_scores[clause_type.value] = score

        # String-keyed (profile types without enum values)
        for slug, keywords in self.string_type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                if slug not in all_scores:   # skip if already scored by enum path
                    all_scores[slug] = score

        if not all_scores:
            return "other"
        return max(all_scores.items(), key=lambda x: x[1])[0]

    def classify_legal_category(self, clause_text: str, clause_heading: str = "") -> Optional[str]:
        """
        Classify a clause into one of the slug categories or None.

        Special case: if clause_heading matches r"^definitions?\\b" (case-insensitive),
        return None — definitions are structural, not functional.

        Args:
            clause_text: Clause verbatim text
            clause_heading: Optional clause heading (checked for "definitions" first)

        Returns:
            Slug string from _LEGAL_CATEGORY_PRIORITY, or None for definitions / no match.
        """
        if clause_heading and _DEFINITIONS_HEADING_RE.match(clause_heading.strip()):
            return None

        text_lower = (clause_text + " " + clause_heading).lower()

        scores: Dict[str, int] = {}
        for slug, keywords in _CATEGORY_SLUG_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > 0:
                scores[slug] = hits

        if not scores:
            return "other"

        # Among categories with equal top score, pick by priority order
        max_score = max(scores.values())
        top_slugs = [s for s in _LEGAL_CATEGORY_PRIORITY if s in scores and scores[s] == max_score]
        if top_slugs:
            return top_slugs[0]
        # Fallback: return highest-scoring slug not in priority list
        return max(scores.items(), key=lambda x: x[1])[0]

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

        return None

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
