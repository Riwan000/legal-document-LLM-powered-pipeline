"""
Contract Review workflow service.

Executes the Contract Review workflow using WorkflowContext: load profile,
gather clauses (from store or extraction), identify missing clauses and risks,
build evidence blocks and executive summary. No raw dict returns; all results
go into ctx.intermediate_results. Failures set ctx.error and ctx.status="failed".
"""
from __future__ import annotations

import logging
import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Tuple, Literal

from backend.config import settings
from backend.models.workflow import WorkflowContext
from backend.workflow_stages import StageStatus
from backend.models.contract_review import (
    ContractReviewResponse,
    RiskItem,
    VerbatimSnippet,
    ClauseEvidenceBlock,
    ExecutiveSummaryItem,
)
from backend.services.contract_profile_loader import load_contract_profile
from backend.services.clause_store import ClauseStore
from backend.services.clause_extraction import ClauseExtractionService
from backend.services.guardrails import enforce_non_prescriptive_language
from backend.utils.file_parser import FileParser


_log = logging.getLogger(__name__)

STEP_NAME = "contract_review"
MIN_SNIPPET_LENGTH = 120
MIN_ALPHA_RATIO = 0.5
TERMINATION_TOKEN_DISTANCE = 1

EXPECTED_CLAUSE_PATTERNS: Dict[str, List[str]] = {
    # Employment profile
    "termination": [
        "terminate",
        "terminated",
        "termination",
        "may be terminated",
        "this agreement may be terminated",
        "termination of this agreement",
    ],
    # 1b: widened
    "notice": [
        "notice period", "written notice", "days notice", "days of notice",
        "notice of termination", "give notice", "provide notice",
    ],
    # 1b: widened
    "salary_wages": [
        "salary", "wages", "remuneration", "pay", "base salary",
        "basic salary", "monthly pay", "hourly rate",
    ],
    # 1a: removed "indemnity" and "blood money" (cross-match fix)
    "compensation": ["compensation", "remuneration", "salary package", "total package"],
    "benefits": ["benefit", "allowance", "insurance"],
    # 1b: widened
    "governing_law": [
        "governing law", "governed by", "shall be governed by",
        "laws of", "subject to the laws", "in accordance with the laws",
        "applicable law",
    ],
    "jurisdiction": ["jurisdiction", "courts of"],
    # 1b: widened
    "conduct_discipline": [
        "discipline", "misconduct", "code of conduct",
        "disciplinary", "disciplinary action",
    ],
    # NDA profile
    # 1b: widened; "non disclosure" = normalized form of "non-disclosure"
    "confidentiality": [
        "confidential", "confidentiality", "non disclosure",
        "confidential information", "proprietary information", "trade secret",
        "duty of confidentiality",               # additional synonyms
        "obligation of confidentiality",
    ],
    "term": ["term", "duration", "effective date"],
    # 1a: removed "indemnify" (cross-match fix)
    "liability": ["liability", "limitation of liability", "liable", "exempt from liability"],
    # 1b: widened
    "remedies": [
        "remedy", "injunctive relief", "specific performance",
        "equitable relief", "damages", "seek relief",
    ],
    # MSA profile
    # 1b: widened
    "dispute_resolution": [
        "dispute resolution", "arbitration", "arbitral",
        "mediation", "conciliation", "settlement of disputes",
        "resolution of disputes",               # word-order variant
        "disputes and arbitration",
        "resolution of disputes and arbitration",
        "governing disputes",
        "amicable settlement",
    ],
    # ── 1c: new entries (RC1 fix) ─────────────────────────────────────────────
    "non_solicitation": [
        "non solicitation", "not solicit", "solicit employees",
        "solicit customers", "non compete",
    ],
    "indemnification": [
        "indemnif",        # prefix-matches: indemnify, indemnification, indemnified…
        "hold harmless",
        "indemnitor",
    ],
    "ip_ownership": [
        "intellectual property", "proprietary rights", "ip rights",
        "work product", "ownership of", "assigns all right",
        "vests in", "shall own", "deliverables", "work for hire",
        "background ip", "foreground ip", "moral rights", "ip ownership",
        "intellectual property rights",
        "ownership of deliverables",             # additional synonyms
        "client shall own",
        "assigned to client",
        "ip indemnification",
    ],
    "sla_obligations": [
        "service level", "sla", "schedule a",
        "service standards", "uptime", "availability", "response time",
        "liquidated damages", "performance monitoring", "business continuity",
        "key performance", "kpi", "service credits", "penalty",
        "sla schedule",
        "sla timelines",                          # additional synonyms
        "service schedule",
        "annexure a",
        "timelines and sla",
        "performance standards",
    ],
    "limitation_of_liability": [
        "limitation of liability", "shall not exceed",
        "aggregate liability", "maximum liability", "liability cap",
    ],
    # ── retained from contract-review-improvements.md ─────────────────────────
    "probation_period":        ["probation", "trial period", "probationary period"],
    "end_of_service_gratuity": ["gratuity", "end of service", "terminal benefits", "severance", "eosb", "end of service benefit", "service benefit"],
    "non_compete":             ["non-compete", "non compete", "not compete", "restrain from competition", "competitive activity"],
    "annual_leave":            ["annual leave", "paid leave", "vacation", "leave entitlement"],
    "carve_outs":              ["excluding", "except for", "public domain", "prior knowledge", "independently developed", "rightfully obtained", "carve out", "exclusion", "permitted disclosure", "disclosure permitted", "exception to confidentiality"],
    "mutual_vs_unilateral":    ["each party", "both parties", "mutual", "reciprocal"],
    "remedies_injunction":     ["injunction", "injunctive relief", "specific performance", "equitable relief"],
    "return_of_information":   ["return", "destroy", "deletion", "certification of destruction"],
}

# Clause heading synonyms: alternative section headings mapped to canonical clause types.
# Used by G7 post-pass to catch clauses whose headings don't match primary keyword set.
CLAUSE_HEADING_SYNONYMS: Dict[str, str] = {
    "proprietary rights":                        "ip_ownership",
    "intellectual property":                     "ip_ownership",
    "resolution of disputes":                    "dispute_resolution",
    "disputes and arbitration":                  "dispute_resolution",
    "resolution of disputes and arbitration":    "dispute_resolution",
    "arbitration clause":                        "dispute_resolution",
    "sla timelines":                             "sla_obligations",
    "sla and timelines":                         "sla_obligations",
    "service level":                             "sla_obligations",
    "performance monitoring":                    "sla_obligations",
    "liquidated damages":                        "sla_obligations",
    "business continuity plan":                  "sla_obligations",
    "schedule a":                                "sla_obligations",
    "confidentiality clause":                    "confidentiality",
    "non disclosure":                            "confidentiality",
    "duty of confidentiality":                   "confidentiality",
    "obligation of confidentiality":             "confidentiality",
    # Employment headings
    "remuneration":                              "salary_wages",
    "leave entitlement":                         "annual_leave",
    "probationary period":                       "probation_period",
    "gratuity":                                  "end_of_service_gratuity",
    "end of service benefit":                    "end_of_service_gratuity",
    "notice period":                             "notice",
    "salary and wages":                          "salary_wages",
    "compensation and benefits":                 "compensation",
    "conduct and discipline":                    "conduct_discipline",
    # NDA headings
    "confidentiality agreement":                 "confidentiality",
    "non-disclosure agreement":                  "confidentiality",
    "confidentiality and non-disclosure":        "confidentiality",
    "term of agreement":                         "term",
    "remedies for breach":                       "remedies",
    "injunctive relief":                         "remedies_injunction",
    "return of confidential information":        "return_of_information",
    "carve-outs":                                "carve_outs",
    "mutual confidentiality":                    "mutual_vs_unilateral",
}

# Human-readable display names for clause IDs (semantic slug -> display name).
# Used at response-build time only; storage unchanged.
# Profile clause key -> human-readable display name (for not_detected list and observations).
EXPECTED_CLAUSE_DISPLAY_NAMES: Dict[str, str] = {
    "termination": "Termination",
    "notice": "Notice Period",
    "salary_wages": "Salary and Wages",
    "compensation": "Compensation",
    "benefits": "Benefits",
    "governing_law": "Governing Law",
    "jurisdiction": "Jurisdiction",
    "conduct_discipline": "Conduct and Discipline",
    "confidentiality": "Confidentiality",
    "term": "Term",
    "liability": "Liability",
    "remedies": "Remedies",
    "dispute_resolution": "Dispute Resolution",
    "probation": "Probation Period",       # updated per contract_fixes1
    "non_solicitation": "Non-Solicitation",  # new per contract_fixes1
    # ── Improvement 3 — new clause display names ──────────────────────────────
    "probation_period":        "Probation Period",
    "end_of_service_gratuity": "End-of-Service Gratuity",
    "non_compete":             "Non-Compete",
    "ip_ownership":            "IP Ownership",
    "annual_leave":            "Annual Leave",
    "carve_outs":              "Confidentiality Carve-Outs",
    "mutual_vs_unilateral":    "Mutual vs Unilateral Confidentiality",
    "remedies_injunction":     "Injunctive Relief Remedies",
    "return_of_information":   "Return / Destruction of Information",
    "indemnification":         "Indemnification",
    "limitation_of_liability": "Limitation of Liability",
    "sla_obligations":         "SLA Obligations",
}

WEAK_EVIDENCE_RULES: Dict[str, List[str]] = {
    "confidentiality": ["duration"],
    "ip_ownership": ["post_employment_scope"],
    "termination": ["notice_period"],
    "non_compete": ["duration", "geographic_scope"],
    # Governing law: presence of jurisdiction text is usually sufficient; no weak-evidence checks.
    "governing_law": [],
}

CANONICAL_CLAUSE_GROUPS: Dict[str, List[str]] = {
    # Compensation group: treat compensation / salary_wages / remuneration as one reviewer-facing concept.
    "compensation": ["compensation", "salary_wages", "remuneration"],
}

# Implicit governing law: contract references an applicable legal regime without a standalone "Governing Law" clause.
# Do not require country/jurisdiction phrase; do not infer country.
IMPLICIT_GOV_LAW_KEYWORDS: List[str] = [
    "labour law",
    "labor law",
    "workman law",
    "law in force",
    "laws in force",
    "laws applicable in",
    "as per the law",
    "as per the law of",
    "as per the provisions of",
    "under the provisions of",
    "according to article",
    "determined by the law",
]
# Implicit dispute resolution (G1.5 post-pass): dispute-resolution language that doesn't use
# the primary keyword set but signals a dispute mechanism exists.
IMPLICIT_DISPUTE_RESOLUTION_KEYWORDS: List[str] = [
    "resolution of disputes",
    "amicable settlement",
    "conciliation",
    "arbitration tribunal",
    "settlement of disputes",
    "disputes shall be",
    "disputes arising",
    "governing disputes",
]

IMPLICIT_IP_OWNERSHIP_KEYWORDS: List[str] = [
    "proprietary rights",
    "intellectual property rights",
    "vests in",
    "client shall own",
    "assigned to client",
    "work for hire",
    "deliverables belong",
    "ip ownership",
]

IMPLICIT_SLA_KEYWORDS: List[str] = [
    "liquidated damages",
    "performance monitoring",
    "business continuity",
    "schedule a",
    "sla schedule",
    "kpi",
    "service credits",
    "penalty clause",
]

# Country/jurisdiction phrases used only to require a geographic reference alongside a keyword (no inference).
GOV_LAW_COUNTRY_REFERENCES: List[str] = [
    "saudi arabia",
    "kingdom of saudi arabia",
    "ksa",
    "uae",
    "emirates",
    "gcc",
    "bahrain",
    "kuwait",
    "oman",
    "qatar",
]

# Implicit coverage: clause types that can be "implicitly_covered" when evidence suggests related obligations elsewhere.
# e.g. conduct_discipline when termination text mentions discipline/misconduct/breach.
IMPLICIT_COVERAGE_RULES: Dict[str, Dict[str, List[str]]] = {
    "conduct_discipline": {
        "primary": ["terminate", "termination", "terminated"],
        "secondary": ["discipline", "misconduct", "breach", "code of conduct"],
        "coverage_note": "Disciplinary logic appears within termination provisions and statutory references.",
    },
}

# Benefits: composite detection across evidence (medical, housing, meals, transport, leave, indemnity).
# Spec-aligned: single-word and phrase signals for GCC/distributed benefit phrasing.
BENEFITS_SIGNALS: Dict[str, List[str]] = {
    "medical": ["medical", "treatment", "health", "medical treatment", "medical care"],
    "housing": ["accommodation", "housing", "lodging"],
    "meals": ["meals", "food", "food allowance"],
    "transport": ["transport", "transportation", "airfare", "visa", "visa cost"],
    "leave": ["leave", "vacation", "annual leave"],
    "indemnity": ["compensation", "blood money", "death"],
}

DISPLAY_NAME_MAP: Dict[str, str] = {
    "confidentialinformation": "Confidentiality",
    "confidentialinformat": "Confidentiality",
    "governinglaw": "Governing Law",
    "intellectualproperty": "Intellectual Property",
    "positionandduties": "Position and Duties",
    "compensation": "Compensation",
    "termination": "Termination",
    "notice": "Notice Period",
    "salarywages": "Salary and Wages",
    "benefits": "Benefits",
    "jurisdiction": "Jurisdiction",
    "conductdiscipline": "Conduct and Discipline",
    "disputeresolution": "Dispute Resolution",
    "liability": "Liability",
    "remedies": "Remedies",
    "term": "Term",
}


def _evidence_block_display_name(
    semantic_label: Optional[str],
    clause_type_key: Optional[str],
    page_number: int,
) -> str:
    """
    Sanitized display name for evidence block: never use raw OCR/title.
    Priority: semantic_label -> clause-type display name -> "Clause (Page N)".
    """
    if semantic_label and semantic_label.strip():
        return semantic_label.strip()
    if clause_type_key:
        display = EXPECTED_CLAUSE_DISPLAY_NAMES.get(
            clause_type_key, clause_type_key.replace("_", " ").title()
        )
        if display and display.strip():
            return display.strip()
    return f"Clause (Page {page_number})"


def _resolve_clause_display_name(clause_id: str) -> str:
    """
    Resolve a clause_id to a human-readable display name for UI.
    Parses semantic part from IDs like unknown_ConfidentialInformat_6bf9001f.
    Fallback: title-case with spaces (no raw hash or unknown_).
    """
    if not clause_id or not clause_id.strip():
        return "Clause"
    parts = clause_id.strip().split("_")
    # Format: section_slug_heading_slug_hash -> take middle segment (heading_slug)
    if len(parts) >= 3:
        semantic = " ".join(parts[1:-1])   # all middle segments, skip section prefix + hash suffix
    elif len(parts) == 2:
        semantic = parts[1]
    else:
        semantic = parts[0]
    key = re.sub(r"[^a-z0-9]", "", semantic.lower())
    if key in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[key]
    # Fallback: insert space before capitals, then title-case first letter of each word
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", semantic)
    if cleaned:
        return cleaned.strip()
    return "Clause"


def _find_document_path(document_id: str) -> Optional[str]:
    """Resolve document file path from document_id."""
    # Exact match first
    for ext in [".pdf", ".docx", ".doc"]:
        p = settings.DOCUMENTS_PATH / f"{document_id}{ext}"
        if p.exists():
            return str(p)
    # Prefix match: handles versioned filenames like <id>_v1.pdf
    for ext in [".pdf", ".docx", ".doc"]:
        matches = list(settings.DOCUMENTS_PATH.glob(f"{document_id}*{ext}"))
        if matches:
            return str(sorted(matches)[-1])  # pick latest version
    return None


# Document classification: heuristic to detect non-operative (e.g. academic) documents.
DOC_CLASSIFY_OPERATIVE_PHRASES = (
    "agreement", "employer", "employee", "shall pay", "shall terminate",
    "service provider", "disclosing party", "receiving party", "vendor", "contractor",
)
# Party terms that confirm an operative contract preamble (covers employment, NDA, MSA).
_DOC_CLASSIFY_PARTY_TERMS = frozenset({
    "employer", "employee",
    "disclosing party", "receiving party",
    "service provider", "client",
    "vendor", "contractor",
    "licensor", "licensee",
    "the company", "the consultant",
})
DOC_CLASSIFY_ACADEMIC_PATTERNS = [
    re.compile(r"\bv\.\s", re.IGNORECASE),  # case citation
    re.compile(r"\bF\.\s*2d\b", re.IGNORECASE),
    re.compile(r"\bU\.\s*S\.\b", re.IGNORECASE),
    re.compile(r"§\s*\d+", re.IGNORECASE),
    re.compile(r"\bthe court held\b", re.IGNORECASE),
    re.compile(r"\baccording to\b", re.IGNORECASE),
    re.compile(r"\bfootnote\s+\d+", re.IGNORECASE),
    re.compile(r"\b\d+\s+F\.\s*Supp", re.IGNORECASE),
]
DOC_CLASSIFY_ACADEMIC_DENSITY_THRESHOLD = 0.012  # hits per word above this -> academic
DOC_CLASSIFY_LEAD_CHARS = 5000  # check parties/agreement in first N chars


def _is_likely_operative_contract(full_doc_text: str) -> bool:
    """
    Rule-based heuristic: True if document looks like an operative contract (no warning).
    False triggers the document_classification_warning banner.
    """
    if not (full_doc_text or "").strip():
        return True  # no text -> do not warn
    text = full_doc_text
    t_lower = text.lower()
    lead = text[:DOC_CLASSIFY_LEAD_CHARS].lower()

    # No parties / no agreement in lead
    if "agreement" not in lead and "contract" not in lead:
        return False
    if not any(term in lead for term in _DOC_CLASSIFY_PARTY_TERMS):
        return False

    # Must have at least one operative phrase
    if not any(p in t_lower for p in DOC_CLASSIFY_OPERATIVE_PHRASES):
        return False

    # High density of academic/case-law signals
    word_count = max(1, len(t_lower.split()))
    hits = sum(1 for pat in DOC_CLASSIFY_ACADEMIC_PATTERNS if pat.search(text))
    if (hits / word_count) >= DOC_CLASSIFY_ACADEMIC_DENSITY_THRESHOLD:
        return False

    return True


def _normalize_clause_type_for_profile(t: str) -> str:
    """Normalize clause type string to match profile keys (lowercase, underscores)."""
    t = (t or "").strip().lower().replace(" ", "_")
    # ClauseStore historically used str(enum) which can become "clausetype.xxx"
    if "." in t and t.startswith("clausetype."):
        t = t.split(".", 1)[1]
    return t


def _severity_rank(sev: str) -> int:
    sev = (sev or "").lower()
    if sev == "high":
        return 0
    if sev == "medium":
        return 1
    return 2


def _display_status_for_internal(status: str) -> str:
    """Map internal status to UI display status (FIX 4)."""
    if status == "detected":
        return "Detected"
    if status == "uncertain":
        return "Detected (Weak Evidence)"
    return "Not Detected"


CRITICAL_CLAUSES = {"termination", "governing_law", "compensation"}


def _build_observation_item(
    clause_key: str,
    display_name: str,
    status: str,
    base_severity: str,
    document_confidence_high: bool = False,
    coverage_note: Optional[str] = None,
    observation_text_override: Optional[str] = None,
) -> ExecutiveSummaryItem:
    """
    Build a single observation item with tuned severity:
    - Missing critical clause: High only if document_confidence_high, else Medium
    - Ambiguous termination/notice: Medium
    - Implicit/distributed: cap at Medium
    - Present & standard (detected): no label (severity=None)
    """
    sev = (base_severity or "medium").lower()
    if sev not in ("high", "medium", "low"):
        sev = "medium"

    # Severity tuning based on status and criticality
    if status in ("detected_implicit", "detected_distributed"):
        final_sev = "medium"  # not displayed; severity=None and no tag for these statuses
    elif status == "detected_weak":
        final_sev = "medium"  # never High for implicit/distributed
    elif status == "not_detected" and clause_key in CRITICAL_CLAUSES:
        final_sev = "high" if document_confidence_high else "medium"
    elif status == "uncertain" and clause_key in {"termination", "intellectual_property"}:
        final_sev = "medium"  # ambiguous notice/termination
    elif status == "uncertain":
        final_sev = "medium"
    elif status == "detected":
        final_sev = "low"
    else:
        final_sev = sev

    # No severity badge or tag for implicit governing law, distributed benefits, or implicitly_covered (spec)
    tag = ""
    if status not in ("detected", "detected_implicit", "detected_distributed", "implicitly_covered"):
        tag = f"[{final_sev.capitalize()}] "

    if status == "detected":
        text = f"A {display_name} clause is present."
    elif status == "uncertain":
        if clause_key == "notice":
            text = "A Notice Period clause is present; commonly expected details are not clearly specified."
        else:
            text = (
                f"A {display_name} clause is present; commonly expected details "
                f"(e.g. duration, scope) are not clearly specified."
            )
    elif status == "detected_implicit":
        text = observation_text_override or "A Governing Law reference is present implicitly through citation to local labor law."
    elif status == "detected_distributed":
        text = "Benefits provisions appear to be present across multiple clauses."
    elif status == "detected_weak":
        text = "Benefits appear in limited form (single category detected)."
    elif status == "implicitly_covered":
        text = "No standalone clause detected. Related obligations appear implicitly across other provisions."
        if coverage_note:
            text += f" {coverage_note}"
    else:
        text = f"No {display_name} clause was detected."

    out_severity: Optional[str] = None
    if status not in ("detected", "detected_implicit", "detected_distributed", "implicitly_covered"):
        out_severity = final_sev

    category: Optional[Literal["risk", "finding", "confirmation"]] = None
    if status == "not_detected":
        category = "risk"
    elif status in ("uncertain", "detected_weak", "detected_implicit", "detected_distributed", "implicitly_covered"):
        category = "finding"
    elif status == "detected":
        category = "confirmation"

    return ExecutiveSummaryItem(
        text=(tag + text).strip(),
        severity=out_severity,
        category=category,
        related_risk_indices=[],
    )


def _build_key_review_observations(
    presence_map: Dict[str, Any],
    risk_weights: Dict[str, Any],
    expected: List[str],
    document_confidence_high: bool = False,
    jurisdiction: Optional[str] = None,
) -> List[ExecutiveSummaryItem]:
    """
    Build Key Review Observations from presence_map (template-based, no LLM).
    Order: (1) Ambiguities / Weak Evidence, (2) Missing Clauses, (3) Present & Standard;
    within each group, higher severity first. High severity only when critical missing and document_confidence_high.
    """
    items: List[ExecutiveSummaryItem] = []

    # First, build grouped view for canonical clause groups (e.g. compensation vs salary_wages).
    handled: set[str] = set()

    def _aggregate_status(member_keys: List[str]) -> str:
        statuses = [presence_map.get(k, {}).get("status", "not_detected") for k in member_keys]
        if any(s == "detected" for s in statuses):
            return "detected"
        if any(s == "detected_implicit" for s in statuses):
            return "detected_implicit"
        if any(s == "detected_distributed" for s in statuses):
            return "detected_distributed"
        if any(s == "detected_weak" for s in statuses):
            return "detected_weak"
        if any(s == "implicitly_covered" for s in statuses):
            return "implicitly_covered"
        if any(s == "uncertain" for s in statuses):
            return "uncertain"
        return "not_detected"

    def _max_severity(member_keys: List[str]) -> str:
        sev_values = []
        for k in member_keys:
            sev = (risk_weights.get(k) or "medium")
            if isinstance(sev, str):
                sev = sev.lower()
            if sev not in ("high", "medium", "low"):
                sev = "medium"
            sev_values.append(sev)
        if not sev_values:
            return "medium"
        # choose highest (high < medium < low in rank)
        return min(sev_values, key=_severity_rank)

    # Canonical groups
    for canon_key, members in CANONICAL_CLAUSE_GROUPS.items():
        member_keys = [m for m in members if m in expected]
        if not member_keys:
            continue
        status = _aggregate_status(member_keys)
        severity = _max_severity(member_keys)
        display_name = EXPECTED_CLAUSE_DISPLAY_NAMES.get(canon_key, canon_key.replace("_", " ").title())
        handled.update(member_keys)
        coverage_note_canon = None
        if status == "implicitly_covered":
            for m in member_keys:
                if presence_map.get(m, {}).get("coverage_note"):
                    coverage_note_canon = presence_map[m]["coverage_note"]
                    break
        items.append(
            _build_observation_item(
                clause_key=canon_key,
                display_name=display_name,
                status=status,
                base_severity=severity,
                document_confidence_high=document_confidence_high,
                coverage_note=coverage_note_canon,
            )
        )

    # Standalone expected keys not covered by any canonical group
    for clause_type in expected:
        if clause_type in handled:
            continue
        entry = presence_map.get(clause_type, {})
        status = entry.get("status", "not_detected")
        display_name = EXPECTED_CLAUSE_DISPLAY_NAMES.get(clause_type, clause_type.replace("_", " ").title())
        base_severity = (risk_weights.get(clause_type) or "medium")
        if isinstance(base_severity, str):
            base_severity = base_severity.lower()
        if base_severity not in ("high", "medium", "low"):
            base_severity = "medium"
        obs_override: Optional[str] = None
        if clause_type == "governing_law" and status == "detected_implicit":
            if not (jurisdiction or "").strip():
                obs_override = "Reference to labor law provisions detected; jurisdiction not specified."
            elif not entry.get("implicit_evidence"):
                obs_override = "A possible implicit governing law reference was observed through statutory citations; no explicit governing law clause was detected."
        # ── Improvement 3 — jurisdiction override for end_of_service_gratuity ──
        if clause_type == "end_of_service_gratuity" and status == "not_detected":
            if (jurisdiction or "").strip() in ("KSA", "UAE", "Generic GCC"):
                base_severity = "high"
        items.append(
            _build_observation_item(
                clause_key=clause_type,
                display_name=display_name,
                status=status,
                base_severity=base_severity,
                document_confidence_high=document_confidence_high,
                coverage_note=entry.get("coverage_note"),
                observation_text_override=obs_override,
            )
        )

    # Sort: risk → finding → confirmation, then severity within group
    def _order_key(item: ExecutiveSummaryItem) -> tuple:
        cat_rank = {"risk": 0, "finding": 1, "confirmation": 2}.get(item.category or "", 1)
        sev_rank = _severity_rank(item.severity or "low")
        return (cat_rank, sev_rank, (item.text or "").lower())
    items.sort(key=_order_key)
    return items


def _clause_type_aliases(profile_key: str) -> List[str]:
    """
    Expand profile clause keys into aliases we consider equivalent for extraction.

    This helps reconcile differences between the profile taxonomy and ClauseType/section labels.
    """
    k = _normalize_clause_type_for_profile(profile_key)
    aliases = {k}
    # Employment profiles frequently distinguish these as separate, while structured clauses
    # often use a combined "compensation_benefits" category.
    if k in ("compensation", "benefits"):
        aliases.add("compensation_benefits")
    # Some extractors label notice under termination umbrella
    if k == "notice":
        aliases.add("termination")
        aliases.add("termination_notice")
    return sorted(aliases)


def _canon_jurisdiction(j: Optional[str]) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Return (canonical_code, positive_keywords, negative_keywords_for_mismatch_scan).

    positive_keywords: evidence tokens that indicate the expected jurisdiction.
    negative_keywords: other jurisdictions to scan for; if found alongside absence of positive keywords,
    we flag a potential mismatch.
    """
    if not j:
        return None, [], []
    jl = j.strip().lower()

    # Minimal, extensible keyword sets. Keep deterministic and non-interpretive.
    if any(tok in jl for tok in ["ksa", "saudi", "saudi arabia", "kingdom of saudi"]):
        return (
            "KSA",
            ["ksa", "saudi", "saudi arabia", "kingdom of saudi", "saudi labor law", "saudi labour law"],
            ["uae", "united arab emirates", "dubai", "abu dhabi", "qatar", "bahrain", "oman", "kuwait"],
        )
    if any(tok in jl for tok in ["uae", "emirates", "united arab emirates", "dubai", "abu dhabi"]):
        return (
            "UAE",
            ["uae", "united arab emirates", "emirates", "dubai", "abu dhabi"],
            ["ksa", "saudi", "saudi arabia", "qatar", "bahrain", "oman", "kuwait"],
        )
    if "qatar" in jl:
        return ("Qatar", ["qatar"], ["ksa", "saudi", "uae", "united arab emirates", "dubai", "abu dhabi"])
    if "bahrain" in jl:
        return ("Bahrain", ["bahrain"], ["ksa", "saudi", "uae", "united arab emirates", "dubai", "abu dhabi"])
    if "oman" in jl:
        return ("Oman", ["oman"], ["ksa", "saudi", "uae", "united arab emirates", "dubai", "abu dhabi"])
    if "kuwait" in jl:
        return ("Kuwait", ["kuwait"], ["ksa", "saudi", "uae", "united arab emirates", "dubai", "abu dhabi"])

    # Default: keep the raw jurisdiction, but with no keyword-based checks.
    return j.strip(), [], []


def _find_hits(patterns: List[re.Pattern], text: str) -> List[str]:
    hits: List[str] = []
    for p in patterns:
        if p.search(text):
            hits.append(p.pattern)
    return hits


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return alpha / max(len(text), 1)


OCR_NOISE_THRESHOLD = 0.35  # Above this (1 - alpha_ratio) -> treat as OCR-noisy for structure classification


def _classify_structure_confidence(
    heading: str,
    text: str,
    ocr_noise_score: float,
) -> Literal["clause", "provision", "section_non_standard"]:
    """
    Deterministic structure confidence for evidence block (display-only).
    Clause = clear heading + contract language + low OCR noise; Provision = contract language only; else section_non_standard.
    """
    has_heading = bool((heading or "").strip())
    t_lower = (text or "").lower()
    contract_language = ["shall", "may", "agrees", "terminate", "governed by"]
    has_contract_language = any(v in t_lower for v in contract_language)
    if has_heading and has_contract_language and ocr_noise_score < OCR_NOISE_THRESHOLD:
        return "clause"
    if has_contract_language:
        return "provision"
    return "section_non_standard"


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _looks_like_terminate(token: str) -> bool:
    if abs(len(token) - len("terminate")) > 2:
        return False
    return _levenshtein_distance(token, "terminate") <= TERMINATION_TOKEN_DISTANCE


# Clause types for which non-contractual evidence must downgrade to not_detected.
DOWNGRADE_IF_NON_CONTRACTUAL = frozenset({"compensation", "salary_wages", "termination", "notice"})
NON_CONTRACTUAL_MAX_HEADER_LEN = 80
NON_CONTRACTUAL_HEADER_PATTERNS = (
    "introduction", "table of contents", "appendix", "index", "references",
    "abstract", "executive summary", "preface",
)


def _is_non_contractual_snippet(text: str) -> bool:
    """
    True if the snippet looks like a section header or non-contractual text:
    section header (very short or header-like), and/or lacks obligation verbs and parties.
    """
    if not (text or "").strip():
        return True
    t = text.strip()
    t_lower = t.lower()
    # Section header: very short or matches header patterns
    is_short = len(t) < NON_CONTRACTUAL_MAX_HEADER_LEN
    is_header_pattern = any(p in t_lower for p in NON_CONTRACTUAL_HEADER_PATTERNS)
    if is_short and is_header_pattern:
        return True
    # Single line, mostly title-case, short -> likely header
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if is_short and len(lines) <= 2 and not any(c in t for c in ".?!"):
        return True
    # Lacks obligation verbs and parties -> non-contractual
    has_obligation = any(v in t_lower for v in ("shall", "must", "agrees", "agree"))
    has_parties = any(
        p in t_lower for p in ("employer", "employee", "party", "parties", "company", "contractor")
    )
    if not has_obligation and not has_parties:
        return True
    return False


def _classify_evidence_label(text: str) -> Optional[str]:
    """
    G3: Deterministic semantic label for evidence block (display-only).
    Returns None when no confident label; no legal interpretation.
    """
    if not (text or "").strip():
        return None
    t = (text or "").lower()
    if "death" in t and "compensation" in t:
        return "Death Compensation / Indemnity"
    if "blood money" in t:
        return "Death Compensation / Indemnity"
    if "terminate" in t or "notice" in t:
        return "Termination & Notice"
    return None


def _detect_clause_presence(
    evidence_candidates: List[Dict[str, Any]],
    keywords: List[str],
    clause_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evidence-first presence detection using conservative substring checks.

    Returns:
      {
        "status": "detected" | "not_detected" | "uncertain",
        "clause_id": Optional[str],
        "page_number": Optional[int],
        "matched_keyword": Optional[str],
      }
    """
    for cand in evidence_candidates:
        raw_text = cand.get("text") or ""
        normalized_text = _normalize_text(raw_text)
        if not normalized_text:
            continue
        tokens = normalized_text.split()

        def _downgrade_if_non_contractual() -> bool:
            if clause_type not in DOWNGRADE_IF_NON_CONTRACTUAL:
                return False
            return cand.get("is_non_contractual") is True

        # Hard rule for termination detection
        if clause_type == "termination":
            if "this agreement" in normalized_text and any(_looks_like_terminate(t) for t in tokens):
                if _downgrade_if_non_contractual():
                    continue
                return {
                    "status": "detected",
                    "clause_id": cand.get("clause_id"),
                    "page_number": cand.get("page_number"),
                    "matched_keyword": "this agreement + terminate*",
                }
        for kw in keywords:
            normalized_kw = _normalize_text(kw)  # Fix RC2: normalize before match (handles hyphens etc.)
            # Primary: substring match
            kw_matched = normalized_kw and normalized_kw in normalized_text
            # Fallback: word-set match for multi-word keywords (handles word-order variants)
            if not kw_matched and normalized_kw and " " in normalized_kw:
                kw_words = set(normalized_kw.split())
                text_words = set(normalized_text.split())
                kw_matched = kw_words <= text_words
            if kw_matched:
                if _downgrade_if_non_contractual():
                    continue
                is_weak = len(raw_text) < MIN_SNIPPET_LENGTH or _alpha_ratio(raw_text) < MIN_ALPHA_RATIO
                if len(normalized_text) > 200:
                    is_weak = False
                # Heading-only evidence: short text with no obligation verbs is a heading
                # match, not an uncertain detection — treat as detected_implicit to avoid
                # false risk items for legitimately-present clauses.
                if is_weak and not any(v in normalized_text for v in ("shall", "must", "agree", "terminate")):
                    return {
                        "status": "detected_implicit",
                        "clause_id": cand.get("clause_id"),
                        "page_number": cand.get("page_number"),
                        "matched_keyword": kw,
                    }
                rules = WEAK_EVIDENCE_RULES.get(clause_type or "", None)
                if rules == []:
                    status = "detected"
                else:
                    status = "uncertain" if is_weak else "detected"
                return {
                    "status": status,
                    "clause_id": cand.get("clause_id"),
                    "page_number": cand.get("page_number"),
                    "matched_keyword": kw,
                }

        # OCR-tolerant termination token scan (keyword-independent)
        if clause_type == "termination" and any(_looks_like_terminate(t) for t in tokens):
            if _downgrade_if_non_contractual():
                continue
            is_weak = len(raw_text) < MIN_SNIPPET_LENGTH or _alpha_ratio(raw_text) < MIN_ALPHA_RATIO
            if len(normalized_text) > 200:
                is_weak = False
            rules = WEAK_EVIDENCE_RULES.get("termination", None)
            if rules == []:
                status = "detected"
            else:
                status = "uncertain" if is_weak else "detected"
            return {
                "status": status,
                "clause_id": cand.get("clause_id"),
                "page_number": cand.get("page_number"),
                "matched_keyword": "terminate*",
            }
    return {
        "status": "not_detected",
        "clause_id": None,
        "page_number": None,
        "matched_keyword": None,
    }


def _page_fallback_scan(
    document_id: str,
    missing_clause_types: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Last-resort scan: parses raw document pages to find keywords for clauses
    that remain not_detected after all structured and implicit post-passes.
    Catches clauses in Schedules/Annexures excluded from structured extraction.
    Returns {} silently on any error (non-fatal).
    """
    if not missing_clause_types:
        return {}
    file_path = _find_document_path(document_id)
    if not file_path:
        _log.warning("[ContractReview] _page_fallback_scan: document not found for id=%s", document_id)
        return {}
    try:
        pages = FileParser.parse_file(Path(file_path))
    except Exception as exc:
        _log.warning(
            "[ContractReview] _page_fallback_scan: FileParser failed for %s: %s",
            file_path, exc,
        )
        return {}
    if not pages:
        _log.warning(
            "[ContractReview] _page_fallback_scan: no pages parsed for %s", file_path
        )
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    for clause_type in missing_clause_types:
        if clause_type in results:
            continue
        keywords = EXPECTED_CLAUSE_PATTERNS.get(clause_type, [])
        if not keywords:
            continue
        for page_entry in pages:
            if len(page_entry) == 3:
                page_text, page_num, _ = page_entry
            else:
                page_text, page_num = page_entry
            normalized_page = _normalize_text(page_text)
            if not normalized_page:
                continue
            for kw in keywords:
                normalized_kw = _normalize_text(kw)
                if normalized_kw and normalized_kw in normalized_page:
                    results[clause_type] = {
                        "status": "detected_implicit",
                        "page_number": page_num,
                    }
                    break
            if clause_type in results:
                break
    return results


class ContractReviewService:
    """
    Contract Review workflow: profile-driven risk identification and
    evidence-backed deliverable. Every step accepts/returns WorkflowContext.
    """

    def __init__(
        self,
        clause_store: ClauseStore,
        clause_extractor: ClauseExtractionService,
    ):
        self.clause_store = clause_store
        self.clause_extractor = clause_extractor
        # ── Improvement 2 — load explanation templates once at init ───────────
        self.risk_explanations: Dict[str, Any] = self._load_risk_explanations()
        # ── Group 7 — jurisdiction statutory notes ────────────────────────────
        self.jurisdiction_statutes: Dict[str, Any] = self._load_jurisdiction_statutes()
        # ── Group 6 — Ollama client for G8 LLM uncertain-clause resolver ──────
        try:
            import ollama as _ollama
            self._ollama_client = _ollama.Client(host=settings.OLLAMA_BASE_URL)
        except Exception:
            self._ollama_client = None

        # Deterministic heuristic patterns for "problematic language detection".
        # Keep these non-prescriptive: we flag ambiguity/breadth and reference evidence.
        self._ambiguous_patterns = [
            re.compile(r"\breasonable\b", re.IGNORECASE),
            re.compile(r"\bfrom time to time\b", re.IGNORECASE),
            re.compile(r"\bat (its|their) discretion\b", re.IGNORECASE),
            re.compile(r"\bsole discretion\b", re.IGNORECASE),
            re.compile(r"\bas (it|they) determine(s)?\b", re.IGNORECASE),
            re.compile(r"\bincluding but not limited to\b", re.IGNORECASE),
            re.compile(r"\bwithout limitation\b", re.IGNORECASE),
            re.compile(r"\bmaterial\b", re.IGNORECASE),
            re.compile(r"\bsubstantial\b", re.IGNORECASE),
        ]
        self._termination_patterns = [
            re.compile(r"\bwithout notice\b", re.IGNORECASE),
            re.compile(r"\bwith(out)? cause\b", re.IGNORECASE),
            re.compile(r"\bfor any reason\b", re.IGNORECASE),
            re.compile(r"\bat any time\b", re.IGNORECASE),
            re.compile(r"\bimmediate(ly)?\b", re.IGNORECASE),
        ]
        self._confidentiality_patterns = [
            re.compile(r"\bin perpetuity\b", re.IGNORECASE),
            re.compile(r"\bforever\b", re.IGNORECASE),
            re.compile(r"\bany and all\b", re.IGNORECASE),
        ]
        self._liability_patterns = [
            re.compile(r"\bunlimited\b", re.IGNORECASE),
            re.compile(r"\bindemnif(y|ies|ication)\b", re.IGNORECASE),
            re.compile(r"\bconsequential damages\b", re.IGNORECASE),
            re.compile(r"\bpunitive damages\b", re.IGNORECASE),
        ]

    # ── Improvement 2 — risk explanation templates ────────────────────────────

    def _load_risk_explanations(self) -> Dict[str, Any]:
        """Load risk_explanations.yaml once at service init. Returns empty dict on missing file."""
        path = Path(__file__).parent.parent / "contract_profiles" / "risk_explanations.yaml"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as exc:
                logging.warning("Failed to load risk_explanations.yaml: %s", exc)
        return {}

    def _load_jurisdiction_statutes(self) -> Dict[str, Any]:
        """Load jurisdiction_statutes.yaml once at service init. Returns empty dict on missing file."""
        path = Path(__file__).parent.parent / "legal_references" / "jurisdiction_statutes.yaml"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as exc:
                logging.warning("Failed to load jurisdiction_statutes.yaml: %s", exc)
        return {}

    def _resolve_uncertain_with_llm(self, clause_type: str, evidence_text: str, display_name: str) -> str:
        """
        Binary LLM confirmation for a clause stuck at status='uncertain'.

        Returns 'detected', 'not_detected', or 'uncertain' (on failure).
        Wraps all errors; never raises.
        """
        if not self._ollama_client:
            return "uncertain"
        try:
            prompt = (
                f"Does the following text contain a {display_name} clause? "
                f"Reply only: YES or NO.\n\n{evidence_text[:600]}"
            )
            response = self._ollama_client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={"temperature": 0.0, "seed": settings.CONTRACT_REVIEW_SEED},
            )
            answer = (response.get("response") or "").strip().lower()
            return "detected" if answer.startswith("yes") else "not_detected"
        except Exception as exc:
            _log.debug("G8 LLM uncertain resolver failed for %s: %s", clause_type, exc)
            return "uncertain"

    def _identify_contradiction_risks(
        self,
        presence_map: Dict[str, Any],
        evidence_candidates: List[Dict[str, Any]],
    ) -> List[RiskItem]:
        """
        Rule-based cross-clause contradiction detection. No LLM.

        Checks:
        1. Notice period mismatch: termination vs notice clause numeric values differ >50%.
        2. Confidentiality perpetuity vs return_of_information co-existence.

        Returns a list of low-severity RiskItems (may be empty).
        """
        results: List[RiskItem] = []

        def _extract_duration(text: str) -> Optional[int]:
            """Extract first duration value in days from text."""
            m = re.search(r"(\d+)\s*(day|month|year)s?", text, re.IGNORECASE)
            if not m:
                return None
            value, unit = int(m.group(1)), m.group(2).lower()
            if unit.startswith("month"):
                return value * 30
            if unit.startswith("year"):
                return value * 365
            return value

        def _evidence_text_for(clause_type: str) -> str:
            ids = (presence_map.get(clause_type) or {}).get("clause_ids") or []
            for cand in evidence_candidates:
                if cand.get("clause_id") in ids:
                    return cand.get("text", "")
            return ""

        # Check 1: notice period mismatch between termination and notice clauses
        term_status = (presence_map.get("termination") or {}).get("status", "")
        notice_status = (presence_map.get("notice") or {}).get("status", "")
        if term_status == "detected" and notice_status == "detected":
            term_text = _evidence_text_for("termination")
            notice_text = _evidence_text_for("notice")
            term_days = _extract_duration(term_text)
            notice_days = _extract_duration(notice_text)
            if term_days and notice_days and term_days != notice_days:
                ratio = max(term_days, notice_days) / min(term_days, notice_days)
                if ratio > 1.5:
                    desc = enforce_non_prescriptive_language(
                        f"Possible notice period inconsistency: termination clause references "
                        f"{term_days} days while notice clause references {notice_days} days.",
                        step=f"{STEP_NAME}.contradiction",
                    )
                    results.append(RiskItem(
                        description=desc,
                        severity="low",
                        status="detected",
                        clause_types=["termination", "notice"],
                        missing_clause=False,
                    ))

        # Check 2: perpetual confidentiality vs return_of_information
        conf_status = (presence_map.get("confidentiality") or {}).get("status", "")
        ret_status = (presence_map.get("return_of_information") or {}).get("status", "")
        if conf_status == "detected" and ret_status == "detected":
            conf_text = _evidence_text_for("confidentiality").lower()
            if "perpetuity" in conf_text or "forever" in conf_text:
                desc = enforce_non_prescriptive_language(
                    "Confidentiality obligation extends in perpetuity while a return/destruction "
                    "of information clause is also present; consider verifying these are consistent.",
                    step=f"{STEP_NAME}.contradiction",
                )
                results.append(RiskItem(
                    description=desc,
                    severity="low",
                    status="detected",
                    clause_types=["confidentiality", "return_of_information"],
                    missing_clause=False,
                ))

        return results

    def _resolve_problematic_key(self, description: str, clause_type: str) -> str:
        """Map problematic language risk descriptions to explanation template keys."""
        desc_lower = (description or "").lower()
        if "without notice" in desc_lower:
            return "problematic_termination_no_notice"
        if "unlimited" in desc_lower:
            return "problematic_liability_unlimited"
        if "in perpetuity" in desc_lower or "forever" in desc_lower:
            return "problematic_confidentiality_perpetuity"
        return clause_type

    def _attach_explanations(self, risks: List[RiskItem]) -> List[RiskItem]:
        """
        Attach severity_reason and recommendation from static templates.
        Keyed on (clause_type, status, severity). No-op if no entry found (graceful fallback).
        Does NOT run enforce_non_prescriptive_language on these fields.
        """
        if not self.risk_explanations:
            return risks
        for risk in risks:
            for clause_type in (risk.clause_types or []):
                # For problematic language risks (missing_clause=False), map to specific key
                if risk.missing_clause:
                    explanation_key = clause_type
                else:
                    explanation_key = self._resolve_problematic_key(risk.description, clause_type)
                entry = (
                    self.risk_explanations
                    .get(explanation_key, {})
                    .get(risk.status or "detected", {})
                    .get(risk.severity, {})
                )
                if entry:
                    risk.severity_reason = entry.get("reason")
                    risk.recommendation = entry.get("recommendation")
                break  # use first clause_type match only
        return risks

    # ── Improvement 1 — verbatim evidence per risk ────────────────────────────

    def _attach_verbatim_evidence(
        self,
        risks: List[RiskItem],
        evidence_blocks: List[ClauseEvidenceBlock],
        max_snippets_per_risk: int = 3,
        clause_matched_keywords: Optional[Dict[str, str]] = None,
    ) -> List[RiskItem]:
        """
        For each RiskItem, match clause_ids against evidence_blocks and attach up to
        max_snippets_per_risk VerbatimSnippet objects.
        Skips risks where status is not_detected or missing_clause=True.
        Uncertain risks retain their clause_ids and receive verbatim snippets.

        clause_matched_keywords: optional map of clause_type -> matched keyword string,
        used to populate VerbatimSnippet.matched_keyword.
        """
        block_by_id: Dict[str, ClauseEvidenceBlock] = {b.clause_id: b for b in evidence_blocks}
        _kw_map = clause_matched_keywords or {}
        for risk in risks:
            if risk.status == "not_detected" or risk.missing_clause:
                continue  # no verbatim text for absent clauses
            # Determine matched keyword for this risk's clause types
            matched_kw: Optional[str] = None
            for ct in risk.clause_types:
                if ct in _kw_map:
                    matched_kw = _kw_map[ct]
                    break
            snippets: List[VerbatimSnippet] = []
            for cid in risk.clause_ids[:max_snippets_per_risk]:
                block = block_by_id.get(cid)
                if block and block.clean_text and not block.is_non_contractual:
                    snippets.append(VerbatimSnippet(
                        clause_id=cid,
                        display_name=block.display_name,
                        page_number=block.page_number,
                        text=block.clean_text[:400],
                        matched_keyword=matched_kw,
                    ))
            risk.verbatim_evidence = snippets
        return risks

    def _identify_problematic_language_risks(
        self,
        *,
        clauses_from_store: list,
        profile_risk_weights: Dict[str, Any],
        review_depth: str,
    ) -> List[RiskItem]:
        """
        Deterministic heuristic risk identification.

        Flags:
        - Ambiguous/broad language (generic)
        - Clause-type specific triggers (termination/confidentiality/liability)

        Output is evidence-linked: clause_ids + page_numbers.
        """
        if not clauses_from_store:
            return []

        depth = (review_depth or "standard").lower()
        include_generic_ambiguity = depth != "quick"

        risks: List[RiskItem] = []
        seen: set = set()

        for c in clauses_from_store:
            clause_id = getattr(c, "clause_id", None) or ""
            evs = getattr(c, "evidence", None) or []
            if not evs:
                continue

            # Aggregate text for scanning. Use clean_text where possible.
            combined = "\n".join([(getattr(ev, "clean_text", None) or getattr(ev, "raw_text", "") or "") for ev in evs]).lower()

            raw_type = getattr(c, "normalized_clause_type", None) or getattr(getattr(c, "type", None), "value", None) or str(getattr(c, "type", "") or "")
            clause_type = _normalize_clause_type_for_profile(raw_type)
            clause_title = (getattr(c, "title", None) or "").strip()

            page_numbers = sorted({int(getattr(ev, "page", 0) or 0) for ev in evs if getattr(ev, "page", None) is not None})

            # 1) Clause-type specific patterns
            if clause_type in ("termination", "termination_notice", "notice"):
                hits = _find_hits(self._termination_patterns, combined)
                if hits:
                    sev = (profile_risk_weights.get("termination") or "medium")
                    sev = sev.lower() if isinstance(sev, str) else "medium"
                    desc = (
                        "Termination-related language includes potentially broad or time-sensitive terms "
                        f"(e.g., {', '.join(sorted(set(hits))[:3])})."
                    )
                    key = ("termination_terms", clause_id, desc)
                    if key not in seen:
                        seen.add(key)
                        risks.append(
                            RiskItem(
                                description=desc,
                                severity=sev if sev in ("high", "medium", "low") else "medium",
                                clause_types=["termination"],
                                missing_clause=False,
                                clause_ids=[clause_id] if clause_id else [],
                                page_numbers=page_numbers,
                                display_names=[_resolve_clause_display_name(clause_id)] if clause_id else [],
                            )
                        )

            if clause_type in ("confidentiality",):
                hits = _find_hits(self._confidentiality_patterns, combined)
                if hits:
                    sev = (profile_risk_weights.get("confidentiality") or "medium")
                    sev = sev.lower() if isinstance(sev, str) else "medium"
                    desc = (
                        "Confidentiality language includes potentially broad duration/scope terms "
                        f"(e.g., {', '.join(sorted(set(hits))[:3])})."
                    )
                    key = ("confidentiality_terms", clause_id, desc)
                    if key not in seen:
                        seen.add(key)
                        risks.append(
                            RiskItem(
                                description=desc,
                                severity=sev if sev in ("high", "medium", "low") else "medium",
                                clause_types=["confidentiality"],
                                missing_clause=False,
                                clause_ids=[clause_id] if clause_id else [],
                                page_numbers=page_numbers,
                                display_names=[_resolve_clause_display_name(clause_id)] if clause_id else [],
                            )
                        )

            if clause_type in ("liability", "limitation_of_liability"):
                hits = _find_hits(self._liability_patterns, combined)
                if hits:
                    sev = (profile_risk_weights.get("liability") or "high")
                    sev = sev.lower() if isinstance(sev, str) else "high"
                    desc = (
                        "Liability-related language includes potentially broad allocation terms "
                        f"(e.g., {', '.join(sorted(set(hits))[:3])})."
                    )
                    key = ("liability_terms", clause_id, desc)
                    if key not in seen:
                        seen.add(key)
                        risks.append(
                            RiskItem(
                                description=desc,
                                severity=sev if sev in ("high", "medium", "low") else "high",
                                clause_types=["liability"],
                                missing_clause=False,
                                clause_ids=[clause_id] if clause_id else [],
                                page_numbers=page_numbers,
                                display_names=[_resolve_clause_display_name(clause_id)] if clause_id else [],
                            )
                        )

            # 2) Generic ambiguity patterns (standard depth only)
            if include_generic_ambiguity:
                hits = _find_hits(self._ambiguous_patterns, combined)
                if hits:
                    # Base severity from clause type weight if present; else low.
                    sev = (profile_risk_weights.get(clause_type) or "low")
                    sev = sev.lower() if isinstance(sev, str) else "low"
                    if sev not in ("high", "medium", "low"):
                        sev = "low"
                    desc = (
                        f"Clause language includes potentially ambiguous or broad terms (e.g., {', '.join(sorted(set(hits))[:3])})."
                    )
                    key = ("ambiguity_terms", clause_id, desc)
                    if key not in seen:
                        seen.add(key)
                        risks.append(
                            RiskItem(
                                description=desc,
                                severity=sev,
                                clause_types=[clause_type] if clause_type else [],
                                missing_clause=False,
                                clause_ids=[clause_id] if clause_id else [],
                                page_numbers=page_numbers,
                                display_names=[_resolve_clause_display_name(clause_id)] if clause_id else [],
                            )
                        )

        return risks

    def _identify_jurisdiction_risks(
        self,
        *,
        clauses_from_store: list,
        jurisdiction: Optional[str],
        profile_risk_weights: Dict[str, Any],
    ) -> List[RiskItem]:
        """
        Jurisdiction-aware rule checks (deterministic, evidence-based).

        This does not provide legal interpretation. It flags:
        - governing law / jurisdiction clause does not explicitly reference the requested jurisdiction keywords
        - presence of other jurisdiction keywords when expected jurisdiction keywords are absent (potential mismatch)
        """
        canon, positive, negative = _canon_jurisdiction(jurisdiction)
        if not canon or not positive:
            return []

        # Focus only on governing_law / jurisdiction / dispute_resolution clauses.
        relevant_types = {"governing_law", "jurisdiction", "dispute_resolution"}
        candidate_clauses = []
        for c in clauses_from_store or []:
            raw_type = getattr(c, "normalized_clause_type", None) or getattr(getattr(c, "type", None), "value", None) or str(getattr(c, "type", "") or "")
            clause_type = _normalize_clause_type_for_profile(raw_type)
            if clause_type in relevant_types:
                candidate_clauses.append(c)

        if not candidate_clauses:
            return []

        combined_text = ""
        clause_ids: List[str] = []
        page_numbers: List[int] = []
        for c in candidate_clauses:
            clause_ids.append(getattr(c, "clause_id", "") or "")
            for ev in (getattr(c, "evidence", None) or []):
                t = (getattr(ev, "clean_text", None) or getattr(ev, "raw_text", "") or "")
                combined_text += "\n" + t
                if getattr(ev, "page", None) is not None:
                    page_numbers.append(int(getattr(ev, "page", 0) or 0))

        t_low = combined_text.lower()
        has_expected = any(k in t_low for k in positive)
        has_other = any(k in t_low for k in negative)

        risks: List[RiskItem] = []
        if not has_expected:
            sev = (profile_risk_weights.get("governing_law") or "medium")
            sev = sev.lower() if isinstance(sev, str) else "medium"
            if sev not in ("high", "medium", "low"):
                sev = "medium"
            desc = f"Governing law / jurisdiction references for '{canon}' were not explicitly found in the provided evidence."
            cids = [cid for cid in clause_ids if cid]
            risks.append(
                RiskItem(
                    description=desc,
                    severity=sev,
                    clause_types=["governing_law", "jurisdiction"],
                    missing_clause=False,
                    clause_ids=cids,
                    page_numbers=sorted({p for p in page_numbers if p}),
                    display_names=[_resolve_clause_display_name(cid) for cid in cids],
                )
            )
        elif has_other:
            # Expected present, but other jurisdictions also appear; flag as a consistency check.
            desc = (
                f"Evidence includes references consistent with '{canon}' and also mentions other jurisdictions; "
                "this may require consistency verification."
            )
            cids = [cid for cid in clause_ids if cid]
            risks.append(
                RiskItem(
                    description=desc,
                    severity="low",
                    clause_types=["governing_law", "jurisdiction"],
                    missing_clause=False,
                    clause_ids=cids,
                    page_numbers=sorted({p for p in page_numbers if p}),
                    display_names=[_resolve_clause_display_name(cid) for cid in cids],
                )
            )

        return risks

    def run(self, ctx: WorkflowContext) -> WorkflowContext:
        """
        Execute the full Contract Review workflow. Reads contract_id, contract_type,
        jurisdiction, review_depth from ctx.document_ids[0] and ctx.metadata.
        """
        if ctx.status == "failed":
            return ctx

        contract_id = (ctx.document_ids or [None])[0]
        if not contract_id:
            ctx.workflow_state.legal_analysis = StageStatus.FAILED
            ctx.fail(
                code="MISSING_INPUT",
                message="Contract Review requires document_ids[0] (contract_id).",
                step=STEP_NAME,
                details={},
            )
            return ctx

        meta = ctx.metadata or {}
        contract_type_key = meta.get("contract_type")
        if not contract_type_key:
            _log.warning(
                "No contract_type in metadata for %s — defaulting to 'employment'", contract_id
            )
            contract_type_key = "employment"
        jurisdiction = ctx.jurisdiction or meta.get("jurisdiction")
        review_depth = meta.get("review_depth") or "standard"

        # 1) Load contract profile
        ctx.current_step = f"{STEP_NAME}.load_profile"
        try:
            profile = load_contract_profile(contract_type_key)
        except FileNotFoundError as e:
            ctx.workflow_state.legal_analysis = StageStatus.FAILED
            ctx.fail(
                code="PROFILE_NOT_FOUND",
                message=f"Contract profile not found for type '{contract_type_key}'.",
                step=STEP_NAME,
                details={"contract_type": contract_type_key, "path": str(e)},
            )
            return ctx
        except ValueError as e:
            ctx.workflow_state.legal_analysis = StageStatus.FAILED
            ctx.fail(
                code="INVALID_PROFILE",
                message=f"Invalid contract profile: {e}",
                step=STEP_NAME,
                details={"contract_type": contract_type_key},
            )
            return ctx
        except Exception as e:
            ctx.workflow_state.legal_analysis = StageStatus.FAILED
            ctx.fail(
                code="PROFILE_LOAD_ERROR",
                message=f"Failed to load contract profile: {e}",
                step=STEP_NAME,
                details={"contract_type": contract_type_key},
            )
            return ctx

        ctx.add_result(f"{STEP_NAME}.profile", profile)

        # 1b) Document classification (mandatory check before review)
        document_classification_warning: Optional[str] = None
        doc_path = _find_document_path(contract_id)
        if doc_path:
            try:
                pages = FileParser.parse_file(Path(doc_path))
                full_text = "\n".join((t or "") for t, _ in pages)
                if not _is_likely_operative_contract(full_text):
                    document_classification_warning = (
                        f"This document does not appear to be an operative {contract_type_key} contract. "
                        "Results are based on the selected contract profile and may reflect profile mismatch rather than missing clauses."
                    )
            except Exception:
                pass  # do not fail workflow; leave warning unset

        # 2) Get clauses (from store or extract)
        ctx.current_step = f"{STEP_NAME}.clauses"
        clauses_from_store = self.clause_store.get_clauses_by_document(contract_id)
        evidence_candidates: List[Dict[str, Any]] = []
        if clauses_from_store:
            # Use structured clauses; types from enum value / normalized_clause_type
            extracted_types = set()
            evidence_blocks: List[ClauseEvidenceBlock] = []
            for c in clauses_from_store:
                # Prefer enum values (canonical), fall back to normalized strings.
                norm = getattr(getattr(c, "type", None), "value", None) or getattr(c, "normalized_clause_type", None) or str(getattr(c, "type", "") or "")
                norm = _normalize_clause_type_for_profile(norm)
                extracted_types.add(norm)
                # Add aliases to improve profile matching.
                for alias in _clause_type_aliases(norm):
                    extracted_types.add(alias)
                for ev in (c.evidence or []):
                    ev_text = ev.clean_text or ev.raw_text or ""
                    is_nc = _is_non_contractual_snippet(ev_text)
                    heading = (getattr(c, "title", None) or "").strip() or _resolve_clause_display_name(c.clause_id)
                    raw = getattr(ev, "raw_text", None) or ""
                    ocr_noise_score = 1.0 - _alpha_ratio(raw) if raw else 0.0
                    structure_class = _classify_structure_confidence(heading, ev_text, ocr_noise_score)
                    sem_label = _classify_evidence_label(ev_text)
                    page_num = ev.page or 0
                    evidence_blocks.append(
                        ClauseEvidenceBlock(
                            clause_id=c.clause_id,
                            page_number=page_num,
                            raw_text=ev.raw_text or "",
                            clean_text=ev.clean_text or ev.raw_text or "",
                            display_name=_evidence_block_display_name(sem_label, norm, page_num),
                            is_non_contractual=is_nc,
                            semantic_label=sem_label,
                            structure_class=structure_class,
                        )
                    )
                    # Fix 4: When heading is empty, use canonical display name so keyword
                    # matching can pick up clause type even for heading-only store entries.
                    if not heading:
                        norm_type = _normalize_clause_type_for_profile(norm)
                        heading = EXPECTED_CLAUSE_DISPLAY_NAMES.get(norm_type, "")
                    evidence_candidates.append(
                        {
                            "clause_id": c.clause_id,
                            "page_number": ev.page or 0,
                            "text": (heading + " " + ev_text).strip() if (heading or ev_text) else "",
                            "is_non_contractual": is_nc,
                        }
                    )
        else:
            file_path = _find_document_path(contract_id)
            if not file_path:
                ctx.workflow_state.legal_analysis = StageStatus.FAILED
                ctx.fail(
                    code="DOCUMENT_NOT_FOUND",
                    message="Contract document file not found.",
                    step=STEP_NAME,
                    details={"document_id": contract_id},
                )
                return ctx
            raw_clauses = self.clause_extractor.extract_clauses(file_path, contract_id, use_structured=True)
            if not raw_clauses:
                ctx.workflow_state.legal_analysis = StageStatus.FAILED
                ctx.fail(
                    code="NO_CLAUSES",
                    message="No clauses could be extracted from the document.",
                    step=STEP_NAME,
                    details={"document_id": contract_id},
                )
                return ctx
            # Map extractor output to types: we have document_section only; map to a minimal set
            extracted_types = set()
            evidence_blocks = []
            for cl in raw_clauses:
                doc_sec = (cl.get("document_section") or "other").lower().replace(" ", "_")
                extracted_types.add(doc_sec)
                for alias in _clause_type_aliases(doc_sec):
                    extracted_types.add(alias)
                evidence_text = cl.get("normalized_text") or cl.get("verbatim_text", "")
                cid = cl.get("clause_id", "")
                is_nc = _is_non_contractual_snippet(evidence_text or "")
                verbatim = cl.get("verbatim_text", "") or ""
                heading = (cl.get("clause_heading") or cl.get("clause_title") or cl.get("document_section") or "").strip()
                if not heading and verbatim:
                    first_line = (verbatim.split("\n")[0] or "").strip()
                    if len(first_line) <= 80:
                        heading = first_line
                ocr_noise_score = 1.0 - _alpha_ratio(verbatim) if verbatim else 0.0
                structure_class = _classify_structure_confidence(heading, evidence_text or "", ocr_noise_score)
                doc_sec_norm = _normalize_clause_type_for_profile(doc_sec)
                sem_label = _classify_evidence_label(evidence_text or "")
                page_start = cl.get("page_start", 0)
                evidence_blocks.append(
                    ClauseEvidenceBlock(
                        clause_id=cid,
                        page_number=page_start,
                        raw_text=cl.get("verbatim_text", ""),
                        clean_text=cl.get("normalized_text") or cl.get("verbatim_text", ""),
                        display_name=_evidence_block_display_name(sem_label, doc_sec_norm, page_start),
                        is_non_contractual=is_nc,
                        semantic_label=sem_label,
                        structure_class=structure_class,
                    )
                )
                evidence_candidates.append(
                    {
                        "clause_id": cl.get("clause_id", ""),
                        "page_number": cl.get("page_start", 0),
                        "text": (heading + " " + (evidence_text or "")).strip() if heading else (evidence_text or ""),
                        "is_non_contractual": is_nc,
                    }
                )
            # Also add common types if verbatim text suggests them (simple keyword match)
            from backend.config import settings as cfg
            for cl in raw_clauses:
                text = (cl.get("verbatim_text") or "").lower()
                for topic, keywords in (getattr(cfg, "CLAUSE_TOPIC_KEYWORDS", {}) or {}).items():
                    if any(kw in text for kw in keywords):
                        extracted_types.add(_normalize_clause_type_for_profile(topic))
                        break

        ctx.add_result(f"{STEP_NAME}.extracted_types", list(extracted_types))
        ctx.add_result(f"{STEP_NAME}.evidence_blocks", [e.model_dump() for e in evidence_blocks])

        # Fail closed: Contract Review requires evidence text to support any output.
        # If evidence blocks are empty, we cannot provide an evidence-backed review.
        if not evidence_blocks:
            ctx.workflow_state.legal_analysis = StageStatus.FAILED
            ctx.fail(
                code="NO_EVIDENCE",
                message="No extractable evidence was found in the provided document.",
                step=STEP_NAME,
                details={"document_id": contract_id},
            )
            return ctx

        # 3) Expected clause checks (evidence-first, conservative)
        expected = list(profile["expected_clauses"])
        optional = set(profile.get("optional_clauses") or [])
        risk_weights = profile.get("risk_weights") or {}

        risks: List[RiskItem] = []
        presence_map: Dict[str, Any] = {}

        def _display_status_for(status: str) -> str:
            if status == "detected":
                return "Detected"
            if status == "uncertain":
                return "Detected (Weak Evidence)"
            if status == "detected_implicit":
                return "Detected (Implicit Reference)"
            if status == "detected_distributed":
                return "Detected (Distributed Provisions)"
            if status == "detected_weak":
                return "Detected (Limited Coverage)"
            if status == "implicitly_covered":
                return "Implicitly Covered"
            return "Not Detected"

        for clause_type in expected:
            keywords = EXPECTED_CLAUSE_PATTERNS.get(clause_type, [])
            result = _detect_clause_presence(evidence_candidates, keywords, clause_type=clause_type)
            status = result["status"]
            presence_map[clause_type] = {
                "status": status,
                "display_status": _display_status_for(status),
                "clause_ids": [result["clause_id"]] if result.get("clause_id") else [],
                "page_numbers": [result["page_number"]] if result.get("page_number") else [],
                "matched_keyword": result.get("matched_keyword"),
            }

        # G1: Post-pass for implicit governing law (no country requirement; do not infer country)
        # Store the matching candidate so we can attach evidence (synthetic block) and avoid "present" without evidence.
        if "governing_law" in expected and presence_map.get("governing_law", {}).get("status") == "not_detected":
            for cand in evidence_candidates:
                t = (cand.get("text") or "").lower()
                if any(kw in t for kw in IMPLICIT_GOV_LAW_KEYWORDS):
                    presence_map["governing_law"]["status"] = "detected_implicit"
                    presence_map["governing_law"]["display_status"] = "Detected (Implicit Reference)"
                    presence_map["governing_law"]["implicit_evidence"] = {
                        "page_number": cand.get("page_number", 0),
                        "text": (cand.get("text") or "")[:500],
                    }
                    break

        # G1.5: Post-pass for implicit dispute_resolution
        if (
            "dispute_resolution" in expected
            and presence_map.get("dispute_resolution", {}).get("status") == "not_detected"
        ):
            for cand in evidence_candidates:
                t = (cand.get("text") or "").lower()
                if any(kw in t for kw in IMPLICIT_DISPUTE_RESOLUTION_KEYWORDS):
                    presence_map["dispute_resolution"]["status"] = "detected_implicit"
                    presence_map["dispute_resolution"]["display_status"] = "Detected (Implicit Reference)"
                    break

        # G4: Post-pass for implicit ip_ownership
        if (
            "ip_ownership" in expected
            and presence_map.get("ip_ownership", {}).get("status") == "not_detected"
        ):
            for cand in evidence_candidates:
                t = (cand.get("text") or "").lower()
                if any(kw in t for kw in IMPLICIT_IP_OWNERSHIP_KEYWORDS):
                    presence_map["ip_ownership"]["status"] = "detected_implicit"
                    presence_map["ip_ownership"]["display_status"] = _display_status_for("detected_implicit")
                    break

        # G5: Post-pass for implicit sla_obligations
        if (
            "sla_obligations" in expected
            and presence_map.get("sla_obligations", {}).get("status") == "not_detected"
        ):
            for cand in evidence_candidates:
                t = (cand.get("text") or "").lower()
                if any(kw in t for kw in IMPLICIT_SLA_KEYWORDS):
                    presence_map["sla_obligations"]["status"] = "detected_implicit"
                    presence_map["sla_obligations"]["display_status"] = _display_status_for("detected_implicit")
                    break

        # G2: Post-pass for benefits (composite). 0 categories must remain not_detected (no overwrite).
        if "benefits" in expected:
            detected_categories: set = set()
            for cand in evidence_candidates:
                t = (cand.get("text") or "").lower()
                for category, kws in BENEFITS_SIGNALS.items():
                    if any(k in t for k in kws):
                        detected_categories.add(category)
            if len(detected_categories) >= 2:
                presence_map["benefits"]["status"] = "detected_distributed"
                presence_map["benefits"]["display_status"] = "Detected (Distributed Provisions)"
            elif len(detected_categories) == 1:
                presence_map["benefits"]["status"] = "detected_weak"
                presence_map["benefits"]["display_status"] = "Detected (Limited Coverage)"
            # len(detected_categories) == 0: leave status as not_detected (regression guard for employment profile)

        # G3: Post-pass for implicitly_covered (e.g. conduct_discipline when termination text mentions discipline)
        for clause_type, rule in IMPLICIT_COVERAGE_RULES.items():
            if clause_type not in expected:
                continue
            entry = presence_map.get(clause_type, {})
            if entry.get("status") != "not_detected":
                continue
            primary = rule.get("primary") or []
            secondary = rule.get("secondary") or []
            note = rule.get("coverage_note") or ""
            for cand in evidence_candidates:
                t = (cand.get("text") or "").lower()
                if any(p in t for p in primary) and any(s in t for s in secondary):
                    presence_map[clause_type]["status"] = "implicitly_covered"
                    presence_map[clause_type]["display_status"] = "Implicitly Covered"
                    presence_map[clause_type]["coverage_note"] = note
                    break

        # G6: Page-level fallback for any remaining not_detected clauses.
        _still_missing = [
            ct for ct in expected
            if presence_map.get(ct, {}).get("status") == "not_detected"
        ]
        if _still_missing:
            _fallback_hits = _page_fallback_scan(contract_id, _still_missing)
            for ct, hit in _fallback_hits.items():
                if presence_map.get(ct, {}).get("status") == "not_detected":
                    presence_map[ct]["status"] = hit["status"]
                    presence_map[ct]["display_status"] = _display_status_for(hit["status"])
                    if hit.get("page_number"):
                        presence_map[ct].setdefault("page_numbers", []).append(hit["page_number"])

        # G7: Heading-synonym post-pass — scan clause titles from store against CLAUSE_HEADING_SYNONYMS.
        # Catches alternative section headings (e.g. "Proprietary Rights" for ip_ownership) that
        # survive all earlier passes because their title text is too short to be "detected".
        if clauses_from_store:
            for c in clauses_from_store:
                title_norm = _normalize_text(getattr(c, "title", "") or "")
                if not title_norm:
                    continue
                for heading_kw, canon_type in CLAUSE_HEADING_SYNONYMS.items():
                    if canon_type not in expected:
                        continue
                    if presence_map.get(canon_type, {}).get("status") != "not_detected":
                        continue
                    norm_heading_kw = _normalize_text(heading_kw)
                    if norm_heading_kw in title_norm:
                        presence_map[canon_type]["status"] = "detected_implicit"
                        presence_map[canon_type]["display_status"] = "Detected (Heading Match)"
                        break

        # G8: LLM uncertain-clause resolver — only for standard depth, gates on Ollama availability.
        # Resolves clauses stuck at 'uncertain' by asking the LLM a binary YES/NO question.
        if review_depth == "standard":
            for _g8_clause_type in expected:
                if presence_map.get(_g8_clause_type, {}).get("status") == "uncertain":
                    _g8_ev_text = next(
                        (c.get("text", "") for c in evidence_candidates
                         if c.get("clause_id") in (presence_map[_g8_clause_type].get("clause_ids") or [])),
                        "",
                    )
                    _g8_disp = EXPECTED_CLAUSE_DISPLAY_NAMES.get(_g8_clause_type, _g8_clause_type)
                    _g8_resolved = self._resolve_uncertain_with_llm(_g8_clause_type, _g8_ev_text, _g8_disp)
                    if _g8_resolved != "uncertain":
                        presence_map[_g8_clause_type]["status"] = _g8_resolved
                        presence_map[_g8_clause_type]["display_status"] = _display_status_for(_g8_resolved)

        # G9: Cross-clause contradiction detection — only for standard depth.
        contradiction_risks: List[RiskItem] = []
        if review_depth == "standard":
            contradiction_risks = self._identify_contradiction_risks(presence_map, evidence_candidates)

        # Add risk rows only for clause_types that are not detected (any variant)
        _skip_risk_statuses = (
            "detected", "detected_implicit", "detected_distributed", "detected_weak",
            "implicitly_covered",
            "uncertain",   # heading/weak-evidence matches don't generate misleading risk items
        )
        for clause_type in expected:
            status = presence_map.get(clause_type, {}).get("status", "not_detected")
            if status in _skip_risk_statuses:
                continue
            entry = presence_map[clause_type]
            result = {"clause_id": (entry.get("clause_ids") or [None])[0], "page_number": (entry.get("page_numbers") or [None])[0]}
            severity = risk_weights.get(clause_type, "medium")
            if isinstance(severity, str):
                severity = severity.lower()
            else:
                severity = "medium"
            if severity not in ("high", "medium", "low"):
                severity = "medium"
            description = f"Clause not confidently detected: {clause_type.replace('_', ' ')}."
            clause_ids = [result["clause_id"]] if result.get("clause_id") else []
            page_numbers = [result["page_number"]] if result.get("page_number") else []
            missing_clause = True
            display_names = [_resolve_clause_display_name(cid) for cid in clause_ids] if clause_ids else []
            risks.append(
                RiskItem(
                    description=description,
                    severity=severity,
                    status=status,
                    clause_types=[clause_type],
                    missing_clause=missing_clause,
                    clause_ids=clause_ids,
                    page_numbers=page_numbers,
                    display_names=display_names,
                )
            )

        ctx.add_result(f"{STEP_NAME}.presence_map", presence_map)

        # Build not_detected_clauses with canonical grouping (e.g. compensation vs salary_wages).
        missing_keys = [ct for ct in expected if presence_map.get(ct, {}).get("status") == "not_detected"]
        not_detected_clauses: List[str] = []
        # Handle canonical groups first (only show one entry for each group)
        for canon_key, members in CANONICAL_CLAUSE_GROUPS.items():
            group_members = [m for m in members if m in missing_keys]
            if group_members:
                not_detected_clauses.append(
                    EXPECTED_CLAUSE_DISPLAY_NAMES.get(canon_key, canon_key.replace("_", " ").title())
                )
                missing_keys = [ct for ct in missing_keys if ct not in group_members]

        # Add remaining standalone missing keys
        for ct in missing_keys:
            not_detected_clauses.append(
                EXPECTED_CLAUSE_DISPLAY_NAMES.get(ct, ct.replace("_", " ").title())
            )

        # 3b) Problematic language detection (heuristic)
        if clauses_from_store:
            risks.extend(
                self._identify_problematic_language_risks(
                    clauses_from_store=clauses_from_store,
                    profile_risk_weights=risk_weights,
                    review_depth=review_depth,
                )
            )

        # 3c) Jurisdiction checks (evidence-based)
        if clauses_from_store and jurisdiction:
            risks.extend(
                self._identify_jurisdiction_risks(
                    clauses_from_store=clauses_from_store,
                    jurisdiction=jurisdiction,
                    profile_risk_weights=risk_weights,
                )
            )

        # --- deduplication pass ---
        _seen: dict = {}
        _deduped: list = []
        for _r in risks:
            _key = (_r.description, frozenset(_r.clause_ids or []))
            if _key not in _seen:
                _seen[_key] = len(_deduped)
                _deduped.append(_r)
            else:
                # merge page_numbers into the first occurrence
                _existing = _deduped[_seen[_key]]
                for _pn in (_r.page_numbers or []):
                    if _pn not in (_existing.page_numbers or []):
                        (_existing.page_numbers or []).append(_pn)
        risks = _deduped

        # Guardrail: enforce on generated text only (risk descriptions, executive summary).
        for r in risks:
            r.description = enforce_non_prescriptive_language(r.description, step=f"{STEP_NAME}.risk_description")

        # Sort deterministically: severity then missing clauses first, then description.
        risks = sorted(
            risks,
            key=lambda r: (_severity_rank(r.severity), 0 if r.missing_clause else 1, (r.description or "").lower()),
        )

        # ── Improvement 1: attach verbatim evidence snippets ──────────────────
        # Build clause_type → matched_keyword map from presence_map for snippet highlighting
        clause_matched_keywords: Dict[str, str] = {
            ct: entry["matched_keyword"]
            for ct, entry in presence_map.items()
            if entry.get("matched_keyword")
        }
        risks = self._attach_verbatim_evidence(risks, evidence_blocks, clause_matched_keywords=clause_matched_keywords)

        # ── Improvement 2: attach severity reason + recommendation ─────────────
        risks = self._attach_explanations(risks)

        ctx.add_result(f"{STEP_NAME}.risks", [r.model_dump() for r in risks])

        # 4) Document confidence (for severity cap and guardrail)
        section_non_standard_count = sum(1 for b in evidence_blocks if getattr(b, "structure_class", None) == "section_non_standard")
        total_blocks = max(len(evidence_blocks), 1)
        document_confidence = "high" if (
            not document_classification_warning
            and (section_non_standard_count / total_blocks) < 0.2
        ) else ("medium" if not document_classification_warning else "low")

        # 5) Final output guardrail: on clean documents, do not show section_non_standard
        if document_confidence == "high" and section_non_standard_count > 0:
            for block in evidence_blocks:
                if getattr(block, "structure_class", None) == "section_non_standard":
                    block.structure_class = "provision"  # downgrade; no heading on block to prefer clause
                    logging.debug(
                        "contract_review: downgraded evidence block structure_class to provision (document_confidence=high)"
                    )

        # 6) used_implicit_or_distributed_logic: after guardrail so downgraded blocks do not count
        # OCR confidence below threshold: significant fraction of blocks have high OCR noise
        noisy_count = sum(
            1 for b in evidence_blocks
            if (1.0 - _alpha_ratio(getattr(b, "raw_text", "") or "")) >= OCR_NOISE_THRESHOLD
        )
        ocr_confidence_below_threshold = (noisy_count / total_blocks) > 0.5
        used_implicit_or_distributed_logic = (
            presence_map.get("governing_law", {}).get("status") == "detected_implicit"
            or presence_map.get("benefits", {}).get("status") == "detected_distributed"
            or any(presence_map.get(ct, {}).get("status") == "implicitly_covered" for ct in expected)
            or any(getattr(b, "structure_class", None) == "section_non_standard" for b in evidence_blocks)
            or ocr_confidence_below_threshold
        )

        # 7) Key Review Observations (template-based, reviewer-grade; no raw counts)
        exec_items = _build_key_review_observations(
            presence_map=presence_map,
            risk_weights=risk_weights,
            expected=expected,
            document_confidence_high=(document_confidence == "high"),
            jurisdiction=jurisdiction,
        )
        for item in exec_items:
            item.text = enforce_non_prescriptive_language(item.text, step=f"{STEP_NAME}.executive_summary")

        # 8) Synthetic evidence block for implicit governing law so "present" has visible evidence
        gov_entry = presence_map.get("governing_law", {})
        if gov_entry.get("status") == "detected_implicit" and gov_entry.get("implicit_evidence"):
            ie = gov_entry["implicit_evidence"]
            evidence_blocks.append(
                ClauseEvidenceBlock(
                    clause_id="governing_law_implicit",
                    page_number=ie.get("page_number", 0),
                    raw_text=ie.get("text", ""),
                    clean_text=ie.get("text", ""),
                    display_name="Governing Law (implicit)",
                    is_non_contractual=False,
                    semantic_label=None,
                    structure_class="provision",
                )
            )

        # 9) Implicitly covered clauses (display names + notes for UI)
        implicitly_covered_clauses: List[str] = []
        implicit_coverage_notes: Optional[Dict[str, str]] = None
        for ct in expected:
            entry_status = presence_map.get(ct, {}).get("status")
            if entry_status == "implicitly_covered":
                display = EXPECTED_CLAUSE_DISPLAY_NAMES.get(ct, ct.replace("_", " ").title())
                implicitly_covered_clauses.append(display)
                note = presence_map.get(ct, {}).get("coverage_note")
                if note:
                    if implicit_coverage_notes is None:
                        implicit_coverage_notes = {}
                    implicit_coverage_notes[display] = note
            elif entry_status == "uncertain":
                # Uncertain clauses are detected with weak/heading-only evidence.
                # Surface them in implicitly_covered_clauses with a note rather than
                # generating a misleading risk item.
                display = EXPECTED_CLAUSE_DISPLAY_NAMES.get(ct, ct.replace("_", " ").title())
                display_with_note = f"{display} (Weak Evidence)"
                implicitly_covered_clauses.append(display_with_note)
                if implicit_coverage_notes is None:
                    implicit_coverage_notes = {}
                implicit_coverage_notes[display_with_note] = (
                    "Detected with limited textual evidence; manual review of the relevant section is recommended."
                )

        # 10) Build deliverable and store in context

        # Group 2a: compute overall risk score
        _high_count = sum(1 for r in risks if r.severity == "high")
        _medium_count = sum(1 for r in risks if r.severity == "medium")
        _risk_score = _high_count * 3 + _medium_count
        _risk_label = "high_risk" if _high_count >= 2 else ("medium_risk" if _risk_score >= 2 else "low_risk")

        # Group 7: statutory notes — jurisdiction-specific article references
        statutory_notes: Optional[Dict[str, Any]] = None
        if jurisdiction and self.jurisdiction_statutes:
            _canon_jur, _, _ = _canon_jurisdiction(jurisdiction)
            jur_data = self.jurisdiction_statutes.get(_canon_jur) if _canon_jur else None
            if jur_data:
                statutory_notes = {
                    EXPECTED_CLAUSE_DISPLAY_NAMES.get(k, k): v
                    for k, v in jur_data.items()
                    if k in expected
                }

        response = ContractReviewResponse(
            workflow_id=ctx.workflow_id,
            document_id=contract_id,
            contract_type=profile.get("contract_type", contract_type_key),
            jurisdiction=jurisdiction,
            risks=risks,
            evidence=evidence_blocks,
            executive_summary=exec_items,
            not_detected_clauses=not_detected_clauses,
            implicitly_covered_clauses=implicitly_covered_clauses,
            implicit_coverage_notes=implicit_coverage_notes,
            document_classification_warning=document_classification_warning,
            used_implicit_or_distributed_logic=used_implicit_or_distributed_logic,
            risk_score=_risk_score,
            risk_label=_risk_label,
            contradiction_risks=contradiction_risks,
            statutory_notes=statutory_notes,
        )
        ctx.add_result(f"{STEP_NAME}.response", response.model_dump())
        ctx.workflow_state.legal_analysis = StageStatus.COMPLETE
        ctx.status = "completed"
        return ctx
