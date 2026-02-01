"""
Contract Review workflow service.

Executes the Contract Review workflow using WorkflowContext: load profile,
gather clauses (from store or extraction), identify missing clauses and risks,
build evidence blocks and executive summary. No raw dict returns; all results
go into ctx.intermediate_results. Failures set ctx.error and ctx.status="failed".
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Iterable, Tuple

from backend.config import settings
from backend.models.workflow import WorkflowContext
from backend.models.contract_review import (
    ContractReviewResponse,
    RiskItem,
    ClauseEvidenceBlock,
    ExecutiveSummaryItem,
)
from backend.services.contract_profile_loader import load_contract_profile
from backend.services.clause_store import ClauseStore
from backend.services.clause_extraction import ClauseExtractionService
from backend.services.guardrails import enforce_non_prescriptive_language


STEP_NAME = "contract_review"
MIN_SNIPPET_LENGTH = 120
MIN_ALPHA_RATIO = 0.5
TERMINATION_TOKEN_DISTANCE = 2

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
    "notice": ["notice period", "written notice", "days notice"],
    "salary_wages": ["salary", "wages", "remuneration"],
    "compensation": ["compensation", "indemnity", "blood money"],
    "benefits": ["benefit", "allowance", "insurance"],
    "governing_law": ["governing law", "shall be governed by"],
    "jurisdiction": ["jurisdiction", "courts of"],
    "conduct_discipline": ["discipline", "misconduct", "code of conduct"],
    # NDA profile
    "confidentiality": ["confidential", "confidentiality", "non-disclosure"],
    "term": ["term", "duration", "effective date"],
    "liability": ["liability", "limitation of liability", "indemnify"],
    "remedies": ["remedy", "injunctive relief"],
    # MSA profile
    "dispute_resolution": ["dispute resolution", "arbitration", "arbitral"],
}


def _find_document_path(document_id: str) -> Optional[str]:
    """Resolve document file path from document_id."""
    for ext in [".pdf", ".docx", ".doc"]:
        p = settings.DOCUMENTS_PATH / f"{document_id}{ext}"
        if p.exists():
            return str(p)
    for ext in [".pdf", ".docx", ".doc"]:
        matches = list(settings.DOCUMENTS_PATH.glob(f"{document_id}*{ext}"))
        if matches:
            return str(matches[0])
    return None


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
    return _levenshtein_distance(token, "terminate") <= TERMINATION_TOKEN_DISTANCE


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

        # Hard rule for termination detection
        if clause_type == "termination":
            if "this agreement" in normalized_text and any(_looks_like_terminate(t) for t in tokens):
                return {
                    "status": "detected",
                    "clause_id": cand.get("clause_id"),
                    "page_number": cand.get("page_number"),
                    "matched_keyword": "this agreement + terminate*",
                }
        for kw in keywords:
            if kw and kw in normalized_text:
                is_weak = len(raw_text) < MIN_SNIPPET_LENGTH or _alpha_ratio(raw_text) < MIN_ALPHA_RATIO
                if len(normalized_text) > 200:
                    is_weak = False
                return {
                    "status": "uncertain" if is_weak else "detected",
                    "clause_id": cand.get("clause_id"),
                    "page_number": cand.get("page_number"),
                    "matched_keyword": kw,
                }

        # OCR-tolerant termination token scan (keyword-independent)
        if clause_type == "termination" and any(_looks_like_terminate(t) for t in tokens):
            is_weak = len(raw_text) < MIN_SNIPPET_LENGTH or _alpha_ratio(raw_text) < MIN_ALPHA_RATIO
            if len(normalized_text) > 200:
                is_weak = False
            return {
                "status": "uncertain" if is_weak else "detected",
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
            risks.append(
                RiskItem(
                    description=desc,
                    severity=sev,
                    clause_types=["governing_law", "jurisdiction"],
                    missing_clause=False,
                    clause_ids=[cid for cid in clause_ids if cid],
                    page_numbers=sorted({p for p in page_numbers if p}),
                )
            )
        elif has_other:
            # Expected present, but other jurisdictions also appear; flag as a consistency check.
            desc = (
                f"Evidence includes references consistent with '{canon}' and also mentions other jurisdictions; "
                "this may require consistency verification."
            )
            risks.append(
                RiskItem(
                    description=desc,
                    severity="low",
                    clause_types=["governing_law", "jurisdiction"],
                    missing_clause=False,
                    clause_ids=[cid for cid in clause_ids if cid],
                    page_numbers=sorted({p for p in page_numbers if p}),
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
            ctx.fail(
                code="MISSING_INPUT",
                message="Contract Review requires document_ids[0] (contract_id).",
                step=STEP_NAME,
                details={},
            )
            return ctx

        meta = ctx.metadata or {}
        contract_type_key = meta.get("contract_type") or "employment"
        jurisdiction = ctx.jurisdiction or meta.get("jurisdiction")
        review_depth = meta.get("review_depth") or "standard"

        # 1) Load contract profile
        ctx.current_step = f"{STEP_NAME}.load_profile"
        try:
            profile = load_contract_profile(contract_type_key)
        except FileNotFoundError as e:
            ctx.fail(
                code="PROFILE_NOT_FOUND",
                message=f"Contract profile not found for type '{contract_type_key}'.",
                step=STEP_NAME,
                details={"contract_type": contract_type_key, "path": str(e)},
            )
            return ctx
        except ValueError as e:
            ctx.fail(
                code="INVALID_PROFILE",
                message=f"Invalid contract profile: {e}",
                step=STEP_NAME,
                details={"contract_type": contract_type_key},
            )
            return ctx
        except Exception as e:
            ctx.fail(
                code="PROFILE_LOAD_ERROR",
                message=f"Failed to load contract profile: {e}",
                step=STEP_NAME,
                details={"contract_type": contract_type_key},
            )
            return ctx

        ctx.add_result(f"{STEP_NAME}.profile", profile)

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
                    evidence_blocks.append(
                        ClauseEvidenceBlock(
                            clause_id=c.clause_id,
                            page_number=ev.page or 0,
                            raw_text=ev.raw_text or "",
                            clean_text=ev.clean_text or ev.raw_text or "",
                        )
                    )
                    evidence_candidates.append(
                        {
                            "clause_id": c.clause_id,
                            "page_number": ev.page or 0,
                            "text": ev.clean_text or ev.raw_text or "",
                        }
                    )
        else:
            file_path = _find_document_path(contract_id)
            if not file_path:
                ctx.fail(
                    code="DOCUMENT_NOT_FOUND",
                    message="Contract document file not found.",
                    step=STEP_NAME,
                    details={"document_id": contract_id},
                )
                return ctx
            raw_clauses = self.clause_extractor.extract_clauses(file_path, contract_id, use_structured=True)
            if not raw_clauses:
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
                evidence_blocks.append(
                    ClauseEvidenceBlock(
                        clause_id=cl.get("clause_id", ""),
                        page_number=cl.get("page_start", 0),
                        raw_text=cl.get("verbatim_text", ""),
                        clean_text=cl.get("normalized_text") or cl.get("verbatim_text", ""),
                    )
                )
                evidence_candidates.append(
                    {
                        "clause_id": cl.get("clause_id", ""),
                        "page_number": cl.get("page_start", 0),
                        "text": evidence_text or "",
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
        for clause_type in expected:
            keywords = EXPECTED_CLAUSE_PATTERNS.get(clause_type, [])
            result = _detect_clause_presence(evidence_candidates, keywords, clause_type=clause_type)
            status = result["status"]
            presence_map[clause_type] = {
                "status": status,
                "clause_ids": [result["clause_id"]] if result.get("clause_id") else [],
                "page_numbers": [result["page_number"]] if result.get("page_number") else [],
                "matched_keyword": result.get("matched_keyword"),
            }
            if status == "detected":
                continue

            severity = risk_weights.get(clause_type, "medium")
            if isinstance(severity, str):
                severity = severity.lower()
            else:
                severity = "medium"
            if severity not in ("high", "medium", "low"):
                severity = "medium"

            if status == "uncertain":
                description = f"Clause detected with weak evidence signal: {clause_type.replace('_', ' ')}."
                clause_ids = [result["clause_id"]] if result.get("clause_id") else []
                page_numbers = [result["page_number"]] if result.get("page_number") else []
                missing_clause = False
            else:
                description = f"Clause not confidently detected: {clause_type.replace('_', ' ')}."
                clause_ids = []
                page_numbers = []
                missing_clause = True

            risks.append(
                RiskItem(
                    description=description,
                    severity=severity,
                    status=status,
                    clause_types=[clause_type],
                    missing_clause=missing_clause,
                    clause_ids=clause_ids,
                    page_numbers=page_numbers,
                )
            )

        ctx.add_result(f"{STEP_NAME}.presence_map", presence_map)

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

        # Guardrail: enforce on generated text only (risk descriptions, executive summary).
        for r in risks:
            r.description = enforce_non_prescriptive_language(r.description, step=f"{STEP_NAME}.risk_description")

        # Sort deterministically: severity then missing clauses first, then description.
        risks = sorted(
            risks,
            key=lambda r: (_severity_rank(r.severity), 0 if r.missing_clause else 1, (r.description or "").lower()),
        )

        ctx.add_result(f"{STEP_NAME}.risks", [r.model_dump() for r in risks])

        # 4) Executive summary (structured, no LLM - avoid prescriptive language)
        exec_items: List[ExecutiveSummaryItem] = []
        high = [r for r in risks if r.severity == "high"]
        medium = [r for r in risks if r.severity == "medium"]
        low = [r for r in risks if r.severity == "low"]

        if high:
            exec_items.append(
                ExecutiveSummaryItem(
                    text=f"High-priority review flags identified: {len(high)} item(s).",
                    severity="high",
                    related_risk_indices=[i for i, r in enumerate(risks) if r.severity == "high"],
                )
            )
        if medium:
            exec_items.append(
                ExecutiveSummaryItem(
                    text=f"Medium-priority review flags identified: {len(medium)} item(s).",
                    severity="medium",
                    related_risk_indices=[i for i, r in enumerate(risks) if r.severity == "medium"],
                )
            )
        if low and (review_depth or "standard").lower() != "quick":
            exec_items.append(
                ExecutiveSummaryItem(
                    text=f"Low-priority review flags identified: {len(low)} item(s) (potential ambiguity/breadth markers).",
                    severity="low",
                    related_risk_indices=[i for i, r in enumerate(risks) if r.severity == "low"],
                )
            )
        if not risks:
            exec_items.append(
                ExecutiveSummaryItem(
                    text="No review flags were identified for the selected profile and evidence.",
                    severity=None,
                    related_risk_indices=[],
                )
            )
        else:
            not_detected_count = sum(1 for r in risks if r.status == "not_detected")
            uncertain_count = sum(1 for r in risks if r.status == "uncertain")
            other_flags = len(risks) - not_detected_count - uncertain_count
            exec_items.append(
                ExecutiveSummaryItem(
                    text=(
                        "Summary of review flags: "
                        f"{not_detected_count} clause(s) not confidently detected, "
                        f"{uncertain_count} clause(s) detected with weak evidence signals, "
                        f"{other_flags} other evidence-linked flag(s)."
                    ),
                    severity=None,
                    related_risk_indices=[],
                )
            )

        # Guardrail enforcement for executive summary items
        for item in exec_items:
            item.text = enforce_non_prescriptive_language(item.text, step=f"{STEP_NAME}.executive_summary")

        # 5) Build deliverable and store in context
        response = ContractReviewResponse(
            workflow_id=ctx.workflow_id,
            document_id=contract_id,
            contract_type=profile.get("contract_type", contract_type_key),
            jurisdiction=jurisdiction,
            risks=risks,
            evidence=evidence_blocks,
            executive_summary=exec_items,
        )
        ctx.add_result(f"{STEP_NAME}.response", response.model_dump())
        ctx.status = "completed"
        return ctx
