# Plan: Fix Systemic False-Negative Clause Detection (MSA)

## 1. Diagnosis: Exact Failure Chain

The pipeline has **four compounding failure points**. They are ordered below from most impactful to least. All four must be fixed for complete resolution.

---

### Failure Point 1 — `_page_fallback_scan`: Silent Exception (P0 CRITICAL)

**File**: `backend/services/contract_review_service.py` · Lines ~887–927

The page-level fallback scan (G6 rescue) is broken. The try/except only covers the `FileParser.parse` call, not the for-loop that follows:

```python
try:
    pages = FileParser.parse(Path(file_path))  # ← protected
except Exception:
    return {}

# ← NOT protected
for page_text, page_num in pages:             # ValueError if 3-tuple
    ...
```

`FileParser.parse` almost certainly returns the same `(text, page_num, is_ocr)` 3-tuples that `ingestion_service.parser.parse_file` does (the extraction service explicitly handles both formats). When the for-loop tries to unpack a 3-tuple into 2 variables it throws `ValueError: too many values to unpack`. Since the caller also has no try/except, the exception propagates silently and the fallback returns nothing.

**Consequence**: Confidentiality, Dispute Resolution, IP Ownership, and SLA Obligations ALL have clear keyword matches in the raw document text. This one bug blocks every one of them from being rescued by G6.

---

### Failure Point 2 — `contract_entry_patterns`: Employment-Only Gate (P1 HIGH)

**File**: `backend/services/structured_clause_extraction.py` · Lines ~202–209

The gate that activates clause extraction on a page uses these patterns exclusively:

```python
r"\bemployment\s+agreement\b",
r"\b(first|1st)\s+party\b",
r"\b(second|2nd)\s+party\b",
r"\bthis\s+agreement\b",
r"\bthe\s+first\s+party\s+shall\b",
r"\bthe\s+second\s+party\s+shall\b",
```

An MSA uses "Master Service Agreement", "Service Provider", "Client", "IPPB", "Vendor". Only `\bthis\s+agreement\b` has any chance of matching. If the gate never activates, all pages classified as `CONTRACTUAL_TERMS` are silently skipped without any extracted clauses.

**Consequence**: The ClauseStore is sparse or empty for the MSA. Sparse store → sparse `evidence_candidates` → primary detection keywords find nothing → "not_detected".

---

### Failure Point 3 — `ANNEXURES_SCHEDULES` Excluded from Operative Sections (P1 HIGH)

**File**: `backend/services/structured_clause_extraction.py` · Lines ~194–199

```python
self.operative_sections = {
    DocumentSection.CONTRACTUAL_TERMS,
    DocumentSection.JUDICIAL_REASONING,
    DocumentSection.STATUTORY_TEXT,
    DocumentSection.UNKNOWN,
}
```

`ANNEXURES_SCHEDULES` is not in `operative_sections`. Pages detected as schedule/annexure are discarded with no clause extraction. The MSA's Schedule-A contains SLA terms, KPIs, liquidated damages, and performance monitoring — all labeled SLA Obligations in the profile. These pages are classified as `ANNEXURES_SCHEDULES` and thrown away entirely.

**Consequence**: `sla_obligations` evidence is never extracted. The G5 implicit post-pass also fails because it only scans `evidence_candidates` (built from the store/extractor), not raw pages. G6 fallback would catch it — but it's broken per Failure Point 1.

---

### Failure Point 4 — `_has_party_definitions` Misses MSA Parties (P1 HIGH)

**File**: `backend/services/structured_clause_extraction.py` · `_classify_page_section`, Line ~562

The section classifier tightens `CONTRACTUAL_TERMS` pages by requiring party definitions:

```python
if result == DocumentSection.CONTRACTUAL_TERMS:
    if not self._has_party_definitions(text_lower) and not self._has_sustained_modal_language(text):
        return DocumentSection.AMBIGUOUS
```

If `_has_party_definitions` uses hardcoded employment terms ("employer", "employee"), MSA pages with "Client", "Service Provider", "IPPB", "Vendor" fail this check and are downgraded to AMBIGUOUS → then UNKNOWN for extraction.

UNKNOWN sections are technically operative but lack the contract entry gate, so clause starts may not be detected correctly. The net effect is degraded or missing clause extraction for MSA body pages.

---

## 2. Fix Specifications

### FIX 1 — `_page_fallback_scan`: Robust Error Handling + Tuple Normalization

**File**: `backend/services/contract_review_service.py`
**Priority**: P0 — implement first, independently of all other fixes
**Estimated impact**: Resolves all 4 false negatives on existing MSA without re-extraction

**Current code** (~lines 887–927):
```python
def _page_fallback_scan(document_id, missing_clause_types):
    ...
    try:
        pages = FileParser.parse(Path(file_path))
    except Exception:
        return {}
    results = {}
    for clause_type in missing_clause_types:
        ...
        for page_text, page_num in pages:         # ← BUG: no tuple normalization
            normalized_page = _normalize_text(page_text)
            ...
    return results
```

**Required change**:
```python
def _page_fallback_scan(document_id, missing_clause_types):
    ...
    logger.debug(
        "_page_fallback_scan: scanning %d missing types for document %s",
        len(missing_clause_types), document_id
    )
    try:
        raw_pages = FileParser.parse(Path(file_path))
    except Exception as exc:
        logger.warning("_page_fallback_scan: FileParser.parse failed: %s", exc)
        return {}

    # Normalize: handle both (text, page_num) and (text, page_num, is_ocr) formats
    pages: list = []
    for page_data in raw_pages:
        if len(page_data) >= 2:
            pages.append((page_data[0], page_data[1]))

    results: Dict[str, Dict[str, Any]] = {}
    try:
        for clause_type in missing_clause_types:
            if clause_type in results:
                continue
            keywords = EXPECTED_CLAUSE_PATTERNS.get(clause_type, [])
            if not keywords:
                continue
            for page_text, page_num in pages:
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
    except Exception as exc:
        logger.warning(
            "_page_fallback_scan: exception during page scan (partial results): %s", exc
        )

    logger.debug(
        "_page_fallback_scan: hits for %s: %s",
        document_id, list(results.keys())
    )
    return results
```

---

### FIX 2 — `EXPECTED_CLAUSE_PATTERNS`: Expand for MSA Synonym Coverage

**File**: `backend/services/contract_review_service.py`
**Priority**: P1 — do in the same pass as FIX 1
**Estimated impact**: Strengthens both primary `_detect_clause_presence` and G6 fallback keyword matching

Add the following entries to the existing pattern lists (do NOT replace existing keywords):

```python
"dispute_resolution": [
    # existing...
    "dispute resolution", "arbitration", "arbitral",
    "mediation", "conciliation", "settlement of disputes",
    # NEW — MSA-specific
    "resolution of disputes",
    "disputes and arbitration",
    "arbitration clause",
    "escalation mechanism",
    "escalation procedure",
    "disputes arising out of",
    "disputes arising from",
],

"confidentiality": [
    # existing...
    "confidential", "confidentiality", "non disclosure",
    "confidential information", "proprietary information", "trade secret",
    # NEW
    "confidential obligation",
    "obligation of confidentiality",
    "duty of confidentiality",
    "shall maintain confidentiality",
    "non-disclosure obligation",
],

"ip_ownership": [
    # existing...
    "intellectual property", "proprietary rights", "ip rights",
    "work product", "ownership of", "assigns all right",
    "vests in", "shall own", "deliverables", "work for hire",
    "background ip", "foreground ip", "moral rights", "ip ownership",
    "intellectual property rights",
    # NEW — MSA-specific
    "proprietary right",              # singular
    "ownership of all deliverables",
    "client shall own",
    "ippb shall own",
    "vendor shall assign",
    "all rights title and interest",
    "rights in the deliverables",
],

"sla_obligations": [
    # existing...
    "service level", "sla", "schedule a",
    "service standards", "uptime", "availability", "response time",
    "liquidated damages", "performance monitoring", "business continuity",
    "key performance", "kpi", "service credits", "penalty",
    "sla schedule",
    # NEW — MSA-specific
    "service level agreement",
    "sla timelines",
    "performance standard",
    "penalty for delay",
    "delay penalty",
    "penalty for breach",
    "service quality",
    "timelines and milestones",
    "target uptime",
    "response sla",
],
```

---

### FIX 3 — `IMPLICIT_*_KEYWORDS`: Expand Post-Pass Detection

**File**: `backend/services/contract_review_service.py`
**Priority**: P1 — same pass as FIX 1 and FIX 2
**Estimated impact**: Strengthens G1.5 (dispute), G4 (IP), G5 (SLA) implicit post-passes

```python
IMPLICIT_DISPUTE_RESOLUTION_KEYWORDS: List[str] = [
    # existing...
    "resolution of disputes",
    "amicable settlement",
    "conciliation",
    "arbitration tribunal",
    "settlement of disputes",
    "disputes shall be",
    "disputes arising",
    "governing disputes",
    # NEW
    "disputes and arbitration",
    "arbitration clause",
    "escalation mechanism",
    "escalation procedure",
    "disputes arising out of",
    "disputes arising from",
    "refer to arbitration",
]

IMPLICIT_IP_OWNERSHIP_KEYWORDS: List[str] = [
    # existing...
    "proprietary rights",
    "intellectual property rights",
    "vests in",
    "client shall own",
    "assigned to client",
    "work for hire",
    "deliverables belong",
    "ip ownership",
    # NEW
    "ippb shall own",
    "vendor shall assign",
    "ownership of deliverables",
    "all rights in and to",
    "rights in the deliverables",
    "all intellectual property",
]

IMPLICIT_SLA_KEYWORDS: List[str] = [
    # existing...
    "liquidated damages",
    "performance monitoring",
    "business continuity",
    "schedule a",
    "sla schedule",
    "kpi",
    "service credits",
    "penalty clause",
    # NEW
    "service level",
    "performance standard",
    "service level agreement",
    "uptime",
    "response time",
    "penalty for delay",
    "sla timelines",
    "target uptime",
    "timelines and milestones",
]
```

---

### FIX 4 — `contract_entry_patterns`: Add MSA-Specific Gate Patterns

**File**: `backend/services/structured_clause_extraction.py`
**Priority**: P1 — required for any newly uploaded MSA documents
**Estimated impact**: Fixes extraction for all future MSA uploads; requires re-extraction to fix current MSA

The gate logic in `_meets_contract_entry_gate` requires at least 2 patterns to match on the same page. Add the following alongside existing employment patterns:

```python
self.contract_entry_patterns = [
    # Employment agreements (existing)
    r"\bemployment\s+agreement\b",
    r"\b(first|1st)\s+party\b",
    r"\b(second|2nd)\s+party\b",
    r"\bthis\s+agreement\b",
    r"\bthe\s+first\s+party\s+shall\b",
    r"\bthe\s+second\s+party\s+shall\b",
    # Commercial / MSA patterns (NEW)
    r"\bmaster\s+service\s+agreement\b",
    r"\bservice\s+provider\s+shall\b",
    r"\bthe\s+client\s+shall\b",
    r"\bthe\s+vendor\s+shall\b",
    r"\bservice\s+agreement\b",
    r"\bthe\s+parties\s+agree\b",
    r"\bhereby\s+agree[sd]?\b",
    r"\bin\s+witness\s+whereof\b",
    r"\bnow[\s,]+therefore\b",
    r"\bfor\s+good\s+and\s+valuable\s+consideration\b",
]
```

**Note**: After deploying this fix, trigger re-extraction for all existing MSA documents by clearing their ClauseStore entries and re-uploading or calling the extraction endpoint.

---

### FIX 5 — `operative_sections`: Include Schedules/Annexures for CONTRACT Type

**File**: `backend/services/structured_clause_extraction.py`
**Priority**: P1 — required to capture SLA content from Schedule-A
**Estimated impact**: Captures SLA obligations from annexures in all future MSA extractions

In `extract_structured_clauses`, after detecting document type, compute effective operative sections based on document type:

```python
document_type = self.detect_document_type(pages_text)
self.last_document_type = document_type

# For contracts, schedules/annexures contain operative SLA/IP/payment content.
# For judgments/statutes, annexures are supplementary only.
if document_type == DocumentType.CONTRACT:
    effective_operative_sections = self.operative_sections | {
        DocumentSection.ANNEXURES_SCHEDULES
    }
else:
    effective_operative_sections = self.operative_sections
```

Then replace all uses of `self.operative_sections` in `extract_structured_clauses` with `effective_operative_sections`:
- The global fail-closed check: `any(page['section'] in effective_operative_sections ...)`
- The section skip gate: `if section not in effective_operative_sections:`
- The continuation check: `elif not current_clause_buffer and section in effective_operative_sections:`

---

### FIX 6 — `_has_party_definitions`: Add MSA Party Terms

**File**: `backend/services/structured_clause_extraction.py`
**Priority**: P1 — read the method first to confirm it uses hardcoded terms
**Estimated impact**: Prevents MSA pages from being downgraded to AMBIGUOUS

Read `_has_party_definitions` implementation. If it matches against a hardcoded set of employment-specific party terms, expand the set to include:

```python
MSA_PARTY_TERMS = [
    "service provider", "the client", "the vendor", "licensor", "licensee",
    "purchaser", "the buyer", "the seller", "the supplier", "ippb",
    "hereinafter referred to as", "hereinafter called",
]
```

Also review `_has_sustained_modal_language` — it is likely already correct (modal verbs like "shall", "must", "will" are universal). Confirm this and leave unchanged if so.

---

### FIX 7 — `ClauseTaxonomyService`: Add ip_ownership and sla_obligations

**File**: `backend/services/clause_taxonomy.py`
**Priority**: P2 — improves long-term extraction quality; not required for the immediate false-negative fix
**Estimated impact**: Correctly classifies extracted clauses in the ClauseStore taxonomy

**In `_CATEGORY_SLUG_KEYWORDS`** (module-level dict, ~lines 27–41):
```python
_CATEGORY_SLUG_KEYWORDS: Dict[str, List[str]] = {
    # existing entries...
    # NEW
    "ip_ownership": [
        "intellectual property", "proprietary rights", "ip ownership",
        "deliverables", "work product", "work for hire", "vests in",
        "intellectual property rights", "background ip", "foreground ip",
    ],
    "sla_obligations": [
        "service level", "sla", "performance monitoring",
        "liquidated damages", "uptime", "availability", "response time",
        "key performance", "kpi", "service credits", "business continuity",
    ],
}
```

**In `_LEGAL_CATEGORY_PRIORITY`** (~line 10), insert before "assignment":
```python
_LEGAL_CATEGORY_PRIORITY = [
    "governing_law",
    "termination",
    "indemnity",
    "liability",
    "confidentiality",
    "arbitration",
    "ip_ownership",      # NEW
    "sla_obligations",   # NEW
    "assignment",
    "payment",
    ...
]
```

**In `ClauseTaxonomyService.__init__` category_keywords dict** (~lines 59–107), add:
```python
ClauseType.IP_OWNERSHIP: [
    "intellectual property", "proprietary rights", "ip ownership",
    "deliverables", "work product", "work for hire", "vests in",
    "intellectual property rights",
],
ClauseType.SLA_OBLIGATIONS: [
    "service level", "sla", "performance monitoring",
    "liquidated damages", "uptime", "kpi", "service credits",
    "business continuity",
],
```

**Prerequisite**: Confirm `ClauseType.IP_OWNERSHIP` and `ClauseType.SLA_OBLIGATIONS` exist in `backend/models/clause.py`. If not, add them to the ClauseType enum.

---

## 3. Implementation Order

Implement in this exact sequence (dependencies respected):

| Step | Fix | File | Notes |
|------|-----|------|-------|
| 1 | FIX 1 | `contract_review_service.py` | P0 — fixes G6 fallback bug immediately |
| 2 | FIX 2 | `contract_review_service.py` | Same file — expand EXPECTED_CLAUSE_PATTERNS |
| 3 | FIX 3 | `contract_review_service.py` | Same file — expand IMPLICIT_* keywords |
| 4 | FIX 4 | `structured_clause_extraction.py` | Extraction layer gate fix |
| 5 | FIX 5 | `structured_clause_extraction.py` | Same file — include ANNEXURES_SCHEDULES |
| 6 | FIX 6 | `structured_clause_extraction.py` | Same file — MSA party definitions |
| 7 | FIX 7 | `clause_taxonomy.py` | Taxonomy enrichment — lowest urgency |

Steps 1–3 can be deployed immediately and will resolve the false negatives on the **currently stored MSA** without re-extraction.

Steps 4–6 require re-extraction of all MSA documents (clear ClauseStore entries for those document IDs) to take effect.

Step 7 improves future extraction quality and only affects documents extracted after deployment.

---

## 4. Verification Checklist

After implementing steps 1–3, run the contract review on the same MSA document and confirm:

| Clause | Expected `presence_map[*]["status"]` | Detecting keyword |
|--------|--------------------------------------|-------------------|
| `confidentiality` | `"detected"` | `"confidentiality"` |
| `dispute_resolution` | `"detected"` | `"arbitration"` or `"resolution of disputes"` |
| `ip_ownership` | `"detected"` | `"proprietary rights"` |
| `sla_obligations` | `"detected"` | `"liquidated damages"` or `"performance monitoring"` |

Also confirm:
- `not_detected_clauses` list in the response is empty for the MSA
- Executive Summary shows confirmations for all 4 clauses, not risk items
- No regression on employment or NDA profiles (run existing tests)

Add a debug log call before G6 to expose presence_map state before fallback:
```python
logger.debug(
    "contract_review: G6 fallback triggered for %s; still missing: %s",
    contract_id, _still_missing
)
```

---

## 5. Confidence Assessment

| Fix | Estimated contribution to resolving reported false negatives |
|-----|-------------------------------------------------------------|
| FIX 1 (fallback bug) | ~80% — most likely single root cause of all 4 false negatives |
| FIX 2 (pattern expansion) | ~15% additive — catches cases where fallback-rescued status still shows uncertain |
| FIX 3 (implicit keywords) | ~10% additive — strengthens G1.5/G4/G5 post-passes |
| FIX 4–6 (extraction) | ~100% for future MSA uploads; 0% for current run without re-extract |
| FIX 7 (taxonomy) | Prevents taxonomy misclassification in long term; 0% on existing reports |

**Combined FIX 1 + 2 + 3**: Expected to resolve the reported false negatives on the existing MSA document with no re-extraction needed.

---

## 6. Regression Safeguards

- `DOWNGRADE_IF_NON_CONTRACTUAL` should NOT be expanded — it is intentionally narrow (employment-specific clause types)
- The `WEAK_EVIDENCE_RULES` for `governing_law: []` (force "detected") should be preserved
- Do not add `sla_obligations` or `ip_ownership` to `CRITICAL_CLAUSES` — they are `medium` severity per the MSA profile
- The contract entry gate requires ≥2 pattern matches — confirm the gate logic before adding patterns that are too generic (e.g., `\bshall\b` alone would be too broad)
