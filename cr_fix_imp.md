# Plan: Contract Review Report — Full Improvements (All Contract Types)

## Context

The contract review pipeline produces a risk table, evidence blocks, and executive summary. There are two categories of improvements needed:

1. **Cross-profile parity fixes** — The ClauseType enum (13 values) is severely underspecified vs. 28+ clause types across the three profiles. NDA and employment contracts have the same taxonomy/detection gaps that were previously only fixed for MSA. These must be fixed in parallel across all profiles.

2. **Report quality upgrades** — New UX features: coverage matrix, risk score, keyword highlighting, contradiction detection, LLM uncertain-clause resolution, PDF improvements, jurisdiction statutory notes.

### Current State of Previously-Planned Fixes
- FP1 (_page_fallback_scan tuple crash): **ALREADY FIXED**
- FP2 (contract_entry_patterns MSA): **PARTIALLY FIXED** — basic MSA patterns present; NDA patterns entirely missing
- FP3 (ANNEXURES_SCHEDULES in operative_sections): **ALREADY FIXED**
- FP4 (_has_party_definitions): **NOT FIXED** — employment-only (`"first party"` + `"second party"`)

---

## Implementation Plan (9 groups, ordered by dependencies)

---

### Group 1 — Cross-Profile Clause Type Infrastructure

This is the largest group. All three profiles (employment, NDA, MSA) need parity across three layers: enum, taxonomy, and heading synonyms.

#### 1a. Profile-driven taxonomy (replaces enum expansion)

Instead of adding 19 hardcoded enum values, make the taxonomy service load valid clause types dynamically from the profile YAMLs. The ClauseType enum stays at its 13 values — it only needs members where code explicitly branches on specific types (none of the 19 missing ones do).

**Rationale**: The profiles (employment.yaml, nda.yaml, msa.yaml) are already the canonical source of all valid clause types. The enum was always just an internal extraction layer concept — not a complete registry. Profile-driven loading means: add a clause type to any profile YAML → taxonomy picks it up automatically, no code change ever needed again.

**`backend/services/clause_taxonomy.py`** — change to profile-aware keyword loading:

Add a module-level loader that reads all profiles at import time and builds the keyword dict from `EXPECTED_CLAUSE_PATTERNS` in `contract_review_service.py`. Since that would create a circular import, instead move `EXPECTED_CLAUSE_PATTERNS` (or a subset — the keyword lists) to a shared location: `backend/contract_profiles/clause_keywords.yaml`.

**New file**: `backend/contract_profiles/clause_keywords.yaml`

Extract the keyword lists from `EXPECTED_CLAUSE_PATTERNS` (contract_review_service.py lines 41-152) into this YAML. Example:
```yaml
ip_ownership:
  - intellectual property
  - proprietary rights
  - ip rights
  - work product
  - vests in
  - shall own
  - deliverables
  - work for hire
sla_obligations:
  - service level
  - sla
  - uptime
  - availability
  - response time
  - liquidated damages
  - kpi
# ... all other clause types
```

**`backend/services/clause_taxonomy.py`** — load from YAML at startup:
```python
import yaml
from pathlib import Path

def _load_clause_keywords() -> Dict[str, List[str]]:
    path = Path("backend/contract_profiles/clause_keywords.yaml")
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

_CLAUSE_KEYWORDS: Dict[str, List[str]] = _load_clause_keywords()
```

In `ClauseTaxonomyService.__init__`, replace the hardcoded `category_keywords` dict with a dynamically built one:
```python
# Build from YAML — covers all profile types without enum changes
self.string_type_keywords: Dict[str, List[str]] = _CLAUSE_KEYWORDS

# Existing enum-based dict kept for backward compat with ClauseType enum values
self.category_keywords = {
    ClauseType.GOVERNING_LAW: _CLAUSE_KEYWORDS.get("governing_law", [...]),
    ClauseType.CONFIDENTIALITY: _CLAUSE_KEYWORDS.get("confidentiality", [...]),
    # ... only for the 13 existing enum values
}
```

**`backend/services/clause_taxonomy.py`** — `classify()` method update:

Extend the existing classify logic to also check `string_type_keywords` for types not in the enum. When a match is found for a non-enum type (e.g., "ip_ownership"), store it as `ClauseType.OTHER` in the enum field but also set a `normalized_clause_type: str` attribute on the clause with the raw string value. This is already how `contract_review_service.py` reads it (`getattr(c, "normalized_clause_type", None)`) — so no downstream changes needed.

**`backend/models/clause.py`** — confirm `normalized_clause_type: Optional[str]` field exists on the Clause model. If not, add it. This field is already referenced in `contract_review_service.py` line 1436.

**`_LEGAL_CATEGORY_PRIORITY`** and **`_CATEGORY_SLUG_KEYWORDS`** in `clause_taxonomy.py`: These also need the profile types added. Since the YAML now drives the data, update these lists to include all keys present in `clause_keywords.yaml`. This can be done by deriving the priority list from the YAML keys in a defined order (profile precedence: governing_law > termination > liability > confidentiality > ... > other).

#### 1b. Keep `EXPECTED_CLAUSE_PATTERNS` in contract_review_service.py in sync
**File**: `backend/services/contract_review_service.py` lines 41-152

The existing `EXPECTED_CLAUSE_PATTERNS` dict stays as-is and remains the authoritative keyword source for detection. The new `clause_keywords.yaml` is derived from it (one-time extraction). Going forward, both must be kept in sync when clause types are added. Add a comment noting this.

#### 1c. Add NDA contract_entry_patterns
**File**: `backend/services/structured_clause_extraction.py` · Lines 206-220

Add 8 NDA-specific patterns to `self.contract_entry_patterns`:
```python
r"\bnon.?disclosure\s+agreement\b",
r"\bconfidentiality\s+agreement\b",
r"\bdisclosing\s+party\b",
r"\breceiving\s+party\b",
r"\brecipient\s+shall\b",
r"\bconfidential\s+information\s+shall\b",
r"\bunauthorized\s+disclosure\b",
r"\bobligation\s+of\s+confidentiality\b",
```

And add remaining missing general patterns:
```python
r"\bservice\s+agreement\b",
r"\bthe\s+parties\s+agree\b",
r"\bhereby\s+agree[sd]?\b",
r"\bin\s+witness\s+whereof\b",
r"\bnow[\s,]+therefore\b",
```

#### 1d. Fix `_has_party_definitions` — add NDA and MSA parties
**File**: `backend/services/structured_clause_extraction.py` · Lines 704-706

Replace employment-only implementation:
```python
def _has_party_definitions(self, text_lower: str) -> bool:
    """Detect party definitions across Employment, NDA, and MSA document types."""
    employment = "first party" in text_lower and "second party" in text_lower
    msa = any(t in text_lower for t in [
        "service provider", "the client", "the vendor", "licensor", "licensee",
        "purchaser", "the buyer", "the seller", "the supplier",
        "hereinafter referred to as", "hereinafter called",
    ])
    nda = any(t in text_lower for t in [
        "disclosing party", "receiving party", "the recipient",
        "the disclosing party", "the receiving party",
    ])
    return employment or msa or nda
```

#### 1e. Add cross-profile heading synonyms
**File**: `backend/services/contract_review_service.py` · `CLAUSE_HEADING_SYNONYMS` dict (Lines 157-175)

Add 17 new entries:

Employment headings (9):
```python
"remuneration":                   "salary_wages",
"leave entitlement":              "annual_leave",
"probationary period":            "probation_period",
"gratuity":                       "end_of_service_gratuity",
"end of service benefit":         "end_of_service_gratuity",
"notice period":                  "notice",
"salary and wages":               "salary_wages",
"compensation and benefits":      "compensation",
"conduct and discipline":         "conduct_discipline",
```

NDA headings (8):
```python
"confidentiality agreement":      "confidentiality",
"non-disclosure agreement":       "confidentiality",
"confidentiality and non-disclosure": "confidentiality",
"term of agreement":              "term",
"remedies for breach":            "remedies",
"injunctive relief":              "remedies_injunction",
"return of confidential information": "return_of_information",
"carve-outs":                     "carve_outs",
"mutual confidentiality":         "mutual_vs_unilateral",
```

---

### Group 2 — Overall Risk Score

**Purpose**: Single at-a-glance health label for the top of the report.

#### 2a. Backend — add fields and compute
**File**: `backend/models/contract_review.py`

Add to `ContractReviewResponse`:
```python
risk_score: int = Field(default=0, description="high*3 + medium*1.")
risk_label: Literal["high_risk", "medium_risk", "low_risk"] = Field(default="low_risk")
```

**File**: `backend/services/contract_review_service.py` · `run()`, before building response (step 10, ~line 1902)

```python
high_count = sum(1 for r in risks if r.severity == "high")
medium_count = sum(1 for r in risks if r.severity == "medium")
risk_score = high_count * 3 + medium_count
risk_label = "high_risk" if high_count >= 2 else ("medium_risk" if risk_score >= 2 else "low_risk")
```

#### 2b. Frontend — show 4 metrics below header
**File**: `frontend/app.py` · `_render_contract_review_details()` after meta_cols block (~line 555)

Four `st.metric()` widgets in a row: 🔴 High, 🟡 Medium, 🟢 Low, Overall label.

---

### Group 3 — Clause Coverage Matrix

**Purpose**: Replace the flat not_detected list and separate implicitly_covered section with a single visual grid.

**File**: `frontend/app.py` · `_render_contract_review_details()` (replace lines 612-628)

Build a matrix from executive_summary items (already in response — no API change):
- Columns: `Clause | Status | Severity`
- Status icons: ✅ Detected · ⚠️ Weak/Implicit/Distributed · ❌ Not Detected
- Color highlight rows by severity using `st.dataframe` with column config

Keep per-risk expanders below for drill-down. Keep separate "Implicitly Covered" and "Not Detected" text sections collapsed inside an `st.expander("Details")`.

---

### Group 4 — Keyword Highlighting in Verbatim Snippets

#### 4a. Backend — surface matched_keyword in VerbatimSnippet
**File**: `backend/models/contract_review.py` · `VerbatimSnippet`

Add field:
```python
matched_keyword: Optional[str] = Field(default=None)
```

**File**: `backend/services/contract_review_service.py`

- Build `clause_matched_keywords: Dict[str, str]` from `presence_map` (clause_type → matched_keyword) before calling `_attach_verbatim_evidence()`
- Pass it to the method and include in each `VerbatimSnippet` construction

#### 4b. Frontend — bold the keyword in snippet text
**File**: `frontend/app.py` · per-risk expanders (~lines 597-610)

Case-insensitive wrap of `matched_keyword` in `**...**` before rendering.

---

### Group 5 — Cross-Clause Contradiction Detection

**Purpose**: Rule-based check for numeric contradictions between detected clauses. No LLM.

**File**: `backend/services/contract_review_service.py`

Add new method `_identify_contradiction_risks()`:
- **Notice period mismatch**: Extract `(\d+)\s*(days?|months?|years?)` from termination and notice clause evidence candidates; flag if values differ by >50%
- **Confidentiality perpetuity vs return_of_information**: If confidentiality evidence contains "perpetuity" or "forever" AND return_of_information is detected, surface as low-severity finding
- Only flag when both clauses are "detected" (not just suspected)
- Returns `List[RiskItem]` with `missing_clause=False`, `severity="low"`, `clause_types=[both types]`

Add to `ContractReviewResponse`:
```python
contradiction_risks: List[RiskItem] = Field(default_factory=list)
```

Hook into `run()` after step 3c (~line 1796), gated by `review_depth != "quick"`.

All generated description text must pass through `enforce_non_prescriptive_language()`.

---

### Group 6 — LLM Pass for Uncertain Clauses (G8)

**Purpose**: Binary LLM confirmation for clauses stuck at `status="uncertain"`. Reduces false positives and false negatives without running LLM on all clauses.

**Ollama call pattern** (reuse from `rag_service.py`):
```python
self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
response = self.ollama_client.generate(
    model=settings.OLLAMA_MODEL,
    prompt=prompt,
    options={"temperature": 0.0, "seed": settings.CONTRACT_REVIEW_SEED}
)
answer = (response.get("response") or "").strip().lower()
```

**File**: `backend/services/contract_review_service.py`

New method `_resolve_uncertain_with_llm(clause_type, evidence_text, display_name) -> str`:
- Prompt: `"Does the following text contain a {display_name} clause? Reply only: YES or NO.\n\n{evidence_text[:600]}"`
- Parse: `"yes"` in answer → `"detected"`, else `"not_detected"`
- Entire method wrapped in `try/except` — any failure returns `"uncertain"` (leaves status unchanged)

Insert as **G8 pass** after G7 heading-synonym pass (~line 1713), before risk row generation:
```python
if review_depth == "standard":
    for clause_type in expected:
        if presence_map.get(clause_type, {}).get("status") == "uncertain":
            ev_text = next(
                (c.get("text", "") for c in evidence_candidates
                 if c.get("clause_id") == presence_map[clause_type].get("clause_ids", [None])[0]),
                ""
            )
            disp = EXPECTED_CLAUSE_DISPLAY_NAMES.get(clause_type, clause_type)
            resolved = self._resolve_uncertain_with_llm(clause_type, ev_text, disp)
            if resolved != "uncertain":
                presence_map[clause_type]["status"] = resolved
                presence_map[clause_type]["display_status"] = _display_status_for(resolved)
```

**Note on latency**: Each uncertain clause adds ~1-3s. Gate on `standard` depth prevents impact on `quick` reviews. In practice, few clauses are uncertain per document.

---

### Group 7 — Jurisdiction-Specific Statutory Notes

**Purpose**: Surface GCC statutory minimums as reviewer reference (not legal advice), sourced from actual legal documents — zero runtime overhead via pre-extraction.

#### Approach: Pre-extracted YAML (hybrid of local PDF + static lookup)

**One-time offline setup** (done by developer, not at review time):
1. Download official GCC labor law PDFs → store in `backend/legal_references/` (versioned in repo)
2. Run a one-time extraction script to pull the relevant articles → output to `backend/legal_references/jurisdiction_statutes.yaml`
3. The YAML is committed to the repo and loaded at service startup

**Sources to download**:
- KSA: Saudi Labor Law (Royal Decree M/46, 2005 + 2015 amendments)
- UAE: UAE Federal Decree-Law No. 33 of 2021
- Qatar: Qatar Labor Law No. 14 of 2004
- Bahrain: Bahrain Labor Law No. 36 of 2012
- Oman: Oman Labor Law (Royal Decree 35/2003)
- Kuwait: Kuwait Private Sector Labor Law No. 6 of 2010

**`backend/legal_references/jurisdiction_statutes.yaml`** structure:
```yaml
KSA:
  notice:
    article: "Article 75"
    text: "The employment contract shall not be terminated by the employer without a notice period of no less than sixty days..."
    source: "Saudi Labor Law, Royal Decree M/46"
  end_of_service_gratuity:
    article: "Article 84"
    text: "Upon termination of the employment contract, the employer shall pay the worker an end-of-service award..."
    source: "Saudi Labor Law, Royal Decree M/46"
  annual_leave:
    article: "Article 109"
    text: "The worker shall be entitled to an annual leave of no less than twenty-one days per year..."
    source: "Saudi Labor Law, Royal Decree M/46"
UAE:
  notice:
    article: "Article 43"
    text: "Either party may terminate an unlimited contract by providing the other party with written notice..."
    source: "UAE Federal Decree-Law No. 33 of 2021"
  # ... etc.
```

#### Backend changes

**File**: `backend/services/contract_review_service.py` · `__init__`

Load YAML once at startup (same pattern as `risk_explanations.yaml`):
```python
self.jurisdiction_statutes: Dict[str, Any] = self._load_jurisdiction_statutes()

def _load_jurisdiction_statutes(self) -> Dict[str, Any]:
    path = Path("backend/legal_references/jurisdiction_statutes.yaml")
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logging.warning("Failed to load jurisdiction_statutes.yaml: %s", exc)
    return {}
```

**File**: `backend/models/contract_review.py` · `ContractReviewResponse`

Add field:
```python
statutory_notes: Optional[Dict[str, Any]] = Field(
    default=None,
    description="Jurisdiction-specific statutory article references keyed by clause display name. Each value has {article, text, source}."
)
```

**File**: `backend/services/contract_review_service.py` · `run()` after jurisdiction resolution

```python
statutory_notes: Optional[Dict[str, Any]] = None
if jurisdiction and self.jurisdiction_statutes:
    canon = _canon_jurisdiction(jurisdiction)[0]
    jur_data = self.jurisdiction_statutes.get(canon) or {}
    if jur_data:
        statutory_notes = {
            EXPECTED_CLAUSE_DISPLAY_NAMES.get(k, k): v
            for k, v in jur_data.items()
            if k in expected
        }
```

**Runtime overhead**: Zero — dict lookup only. YAML loaded once at startup.

#### Frontend: display statutory notes
**File**: `frontend/app.py`

In `_render_contract_review_details()` and `_generate_pdf_buffer()`:
- Inline `st.caption()` per clause row in coverage matrix: `📜 {article}: "{text}" — {source}`
- In PDF: a dedicated "Statutory References" section at the end, one paragraph per note

---

### Group 8 — PDF Improvements (Text Alignment & Layout)

**File**: `frontend/app.py` · `_generate_pdf_buffer()` (Lines 699-814)

#### Root problem with current PDF
The current code puts **raw strings** in table cells. Reportlab does NOT wrap plain strings — the `WORDWRAP` style command only works with `Paragraph` objects inside cells. This causes overflow and clipping. Additionally, the current column widths sum to **18cm** but printable width on A4 with 2cm margins is only **17cm** — the table is already 1cm too wide before content is even considered.

#### 8a. Define reusable cell paragraph styles (add at top of function)
```python
cell_style = ParagraphStyle(
    "cell", parent=normal,
    fontSize=7, leading=9,    # leading=line height; 1.3× font keeps rows compact
    wordWrap="LTR",
    spaceAfter=0, spaceBefore=0,
)
header_cell_style = ParagraphStyle(
    "header_cell", parent=cell_style,
    textColor=colors.whitesmoke, fontName="Helvetica-Bold",
)
```

#### 8b. Fix column widths to exactly 17cm
Replace line 766 (current 18cm total):
```python
# Severity | Description | Status | Clauses | Pages | Recommendation
COL_WIDTHS = [1.5*cm, 5.5*cm, 3.0*cm, 2.5*cm, 1.0*cm, 3.5*cm]  # = 17cm
tbl = Table(table_data, repeatRows=1, colWidths=COL_WIDTHS)
```

#### 8c. Wrap ALL table cell text in Paragraph objects (replace lines 752-765)
Remove hard truncations (`:60`, `:80`). Paragraph objects handle layout via column width.
```python
table_data = [[
    Paragraph("Severity", header_cell_style),
    Paragraph("Description", header_cell_style),
    Paragraph("Status", header_cell_style),
    Paragraph("Clauses", header_cell_style),
    Paragraph("Pages", header_cell_style),
    Paragraph("Recommendation", header_cell_style),
]]
for r in risks:
    if not isinstance(r, dict):
        continue
    display_names = r.get("display_names", []) or r.get("clause_ids", []) or []
    pages = r.get("page_numbers", []) or []
    table_data.append([
        Paragraph(r.get("severity", "").capitalize(), cell_style),
        Paragraph(r.get("description", ""), cell_style),        # no truncation
        Paragraph(r.get("status", "").replace("_", " ").title(), cell_style),
        Paragraph(", ".join(display_names[:3]), cell_style),
        Paragraph(", ".join(str(p) for p in pages[:5]), cell_style),
        Paragraph(r.get("recommendation") or "", cell_style),   # no truncation
    ])
```

#### 8d. Color-code rows by severity + improved TableStyle (replace lines 767-774)
```python
row_colors = []
for i, r in enumerate(risks, start=1):
    if not isinstance(r, dict):
        continue
    bg = (
        colors.HexColor("#FFD5D5") if r.get("severity") == "high" else
        colors.HexColor("#FFF3CD") if r.get("severity") == "medium" else
        colors.HexColor("#D4EDDA")
    )
    row_colors.append(("BACKGROUND", (0, i), (-1, i), bg))

tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 7),
    ("LEADING", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    *row_colors,
]))
```

#### 8e. Fix evidence excerpts — split header from body (replace lines 790-798)
```python
ev_header_style = ParagraphStyle("ev_h", parent=normal,
    fontSize=7, fontName="Helvetica-Bold", leftIndent=12, spaceAfter=1)
ev_body_style = ParagraphStyle("ev_b", parent=normal,
    fontSize=7, leftIndent=12, leading=9,
    textColor=colors.HexColor("#1A3C6B"), spaceAfter=4)
for snippet in snippets:
    s_name = snippet.get("display_name") or snippet.get("clause_id", "")
    s_page = snippet.get("page_number", "")
    s_text = snippet.get("text", "")      # no truncation — Paragraph wraps
    story.append(Paragraph(f"{s_name} — Page {s_page}", ev_header_style))
    story.append(Paragraph(f'"{s_text}"', ev_body_style))
```

#### 8f. New: Clause coverage table (add after Executive Summary, before Risk Analysis)
3-column table at 17cm total: `[9.5cm | 4.5cm | 3.0cm]`
```python
coverage_data = [[
    Paragraph("Clause", header_cell_style),
    Paragraph("Status", header_cell_style),
    Paragraph("Severity", header_cell_style),
]]
for item in exec_items:
    if not isinstance(item, dict):
        continue
    cat = item.get("category", "")
    icon = "Confirmed" if cat == "confirmation" else ("Finding" if cat == "finding" else "Risk")
    coverage_data.append([
        Paragraph(item.get("text", ""), cell_style),   # wraps inside 9.5cm column
        Paragraph(icon, cell_style),
        Paragraph((item.get("severity") or "").capitalize() or "—", cell_style),
    ])
cov_tbl = Table(coverage_data, repeatRows=1, colWidths=[9.5*cm, 4.5*cm, 3.0*cm])
# Apply same TableStyle pattern (header + grid + valign + padding)
```

#### 8g. New: Statutory References section (at end, before Disclaimer)
If `statutory_notes` in response — 2-column table at 17cm: `[4.5cm | 12.5cm]`
Each cell: `Paragraph(text, cell_style)` — no overflow possible.

#### 8h. Table of contents (after title metadata, before Executive Summary)
Simple paragraph list — no page numbers (SimpleDocTemplate doesn't auto-generate them):
```python
toc_items = ["Executive Summary", "Clause Coverage", "Risk Analysis",
             "Evidence Excerpts", "Clauses Not Detected"]
if statutory_notes:
    toc_items.append("Statutory References")
for name in toc_items:
    story.append(Paragraph(f"• {name}", toc_style))
```

---

## File Change Summary

| File | Group(s) | Nature |
|------|----------|--------|
| `backend/models/clause.py` | 1a | Confirm `normalized_clause_type: Optional[str]` field exists on Clause model |
| `backend/contract_profiles/clause_keywords.yaml` | 1a | New file — extracted keyword lists for all clause types across all profiles |
| `backend/services/clause_taxonomy.py` | 1a, 1b | Profile-driven keyword loading; extend priority list + slug keywords + category_keywords for all profile types |
| `backend/services/structured_clause_extraction.py` | 1c, 1d | Add NDA + general contract_entry_patterns; rewrite `_has_party_definitions` |
| `backend/services/contract_review_service.py` | 1e, 2a, 4a, 5, 6, 7 | Add heading synonyms; risk score computation; contradiction method; LLM G8 pass; statutory notes; keyword in VerbatimSnippet |
| `backend/models/contract_review.py` | 2a, 4a, 5, 7 | Add `risk_score`, `risk_label`, `matched_keyword`, `contradiction_risks`, `statutory_notes` |
| `frontend/app.py` | 2b, 3, 4b, 7, 8 | Severity metrics; coverage matrix; keyword highlighting; statutory notes display; PDF improvements |

---

## Implementation Order

| Step | Group | Reason |
|------|-------|--------|
| 1 | 1a (`clause.py`) | Enum must exist before taxonomy references it |
| 2 | 1b (`clause_taxonomy.py`) | Depends on Step 1 |
| 3 | 1c + 1d (`structured_clause_extraction.py`) | Independent; can run in parallel with Step 2 |
| 4 | 1e (`contract_review_service.py` heading synonyms) | Independent; single dict addition |
| 5 | 2a + 4a + 5 + 7 (model + service, non-LLM) | Batch all `contract_review.py` model changes; batch all non-LLM `contract_review_service.py` changes |
| 6 | 6 (LLM G8 pass in service) | Last service change; after other passes are stable |
| 7 | 2b + 3 + 4b + 7 (frontend) | After API response shape is finalized |
| 8 | 8 (PDF) | Frontend-only; last |

---

## Verification Checklist

1. **Employment contract review**: confirm `probation_period`, `end_of_service_gratuity`, `conduct_discipline` no longer show spurious "Not Detected" after taxonomy fix
2. **NDA contract review**: confirm `carve_outs`, `mutual_vs_unilateral`, `return_of_information` clauses now classify correctly; confirm NDA document type triggers extraction gate
3. **MSA contract review**: confirm `ip_ownership`, `sla_obligations` no longer appear in `not_detected_clauses`
4. **Jurisdiction = KSA + employment contract**: confirm `statutory_notes` in response contains note for `end_of_service_gratuity`
5. **Uncertain clause → LLM G8**: artificially force a clause to uncertain (or find a real document that produces one); confirm G8 resolves it
6. **Contradiction detection**: test a document with termination/notice numeric mismatch; confirm `contradiction_risks` list is non-empty
7. **Frontend coverage matrix**: renders for all three profile types; ✅/⚠️/❌ icons correct
8. **PDF**: open downloaded PDF — confirm color rows visible, coverage table present
9. **Regression**: run employment + NDA + MSA reviews; confirm no previously-detected clauses are now lost

---

## Notes

- All new generated text (contradiction descriptions, statutory note text) must pass through `enforce_non_prescriptive_language()` before inclusion in response
- `JURISDICTION_STATUTORY_NOTES` values are static reference strings, not legal conclusions. Mark them with a `[Reference only]` prefix or surface separately from risk items
- The LLM G8 resolver adds ~1-3s per uncertain clause. This is acceptable for `standard` depth; `quick` depth bypasses it
- All new `ContractReviewResponse` fields have default values → backward compatible with existing consumers
- After deploying extraction fixes (Groups 1a-1d), re-extract all existing documents to rebuild ClauseStore with corrected taxonomy
