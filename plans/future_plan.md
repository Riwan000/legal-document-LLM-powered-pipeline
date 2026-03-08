# Future Feature Plan
**Last updated:** 2026-03-05

---

## Global Features

### Contract & Document Automation

| Feature | Description |
|---|---|
| **Contract Drafting Assistant** | Generate first-draft clauses from a prompt (e.g. "draft a limitation of liability clause for a SaaS MSA") |
| **Redline / Negotiation Tracker** | Side-by-side diff between two contract versions with risk-delta scoring; track which party made each change |
| **Playbook Enforcement** | Upload a firm's negotiation playbook; flag deviations in third-party paper and suggest fallback positions |
| **Clause Bank / Precedent Library** | Store approved clause variants; AI retrieves the most appropriate precedent for a given context |
| **Contract Obligation Tracker** | Extract party obligations, deadlines, and renewal dates; generate a calendar/reminder feed (iCal export) |
| **Auto-Redact for Sharing** | One-click PII/confidential-term redaction before external sharing |

### Litigation & Case Intelligence

| Feature | Description |
|---|---|
| **Case File Summarization** _(unfreeze existing)_ | Backend is complete — only needs a new UI page wired to `/api/summarize` and `/api/due-diligence-memo` |
| **Chronology Builder** | Auto-extract dated events across a case bundle into a sortable timeline |
| **Witness Statement Analyzer** | Cross-reference multiple witness statements; flag contradictions and gaps |
| **Opposing Argument Predictor** | Given claimant arguments, generate likely defence responses for trial prep |
| **Bundle Deduplication** | Detect near-duplicate documents across large litigation bundles |
| **Legal Research Memos** | Auto-generate jurisdiction-specific research memos from uploaded statute/case law |

### Compliance & Risk

| Feature | Description |
|---|---|
| **Regulatory Change Alerts** | Monitor published legislation feeds; flag uploaded contracts affected by a new law |
| **GDPR / Data Protection Audit** | Scan contracts for data processing clauses; flag missing DPA provisions, sub-processor chains, transfer mechanisms |
| **ESG Clause Scanner** | Flag presence/absence of sustainability, anti-slavery, carbon-offset, and supplier code-of-conduct clauses |
| **Sanctions & AML Screening** | Screen party names and jurisdictions in contracts against sanctions lists (OFAC, EU, UN) |
| **Limitation Period Calculator** | Parse dates + governing law + cause of action; compute when claims expire |

### Due Diligence & Transactions

| Feature | Description |
|---|---|
| **M&A Due Diligence Room** | Bulk ingest a VDR; auto-tag documents by type; generate due diligence report with risk matrix |
| **IP Schedule Extractor** | Pull IP assignment, licensing, and ownership schedules from deal documents |
| **Change of Control Trigger Detector** | Identify contracts with change-of-control clauses triggered by an acquisition |
| **Lease Abstraction** | Structured extraction of rent, break clauses, service charges, and tenant obligations from commercial leases |

### Productivity & Workflow

| Feature | Description |
|---|---|
| **Matter Management Integration** | Export review results / sessions to Clio, NetDocuments, or iManage via API webhooks |
| **E-Signature Workflow** | Trigger DocuSign / Adobe Sign envelopes directly from the review UI after approval |
| **Client Portal** | Secure document upload link for clients; they upload, system ingests, lawyer gets notification |
| **Audit Trail & Version History** | Per-document log of who ran what analysis, when, and what the outputs were |
| **Multi-Document Q&A** | Ask a question across an entire matter folder — portfolio-level RAG, not just one document |
| **Voice Dictation Input** | Dictate queries on mobile for field use (court attendance, site inspections) |

---

## GCC-Specific Features

### Regulatory Compliance

| Feature | Description |
|---|---|
| **Saudization / Nitaqat Compliance Checker** _(KSA)_ | Scan employment contracts for Saudization quota obligations; flag non-compliant staffing provisions |
| **Emiratisation Tracker** _(UAE)_ | Verify employment contracts reference correct Emiratisation targets and NAFIS scheme provisions |
| **MOHRE / MOL Compliance** _(UAE/KSA)_ | Flag clauses that contradict Ministry of Human Resources standard contract terms required for work permit approval |
| **WPS / Wage Protection System Clause** _(UAE/KSA/Qatar)_ | Detect missing or non-compliant wage payment method clauses required under WPS regulations |
| **Saudi PDPL Compliance Scanner** _(KSA)_ | Check contracts for data processing terms against Saudi Personal Data Protection Law (2021, effective 2023) |
| **UAE PDPL Compliance Scanner** _(UAE)_ | Same for UAE Federal Decree-Law No. 45 of 2021 on Personal Data Protection |

### Islamic Finance & Sharia

| Feature | Description |
|---|---|
| **Sharia Compliance Checker** | Flag riba (interest), gharar (uncertainty), and maysir (speculation) language; flag missing Sharia board approval references |
| **Islamic Finance Contract Templates** | Murabaha, Ijara, Musharaka, Mudaraba, Sukuk — structured extraction and risk review for each instrument type |
| **Waqf Document Analyzer** _(KSA/UAE)_ | Parse and summarize Waqf (Islamic endowment) deeds; extract beneficiary conditions and trustee obligations |
| **Zakat & VAT Clause Detector** | Flag missing or incorrect VAT / Zakat indemnity provisions in commercial contracts |

### Court System & Arbitration

| Feature | Description |
|---|---|
| **DIFC / ADGM Law Profile** _(UAE)_ | Dedicated contract profile for DIFC and ADGM courts — distinct from UAE civil law; flag choice-of-law issues |
| **Arbitration Clause Analyzer** | Classify arbitration clauses by seat (DIAC, BCDR, ICC-MENA, ADCCAC, QICCA) and flag non-standard or pathological clauses |
| **Arabic Court Pleading Formatter** | Format case summaries and submissions to match Arabic court document standards (Saudi MOJ, UAE courts) |
| **Notarization Requirement Flagging** | Identify contract types that require notarization under GCC law (POAs, property transfers, company resolutions) |
| **Saudi Enforcement Court Compatibility** | Check arbitration awards for enforceability criteria under Saudi Arbitration Law (Royal Decree M/34, 2012) |

### Corporate & Commercial

| Feature | Description |
|---|---|
| **Foreign Ownership Compliance** _(KSA/UAE)_ | Flag ownership structure provisions against updated foreign investment rules (100% ownership now permitted in many sectors) |
| **Commercial Agency Law Checker** _(GCC-wide)_ | Flag exclusive agency / distribution agreements for compliance with GCC commercial agency laws |
| **Giga-Project / NEOM Contract Profile** _(KSA)_ | Tailored clause taxonomy for mega-infrastructure projects; custom risk weights for delay, force majeure, government step-in rights |
| **SAGIA / MISA License Reference Extractor** _(KSA)_ | Pull investment license numbers, permitted activities, and conditions from commercial contracts |
| **Free Zone vs. Mainland Classifier** _(UAE)_ | Detect whether a contract is governed by free zone or mainland law; flag conflicts (JAFZA, DAFZA, KIZAD, etc.) |

### Labour & Employment

| Feature | Description |
|---|---|
| **End-of-Service Gratuity Calculator** | Auto-compute EOSB entitlement from contract dates and salary — surfaced alongside the contract review |
| **Kafala / Sponsorship Clause Detector** _(GCC-wide)_ | Flag legacy kafala provisions that conflict with updated mobility rights (Qatar 2020, UAE 2021 reforms) |
| **Non-Compete Enforceability Checker** | GCC non-competes have specific enforceability rules per jurisdiction; flag clauses exceeding permitted scope/duration |
| **Domestic Worker Contract Compliance** _(UAE/KSA)_ | Dedicated profile for domestic worker contracts under UAE Federal Law No. 10/2017 and KSA equivalents |

### Arabic Language

| Feature | Description |
|---|---|
| **Arabic-First Contract Drafting** | Generate contract clauses directly in Arabic (not translated from English) using Arabic legal register |
| **Bilingual Contract Validator** | For bilingual (Arabic + English) contracts, detect inconsistencies between the two versions — Arabic typically governs in GCC |
| **Arabic Legal Citation Extractor** | Extract and normalize Arabic Hijri dates, Royal Decree references, and Ministerial Decision numbers |
| **Arabic Diacritics Normalizer** | Strip tashkeel for consistent search across formal/informal Arabic legal text |

---

## Priority Matrix by Market Segment

| Segment | Top Features |
|---|---|
| **Law firms / Legal departments (GCC)** | Sharia compliance checker, DIFC/ADGM profile, arbitration clause analyzer, bilingual validator, EOSB calculator |
| **HR / In-house (GCC)** | Saudization/Emiratisation tracker, MOHRE compliance, WPS clause checker, non-compete enforceability, kafala detector |
| **Banks / Islamic Finance** | Sharia compliance, Islamic finance templates, Zakat/VAT clause detector, Sukuk documentation extractor |
| **M&A / Transactions (Global)** | Due diligence room, change-of-control detector, IP schedule extractor, sanctions screening |
| **Enterprise / Compliance (Global)** | GDPR/PDPL audit, ESG clause scanner, regulatory change alerts, obligation tracker with calendar export |

---

## Immediate Next Step (Low Effort, High Value)

- **Unfreeze Case Analysis** — backend is 100% complete; only a new Streamlit sidebar page is needed.
  - Add `"📋 Case Analysis"` to the sidebar `selectbox` in `frontend/app.py`
  - Wire the existing `summarize_case_file()` and `due_diligence_memo()` helper functions to a UI
  - Estimated effort: 1–2 days
