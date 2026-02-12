# Product Requirements Document (PRD) — Main Branch

**Product:** Legal Document Intelligence (LLM-powered pipeline)  
**Branch:** `main` (primary/default)  
**Last updated:** February 2025

---

## 1. Executive Summary

The **main branch** delivers a focused, workflow-first Legal Document Intelligence experience. The UI exposes **four areas only**: **Contract Review**, **Document Explorer**, **Upload Document**, and **About**. The backend retains the full API surface (search, clause extraction, comparison, summarization, due diligence, bilingual search, etc.) for programmatic or future use, but the **default user journey on main is intentionally minimal**: upload documents, run contract review, or explore a document (evidence and/or answer).

---

## 2. Main Branch Scope

| Layer | Scope |
|-------|--------|
| **UI (Streamlit)** | Contract Review, Document Explorer, Upload Document, About. No sidebar entries for Clause Extraction, Contract Comparison, or Due Diligence Memo. |
| **Backend (FastAPI)** | All APIs remain implemented and callable (e.g. `/api/compare`, `/api/summarize`, `/api/due-diligence-memo`, `/api/extract-clauses`, `/api/search-bilingual`). |
| **Purpose** | Simplify the demo and primary user path; power users and integrations can still call other endpoints directly. |

---

## 3. User-Facing Features (Main Branch)

### 3.1 Upload Document

- Upload PDF, DOCX, or DOC as document or template.
- Deterministic content-hash document ID; chunking with page tracking; optional OCR for scanned PDFs.
- Documents appear in document list for Contract Review and Document Explorer.

### 3.2 Contract Review (workflow)

- **Page:** “📋 Contract Review”
- **Input:** Select contract, contract type (employment, nda, msa), jurisdiction (optional), review depth (standard/quick).
- **Action:** Run Contract Review → WorkflowOrchestrator → ContractReviewService.
- **Output:** Risks, missing clauses, evidence blocks per clause, executive summary; document classification warning if applicable. Guardrails and disclaimer apply.

### 3.3 Document Explorer

- **Page:** “🔍 Document Explorer”
- **Evidence tab:** Query + mode (both / text / clauses). Calls `POST /api/explore-evidence`. Verbatim snippets only (no LLM). Modes: clauses first then text, or text-only, or clauses-only.
- **Answer tab:** Query (+ optional response language). Calls `POST /api/explore-answer`. RAG search → LLM → answer with sources and citations.
- **Behavior:** Evidence Explorer is recall-first (lexical/heading eligible; no merge/summarize/stitch). RAG Answer may not return “not_specified” unless Evidence would return “not_found”.

### 3.4 About

- **Page:** “ℹ️ About”
- Product description, disclaimer, and quick start (upload → contract review or document explorer). References to other capabilities (e.g. clause extraction) may appear as documentation but are not in the main navigation.

---

## 4. Backend APIs (Available on Main)

All of the following exist on main; only a subset are surfaced in the UI.

| API | Method | UI on main |
|-----|--------|------------|
| Health | `GET /api/health` | — |
| Upload | `POST /api/upload` | ✅ Upload Document |
| Search (RAG) | `POST /api/search` | — |
| Search bilingual | `POST /api/search-bilingual` | — |
| Extract clauses | `POST /api/extract-clauses` | — |
| Get clauses by doc | `GET /api/clauses/{document_id}` | — |
| Get single clause | `GET /api/clauses/single/{clause_id}` | — |
| Query clauses | `POST /api/clauses/query` | — |
| Compare | `POST /api/compare` | — |
| Contract Review | `POST /api/contract-review` | ✅ Contract Review |
| Document Explorer (legacy) | `POST /api/explore` | — (Answer/Evidence used instead) |
| Explore Evidence | `POST /api/explore-evidence` | ✅ Document Explorer (Evidence) |
| Explore Answer | `POST /api/explore-answer` | ✅ Document Explorer (Answer) |
| Summarize | `POST /api/summarize` | — |
| Due Diligence Memo | `POST /api/due-diligence-memo` | — |
| Summarize (stream) | `POST /api/summarize/stream` | — |
| List documents | `GET /api/documents` | ✅ (used by Contract Review & Explorer) |
| Rename document | `PUT /api/documents/{document_id}/rename` | — |
| Stats | `GET /api/stats` | — |

---

## 4.1 RAG Page Invariant (Safety Rail)

The following invariant is enforced across ingestion, retrieval, and answer generation:

- **Every vector represents a chunk from exactly one page.** Chunking is performed per page; no chunk may span pages.
- **Every answer cites page numbers**, not just documents. Citations must include the page (e.g. "Employment Contract, Page 7").
- **No reasoning step may merge content across pages silently.** Legal conclusions are page-local unless explicitly marked as cross-page inference.

---

## 5. Product Goals (Main Branch)

- **Focused UX:** Primary workflows are Contract Review and Document Explorer; no clutter from comparison, due diligence, or clause extraction in the nav.
- **Auditability:** Evidence and answers traceable to chunks/clauses and pages.
- **Recall-first evidence:** Evidence Explorer rules (lexical eligible, no score cutoff for eligible results, no merge/summarize/stitch) apply.
- **Local & private:** Backend + optional Ollama; no training on client data.
- **Demo-ready:** Run via `run_backend.py` and `streamlit run frontend/app.py`.

---

## 6. Out of Scope (Main Branch)

- Legal advice or outcome prediction.
- Contract drafting.
- Internet or external content APIs.
- Multi-tenant/enterprise guarantees.
- Production SLA.

---

## 7. Success Criteria (Main Branch)

- Sidebar shows exactly: Contract Review, Document Explorer, Upload Document, About.
- Contract Review runs end-to-end and returns structured results in WorkflowContext.
- Document Explorer Evidence (text/clauses/both) and Answer work per DOCUMENT_EXPLORER_WORKFLOW and EVIDENCE_EXPLORER_SPEC.
- Upload stores documents and makes them available to Contract Review and Document Explorer.
- All backend endpoints listed above respond as designed (whether or not used by the UI).

---

## 8. References

- **Workflow & data flow:** `docs/DOCUMENT_EXPLORER_WORKFLOW.md`
- **Evidence & RAG rules:** `EVIDENCE_EXPLORER_SPEC.md`
- **Full feature set (e.g. v0.2):** `docs/PRD_CURRENT_STAGE.md`
- **Setup & run:** `README.md`, `START_COMPLETE_SYSTEM.md`

---

*This PRD describes the **main** branch: reduced UI, full backend. For the expanded UI (Clause Extraction, Contract Comparison, Due Diligence Memo in nav), see `docs/PRD_CURRENT_STAGE.md` or the v0.2 branch.*
