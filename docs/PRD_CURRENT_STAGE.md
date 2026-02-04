# Product Requirements Document (PRD) — Current Stage

**Product:** Legal Document Intelligence (LLM-powered pipeline)  
**Stage:** v0.2 — Demo-ready MVP with workflow orchestration  
**Last updated:** February 2025

---

## 1. Executive Summary

Legal Document Intelligence is a **private, local, demo-ready** system for legal document analysis. It combines RAG (Retrieval-Augmented Generation), clause extraction, contract comparison, case summarization, and orchestrated workflows (Contract Review, Due Diligence Memo, Document Explorer). Documents are processed and retrieved at query time; the system is designed **not** to train on client data.

**Current stage scope:** Feature-complete MVP with Evidence Explorer, RAG Answer Explorer, Contract Review, Due Diligence Memo, Contract Comparison, Clause Extraction, Case Summarization, and Bilingual Search — all exposed via FastAPI and a Streamlit “Legal Workflows Engine” UI.

---

## 2. Product Vision & Goals

| Goal | Description |
|------|--------------|
| **Auditability** | Evidence and answers are traceable to chunks/clauses and page numbers. |
| **Recall-first evidence** | Evidence Explorer prioritizes recall over semantic score cutoffs; lexical and heading matches are always eligible. |
| **Workflow-first UX** | Users choose workflows (Contract Review, Document Explorer, etc.) rather than raw API actions. |
| **Local & private** | Backend + optional Ollama run locally; no document data sent to external LLM APIs for training. |
| **Demo-ready** | Single-tenant, runnable via `run_backend.py` + Streamlit; not positioned as production legal advice. |

---

## 3. User Personas & Use Cases

| Persona | Primary use case |
|---------|------------------|
| **Legal / contract reviewer** | Run Contract Review on an uploaded contract (employment/NDA/MSA); get risks, missing clauses, evidence blocks, executive summary. |
| **Due diligence analyst** | Run Due Diligence Memo on a case/transaction document; get readiness, document type, section-level memos, case spine, citations. |
| **Researcher / associate** | Use Document Explorer: Evidence tab for verbatim snippets (text/clauses/both); Answer tab for RAG-generated answers with sources. |
| **Contract drafter / comparer** | Upload template + contract; run Contract Comparison; see matched, modified, missing, extra clauses. |
| **Case analyst** | Summarize a case file (streaming or one-shot); get executive summary, timeline, key arguments, open issues, citations. |
| **Bilingual user** | Query in Arabic or English; get answers in chosen language with bilingual RAG. |

---

## 4. Current Feature Set

### 4.1 Document ingestion & storage

- **Upload:** PDF, DOCX, DOC as “document” or “template.”
- **Chunking:** Configurable chunk size/overlap; page numbers tracked where possible.
- **Document ID:** Deterministic content-hash (SHA-256).
- **OCR:** Fallback for scanned/image PDFs (Tesseract; configurable language, DPI, batch size).
- **Storage:** Raw files in `data/documents/`, `data/templates/`; vector index in `data/vector_store/`; clauses in `data/clause_store/`, `data/extracted_clauses/`.

### 4.2 Document Explorer (current flagship)

Three API entry points; UI uses Evidence + Answer.

| Entry | API | Service | Purpose |
|-------|-----|---------|--------|
| **Evidence** | `POST /api/explore-evidence` | EvidenceExplorerService | Verbatim evidence only (no LLM). Modes: `text`, `clauses`, `both`. |
| **Answer** | `POST /api/explore-answer` | RAGService | RAG search → context → LLM → answer + sources. |
| **Legacy** | `POST /api/explore` | DocumentExplorerService | Combined search (semantic + lexical) → LLM answer + evidence. |

**Evidence Explorer behavior:**

- Query expansion (e.g. termination family) and text normalization for matching.
- **Text mode:** Vector search (with similarity threshold for explorer) + lexical fallback over all document chunks; results are chunks only.
- **Clauses mode:** ExtractedClauseStore → score by heading/body vs expanded keywords; results are clauses only.
- **Both mode:** Clauses first; if no clause results, fall back to text.
- No merging/summarizing/stitching of evidence (verbatim only).
- No hard score cutoff that would exclude lexical/heading-eligible results.

**RAG Answer Explorer:**

- Uses RAGService (vector search, optional clause-store candidates, keyword fallback).
- Builds context from chunks (or clauses); generates answer via Ollama; returns answer, confidence, sources, citation.
- Per product rule: RAG must not return `status="not_specified"` unless Evidence Explorer would return `status="not_found"`.

### 4.3 Contract Review (workflow)

- **Input:** Contract document ID, contract type (employment, nda, msa), optional jurisdiction, review depth (standard/quick).
- **Flow:** WorkflowOrchestrator → ContractReviewService; load profile, gather clauses (store or extraction), identify missing clauses and risks, build evidence blocks and executive summary.
- **Output:** Structured response in WorkflowContext (risks, missing clauses, evidence per clause, executive summary); no raw dict return.
- **Guardrails:** Non-prescriptive language; disclaimer in UI.

### 4.4 Due Diligence Memo (workflow)

- **Input:** Document ID.
- **Flow:** WorkflowOrchestrator → DueDiligenceMemoService; readiness + document-type classification, case spine, section-level summarization (with independent skip on failure).
- **Output:** Memo sections, case spine, citations; results in WorkflowContext.
- **Guardrails:** Non-prescriptive language; determinism-friendly (seed, temperature).

### 4.5 Contract comparison

- **Input:** Contract ID, template ID.
- **API:** `POST /api/compare`.
- **Output:** Matched, modified, missing, extra clauses with similarity where applicable.

### 4.6 Clause extraction & store

- **Extraction:** Structured clause extraction + validation; stored in clause_store and extracted_clause_store (with `extraction_version` for explainability).
- **APIs:** Extract (`POST /api/extract-clauses`), list by document (`GET /api/clauses/{document_id}`), get single (`GET /api/clauses/single/{clause_id}`), query by metadata (`POST /api/clauses/query`).

### 4.7 Case file summarization

- **Input:** Document ID, optional top_k.
- **APIs:** One-shot `POST /api/summarize`; streaming `POST /api/summarize/stream` (SSE).
- **Output:** Executive summary, timeline, key arguments, open issues, citations; optional case spine.

### 4.8 Bilingual search

- **API:** `POST /api/search-bilingual`.
- **Behavior:** Query language detection (Arabic/English); optional translation; response language configurable.

### 4.9 Document management

- **List:** `GET /api/documents` (with optional version info).
- **Rename:** `PUT /api/documents/{document_id}/rename` (display name).
- **Health & stats:** `GET /api/health`, `GET /api/stats`.

---

## 5. System Context & Architecture

- **Frontend:** Streamlit (`frontend/app.py`) — “Legal Workflows Engine” — pages: Contract Review, Contract Comparison, Due Diligence Memo, Document Explorer, Upload Document, Clause Extraction, About.
- **Backend:** FastAPI (`backend/main.py`); CORS enabled for Streamlit.
- **Orchestration:** WorkflowOrchestrator runs Contract Review, Due Diligence Memo, Document Explorer (legacy), Evidence Explorer; passes WorkflowContext; failures set status and error, partial results retained.
- **Key services:** DocumentExplorerService, EvidenceExplorerService, RAGService, ExtractedClauseStore, ClauseStore, ContractReviewService, DueDiligenceMemoService, VectorStore, EmbeddingService, guardrails.
- **LLM:** Ollama (local), optional; required for RAG answers, clause extraction, case summarization, due diligence section generation.

---

## 6. Key Product Rules (from EVIDENCE_EXPLORER_SPEC)

| # | Rule | Purpose |
|---|------|--------|
| 1 | Lexical matches always eligible in Evidence Explorer | OCR/FAISS recall; avoid false negatives |
| 2 | No score cutoff for eligible results in Evidence Explorer | Avoid threshold-based false negatives |
| 3 | `debug_version` in explorer debug blocks | Schema evolution; tests |
| 4 | Evidence Explorer is authority for “absence”; RAG may say not_specified only when Evidence returns not_found | Consistency between Evidence and Answer |
| 5 | `extraction_version` in clause store | Explainability; regression debugging |
| 6 | Evidence Explorer never merges/summarizes/stitches evidence | Verbatim recall only |

---

## 7. Out of Scope (current stage)

- Legal advice or outcome prediction.
- Contract drafting.
- Internet browsing or external API calls for content.
- Multi-tenant or enterprise security/compliance guarantees.
- Production SLA or high-availability deployment.

---

## 8. Success Criteria for Current Stage

- All listed workflows run end-to-end from UI with backend + optional Ollama.
- Document Explorer: Evidence (text/clauses/both) and Answer behave per DOCUMENT_EXPLORER_WORKFLOW and EVIDENCE_EXPLORER_SPEC.
- Contract Review and Due Diligence Memo return structured results in WorkflowContext; guardrails applied.
- Evidence Explorer recall-first behavior and no merge/summarize/stitch are implemented and testable.
- API docs at `/docs` and README/START_COMPLETE_SYSTEM reflect current capabilities.

---

## 9. References

- **Workflow & data flow:** `docs/DOCUMENT_EXPLORER_WORKFLOW.md`
- **Evidence & RAG rules:** `EVIDENCE_EXPLORER_SPEC.md`
- **Setup & run:** `README.md`, `START_COMPLETE_SYSTEM.md`
- **Testing:** `TESTING_GUIDE.md`

---

*This PRD describes the product as implemented at the current stage (v0.2). New features or rule changes should be reflected here and in the referenced specs.*
