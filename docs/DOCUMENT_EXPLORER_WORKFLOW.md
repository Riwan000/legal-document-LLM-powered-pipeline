# Document Explorer – Workflow & Data Flow

This document describes the Document Explorer workflow and how data/chunks flow through the system. There are **three** API entry points; the UI uses the two new ones (Evidence + Answer). The legacy `/api/explore` still runs the old “answer + evidence” flow.

---

## High-Level Entry Points

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT EXPLORER ENTRY POINTS                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  UI "Evidence" tab    →  POST /api/explore-evidence  →  EvidenceExplorerService  │
│  UI "Answer" tab      →  POST /api/explore-answer    →  RAGService.query        │
│  Legacy               →  POST /api/explore            →  DocumentExplorerService  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Evidence Explorer Flow (`/api/explore-evidence`) – No LLM

**Request:** `document_id`, `query`, `top_k`, `mode` (text | clauses | both)

```
┌──────────────┐     POST /api/explore-evidence      ┌─────────────────────────┐
│   Frontend   │ ──────────────────────────────────► │  main.explore_evidence  │
└──────────────┘                                      └───────────┬─────────────┘
                                                                  │
                                                                  ▼
                                                    ┌─────────────────────────────┐
                                                    │ WorkflowOrchestrator        │
                                                    │ .run_evidence_explorer()    │
                                                    │ → EvidenceExplorerRequest   │
                                                    └───────────┬─────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    EvidenceExplorerService.run(ctx, request)                                  │
│  • expand_query_terms(query) → List[str]   (normalize_for_match + QUERY_FAMILIES)             │
│  • detect_ocr_noise(query)                                                                   │
│  • Branch by mode: "both" | "clauses" | "text"                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ├── mode = "both" ──────► _run_both()
         │                              │
         │                              ├─► _run_clauses() ──► if results: return clause_response
         │                              └─► else: _run_text()
         │
         ├── mode = "clauses" ───► _run_clauses()
         │
         └── mode = "text" ──────► _run_text()
```

### Chunk/Data Flow by Mode

**Text mode (`_run_text`):**

- **Query** → `expand_query_terms(query)` → **expanded_keywords** (list of strings).
- **RAGService.search**(query, top_k, document_id_filter, similarity_threshold=0.45):
  - `embedding_service.embed_text(query)` → **query_embedding**
  - **vector_store.search**(query_embedding, top_k, document_id_filter, threshold) → **semantic_results** (list of chunk dicts: text, page_number, chunk_index, score, document_id, …).
- If no semantic results: **vector_store.get_chunks_by_document(document_id)** → **all_chunks**; for each chunk, **normalize_for_match(chunk["text"])** and match **expanded_keywords** → **lexical_results** (chunk dicts + score=1.0, lexical_match=True).
- **combined** = union(semantic_results, lexical_results), sort by score, cap top_k.
- Each combined chunk dict → **EvidenceExplorerResult**(text_snippet=truncated text, page_number, chunk_index, score, source_type="chunk") → **EvidenceExplorerResponse**(status, results, reason, debug).

**Clauses mode (`_run_clauses`):**

- **ExtractedClauseStore.get_document_clauses(document_id)** → **clauses** (list of dicts: clause_heading, verbatim_text, page_start, page_end, clause_id).
- For each clause: **score_clause(clause, expanded_keywords)** uses **normalize_for_match** on heading and verbatim_text, counts heading/body hits → **score**; keep if heading_hits ≥ 1 or score > 0.
- Sorted matches (by score), capped top_k → each clause → **EvidenceExplorerResult**(text_snippet=verbatim truncated, page_number=page_start, source_type="clause", clause_id, page_end) → **EvidenceExplorerResponse**.

**Both mode:** Run **clauses** first; if **clause_response.results** non-empty, return it; else run **text** as above.

**Response:** `{ status, results[], reason, debug, workflow_id }` — **no** `answer` field. Chunks/clauses only become **results[]** (snippets + metadata).

---

## 2. RAG Answer Explorer Flow (`/api/explore-answer`)

**Request:** `document_id`, `query`, `top_k`, `response_language` (optional)

```
┌──────────────┐     POST /api/explore-answer       ┌─────────────────────┐
│   Frontend   │ ─────────────────────────────────► │ main.explore_answer  │
└──────────────┘                                    └──────────┬──────────┘
                                                               │
                                                               ▼
                                                    ┌──────────────────────┐
                                                    │ RAGService.query(    │
                                                    │   query,             │
                                                    │   document_id_filter │
                                                    │   generate_response  │
                                                    │ )                    │
                                                    └──────────┬───────────┘
                                                               │
         ┌────────────────────────────────────────────────────┼────────────────────────────────────────────────────┐
         │                                                     ▼                                                     │
         │  (no chunks_override)  →  clause_store.get_candidate_clauses(document_id)  →  structured_clauses       │
         │                            RAGService.search(query, top_k, document_id_filter)  →  chunks                 │
         │                            optional: _keyword_fallback_search if no chunks                              │
         │                                                     │                                                     │
         │                                                     ▼                                                     │
         │                            _build_context(chunks) or _build_context_from_clauses(structured_clauses)     │
         │                            _build_legal_prompt(query, context, ...)                                       │
         │                            ollama_client.generate(prompt)  →  answer                                      │
         │                            _format_sources_enhanced(chunks)  →  sources                                  │
         │                                                     │                                                     │
         └─────────────────────────────────────────────────────┼─────────────────────────────────────────────────────┘
                                                               ▼
                                                    Response: { status, answer, confidence, sources[], citation }
```

**Chunk flow:** Chunks come from **RAGService.search** (embed → vector_store.search / search_with_priority, optional keyword fallback) or from **ClauseStore**. Those chunks (or clause-based context) feed **context** → **prompt** → **LLM** → **answer**. The same chunks are formatted as **sources** (text, page_number, document_id, score, citation, etc.) and returned with the answer.

---

## 3. Legacy Document Explorer Flow (`/api/explore`)

**Request:** `document_id`, `query`, `top_k`, `mode` (text | clauses)

```
┌──────────────┐     POST /api/explore              ┌─────────────────────────┐
│   Caller     │ ─────────────────────────────────► │ main.document_explorer  │
└──────────────┘                                    └───────────┬─────────────┘
                                                                 │
                                                                 ▼
                                                    ┌─────────────────────────────┐
                                                    │ WorkflowOrchestrator        │
                                                    │ .run_document_explorer()   │
                                                    │ → DocumentExplorerRequest   │
                                                    └───────────┬────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    DocumentExplorerService.run(ctx, request)                                  │
│  (Currently both modes use text search only; mode is only logged.)                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  _search_text_mode(document_id, query)
         │
         ├─► _lexical_keywords(query)  →  keywords (normalize_for_match + termination family)
         ├─► vector_store.get_chunks_by_document(document_id)  →  chunks (all chunk dicts)
         ├─► RAGService.search(query, top_k, document_id_filter, similarity_threshold=None)
         │        → semantic_results (chunk dicts)
         ├─► _lexical_fallback(chunks, keywords)  →  lexical_results (chunk dicts, score=1.0)
         ├─► combined = union(lexical_results, semantic_results), sort by score, cap top_k
         └─► search_result = { "results": combined, "semantic": n, "lexical": m }
         │
         ├─► if combined empty: DocumentExplorerResponse(status="not_found", results=[], reason=...)
         │
         └─► else:
                  RAGService.query(query, document_id_filter, chunks_override=combined)
                        │
                        ├─► Uses combined chunks as context (no new search)
                        ├─► _build_context(chunks)  →  context string
                        ├─► _build_legal_prompt(query, context, ...)  →  prompt
                        ├─► ollama_client.generate(prompt)  →  answer
                        └─► sources = _format_sources_enhanced(chunks)
                        │
                  DocumentExplorerResponse(answer, status, confidence, results=DocumentExplorerResult from sources)
```

**Chunk flow:** **VectorStore.get_chunks_by_document** yields the full list of **chunks** for the document. **RAGService.search** (no threshold) returns **semantic_results** (chunk dicts). **_lexical_fallback(chunks, keywords)** filters **chunks** by keyword match → **lexical_results**. **combined** = merged, sorted, capped chunk list. **combined** is passed as **chunks_override** into **RAGService.query** → used as **context** and for **sources**; LLM produces **answer**. **sources** (from those same chunks) are turned into **DocumentExplorerResult** (snippet, page, score, etc.) and returned with **answer**.

---

## 4. Mermaid: All Three Flows

```mermaid
flowchart TB
    subgraph UI["Frontend (Document Explorer page)"]
        A1["Evidence tab: query + mode"]
        A2["Answer tab: query"]
    end

    subgraph API["FastAPI main.py"]
        E1["POST /api/explore-evidence"]
        E2["POST /api/explore-answer"]
        E3["POST /api/explore (legacy)"]
    end

    subgraph Orch["WorkflowOrchestrator"]
        O1["run_evidence_explorer()"]
        O2["run_document_explorer()"]
    end

    subgraph EvSvc["EvidenceExplorerService"]
        expand["expand_query_terms(query)"]
        run["run(ctx, EvidenceExplorerRequest)"]
        text["_run_text(): RAG.search + get_chunks + lexical → EvidenceExplorerResult"]
        clauses["_run_clauses(): get_document_clauses + score_clause → EvidenceExplorerResult"]
        both["_run_both(): clauses first else text"]
    end

    subgraph DocSvc["DocumentExplorerService"]
        search_t["_search_text_mode(): get_chunks + RAG.search + _lexical_fallback → combined chunks"]
        rag_q["RAGService.query(chunks_override=combined)"]
        to_result["sources → DocumentExplorerResult"]
    end

    subgraph RAG["RAGService"]
        embed["embed_text(query)"]
        vs_search["vector_store.search(embedding, top_k, doc_id, threshold)"]
        kw_fallback["_keyword_fallback_search (if no results)"]
        query_full["query(): search or chunks_override → context → LLM → answer + sources"]
    end

    subgraph Data["Chunk / clause data"]
        VS["VectorStore: metadata (chunks by document)"]
        ECS["ExtractedClauseStore: get_document_clauses"]
    end

    A1 --> E1
    A2 --> E2
    E1 --> O1
    O1 --> run
    run --> expand
    run --> both
    both --> clauses
    both --> text
    run --> clauses
    run --> text

    text --> RAG
    text --> VS
    clauses --> ECS

    RAG --> embed
    embed --> vs_search
    vs_search --> VS
    RAG --> kw_fallback
    kw_fallback --> VS

    E2 --> query_full
    query_full --> RAG
    query_full --> VS

    E3 --> O2
    O2 --> search_t
    search_t --> VS
    search_t --> RAG
    search_t --> rag_q
    rag_q --> RAG
    rag_q --> to_result
```

---

## Summary Table

| Path | Entry | Chunk source | Chunk flow | Output |
|------|--------|--------------|------------|--------|
| **Evidence (text)** | `/api/explore-evidence` mode=text | VectorStore (search + get_chunks_by_document) | query → expand_query_terms → RAG.search(threshold 0.45) → optional lexical over get_chunks → combined → EvidenceExplorerResult[] | status, results[], reason, debug (no answer) |
| **Evidence (clauses)** | `/api/explore-evidence` mode=clauses | ExtractedClauseStore.get_document_clauses | clauses → score_clause(normalize, expanded_keywords) → EvidenceExplorerResult[] | status, results[], reason, debug (no answer) |
| **Evidence (both)** | `/api/explore-evidence` mode=both | Clauses then VectorStore | _run_clauses; if empty → _run_text (same as text row) | status, results[], reason, debug (no answer) |
| **Answer** | `/api/explore-answer` | RAGService.search / ClauseStore / keyword fallback | search → chunks → context → LLM → answer; chunks → sources[] | status, answer, confidence, sources[], citation |
| **Legacy** | `/api/explore` | VectorStore.get_chunks + RAG.search + lexical | get_chunks + search + lexical_fallback → combined → RAG.query(chunks_override) → context + LLM → answer; same chunks → DocumentExplorerResult[] | answer, results[], status, confidence, citation, debug |

---

## Key Files

- **Endpoints:** `backend/main.py` — `explore_evidence`, `explore_answer`, `document_explorer`
- **Orchestration:** `backend/services/workflow_orchestrator.py` — `run_evidence_explorer`, `run_document_explorer`
- **Evidence-only:** `backend/services/evidence_explorer_service.py` — modes text/clauses/both, expand_query_terms, score_clause
- **Legacy answer+evidence:** `backend/services/document_explorer_service.py` — _search_text_mode, RAG.query(chunks_override)
- **Retrieval:** `backend/services/rag_service.py` — search, query; `backend/services/vector_store.py` — search, get_chunks_by_document
- **Clauses:** `backend/services/extracted_clause_store.py` — get_document_clauses
- **Normalization:** `backend/services/text_normalizer.py` — normalize_for_match, detect_ocr_noise

---

## Frontend → Backend → Frontend Flow (Document Explorer UI)

This diagram focuses specifically on how the **Document Explorer dashboard page** in the Next.js UI
collects inputs, calls the backend, and renders outputs for both **Evidence** and **Answer** tabs.

```mermaid
flowchart LR
    %% UI LAYER
    subgraph UI["Next.js UI – Document Explorer page (/document-explorer)"]
        U1["User selects document from list (Document picker)"]
        U2["User types query (e.g. 'Where are the payment terms?')"]
        U3["User chooses tab:
        • Evidence (snippets only)
        • Answer (RAG answer + citations)"]
        U4["Click 'Search' / 'Run'"]
    end

    %% FRONTEND API HELPERS
    subgraph FEAPI["Frontend API helpers (frontend/young-counsel-ui/src/lib/api.ts)"]
        FE_EVID["exploreEvidence({ document_id, query, top_k?, mode })"]
        FE_ANS["exploreAnswer({ document_id, query, top_k?, response_language? })"]

        FE_EVID_BODY["Build URLSearchParams:
        • document_id
        • query
        • top_k (optional)
        • mode ('text' | 'clauses' | 'both')"]

        FE_ANS_BODY["Build URLSearchParams:
        • document_id
        • query
        • top_k (optional)
        • response_language (optional)"]

        FE_EVID_HTTP["POST
        ${API_BASE_URL}/api/explore-evidence
        Content-Type: application/x-www-form-urlencoded"]

        FE_ANS_HTTP["POST
        ${API_BASE_URL}/api/explore-answer
        Content-Type: application/x-www-form-urlencoded"]
    end

    %% BACKEND FASTAPI
    subgraph API["FastAPI endpoints (backend/main.py)"]
        BE_EVID["explore_evidence()
        @app.post('/api/explore-evidence')"]
        BE_ANS["explore_answer()
        @app.post('/api/explore-answer')"]
    end

    %% WORKFLOW / SERVICES
    subgraph SVC["Backend services"]
        subgraph WF["WorkflowOrchestrator (backend/services/workflow_orchestrator.py)"]
            WF_EVID["run_evidence_explorer(document_id, query, top_k, mode, debug)"]
            WF_CTX["Create WorkflowContext
            • workflow_id
            • workflow_type='document_explorer'
            • document_ids=[document_id]
            • metadata={query, top_k, mode}"]
        end

        subgraph EVS["EvidenceExplorerService (evidence only)"]
            EV_RUN["run(ctx, EvidenceExplorerRequest)"]
            EV_EXP["expand_query_terms(query)
            • normalize_for_match
            • add related terms"]
            EV_MODE["Branch by mode:
            • text
            • clauses
            • both"]
            EV_TEXT["_run_text():
            • RAGService.search(...)
            • vector_store.get_chunks_by_document(...)
            • lexical fallback
            • combined EvidenceExplorerResult[]"]
            EV_CLAUSES["_run_clauses():
            • ExtractedClauseStore.get_document_clauses(...)
            • score_clause(...)
            • EvidenceExplorerResult[]"]
            EV_RESP["EvidenceExplorerResponse:
            • status, results[], reason, debug"]
        end

        subgraph RAG["RAG answer path"]
            RAG_Q["RAGService.query(
            query,
            document_id_filter=document_id,
            generate_response=True
            )"]
            RAG_SEARCH["RAGService.search:
            • embed_text(query)
            • vector_store.search / search_with_priority
            • optional keyword fallback
            • authority ranking"]
            RAG_CTX["Build context from retrieved chunks/clauses"]
            RAG_LLM["Ollama LLM:
            • generate legal-style answer
            • consider legal hierarchy + reasoning"]
            RAG_OUT["AnswerResponse:
            • answer
            • status, confidence
            • sources[] with citations"]
        end
    end

    %% FRONTEND RENDERING
    subgraph UI_OUT["UI rendering"]
        OUT_EVID["Render Evidence tab:
        • For each result:
          – page_number
          – score
          – source_type (chunk/clause)
          – text_snippet
        • Show 'not found' / reason if status=not_found"]

        OUT_ANS["Render Answer tab:
        • Show answer text
        • Show status + confidence
        • Show primary citation (e.g. 'Page 4, Clause 7')
        • Show list of sources with page + snippet"]
    end

    %% WIRING: UI → FRONTEND HELPERS
    U1 --> U2 --> U3 --> U4
    U4 -->|Evidence tab active| FE_EVID
    U4 -->|Answer tab active| FE_ANS

    FE_EVID --> FE_EVID_BODY --> FE_EVID_HTTP --> BE_EVID
    FE_ANS --> FE_ANS_BODY --> FE_ANS_HTTP --> BE_ANS

    %% BACKEND WIRING
    BE_EVID --> WF_EVID --> WF_CTX --> EV_RUN
    EV_RUN --> EV_EXP --> EV_MODE
    EV_MODE --> EV_TEXT
    EV_MODE --> EV_CLAUSES
    EV_TEXT --> EV_RESP
    EV_CLAUSES --> EV_RESP

    BE_ANS --> RAG_Q
    RAG_Q --> RAG_SEARCH --> RAG_CTX --> RAG_LLM --> RAG_OUT

    %% RESPONSES BACK TO UI
    EV_RESP -->|JSON: {status, results[], reason, debug, workflow_id}| OUT_EVID
    RAG_OUT -->|JSON: {answer, status, confidence, sources[], citation}| OUT_ANS
```

### Frontend-to-backend narrative

- **Input collection (UI):** The user chooses a document in the Document Explorer page, types a question, and selects either the **Evidence** or **Answer** tab before submitting.
- **Frontend request building:** The page calls `exploreEvidence` or `exploreAnswer` in `src/lib/api.ts`, which serialize the inputs into `application/x-www-form-urlencoded` form data and POST to the appropriate `/api/...` endpoint.
- **Backend processing:** FastAPI endpoints either:
  - Route to `WorkflowOrchestrator.run_evidence_explorer` → `EvidenceExplorerService` (evidence-only, no LLM), or
  - Call `RAGService.query` directly (answer + citations), both using the document’s chunks/clauses plus query.
- **Output rendering:** The JSON responses are used by the UI to render either a list of **evidence snippets with page numbers and scores** (Evidence tab) or a **natural-language answer with citations and supporting snippets** (Answer tab).
