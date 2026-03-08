## Legal Document Intelligence (LLM-powered pipeline)

A private, local Legal Document Intelligence system built on:
- FastAPI backend + Streamlit frontend
- RAG pipeline with multi-engine retrieval and grounded citations
- Multilingual embeddings (Arabic/English) + FAISS vector store
- Contract Review Engine with risk analysis and PDF reporting
- Conversational document Q&A with evidence guardrails

All processing runs locally. No data leaves the machine.

---

## Features

- **Document ingestion**
  - PDF / DOCX / DOC (text-based and scanned)
  - SHA-256 content-hash document IDs (dedup-safe)
  - Chunks with page-number tracking
  - OCR fallback for scanned/image-based PDFs (Tesseract + Poppler)

- **Auto document classification**
  - Detects contract type on upload (employment, NDA, MSA, other, statute, case file…)
  - Used to auto-select the right review profile downstream

- **Contract Review Engine**
  - Clause extraction and coverage matrix per contract type (employment, NDA, MSA, general, other)
  - Risk scoring per clause (high / medium / low) with verbatim evidence snippets
  - Contradiction detection across clauses
  - LLM-based uncertainty resolution (Ollama binary YES/NO pass)
  - GCC statutory references (6 jurisdictions: UAE, KSA, Qatar, Bahrain, Kuwait, Oman)
  - Downloadable PDF report (English + Arabic variants)

- **Conversational Document Q&A**
  - Persistent chat sessions per document
  - 8-intent QueryClassifier (definition, clause lookup, summary, comparison, binary, classification, etc.)
  - 5-engine RetrievalRouter: definition, clause_title, clause_semantic, category, page_fallback
  - EvidenceGuardrailService: 2-pass validation to prevent hallucinated citations
  - Cross-turn consistency via `established_facts` (doc type, jurisdiction, parties extracted from answers)

- **Bilingual search (Arabic/English)**
  - Query language detection
  - Answers in Arabic or English (configurable per request)

- **Multilingual vector search**
  - `paraphrase-multilingual-MiniLM-L12-v2` embeddings
  - FAISS index persisted in `data/vector_store/`

- **Case file summarization** *(backend only — no UI page yet)*
  - Structured summary: executive summary, timeline, key arguments, citations
  - Due diligence memo generation
  - Available via API: `POST /api/summarize`, `POST /api/summarize/stream`, `POST /api/due-diligence-memo`

---

## Architecture

```
Streamlit UI (port 8501)
      │
      ▼
FastAPI backend (port 8000)
      │
      ├── QueryClassifier (8 intents, multi-label)
      ├── QueryRewriter
      ├── RetrievalRouter (5 engines → FAISS)
      ├── ChatOrchestratorService + SessionManager
      ├── EvidenceGuardrailService (2-pass validation)
      └── ContractReviewService → Ollama LLM (port 11434)
```

**Storage layout**

| Path | Contents |
|---|---|
| `data/documents/` | Uploaded raw files |
| `data/templates/` | Template files |
| `data/vector_store/` | FAISS index + metadata |
| `data/clause_store/` | Extracted clause JSON per document |

**Frontend pages (Streamlit)**

| Page | What it does |
|---|---|
| Document Explorer | 3-step workflow: Upload → Auto-classification → Q&A + Contract Review |
| About | Architecture explanation, demo script, privacy/compliance statement |

---

## Requirements

- **Python** 3.9+
- **Ollama** — required for LLM-powered features (contract review, Q&A generation)
  - Default model: `qwen2.5:3b`
- **OCR** (optional) — only needed for scanned/image PDFs
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) (for `pdf2image`)

---

## Setup

### 1. Create and activate a virtual environment

Windows (PowerShell):
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables (optional)

Defaults are in `backend/config.py`. Override by creating a `.env` in the project root:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

### 4. Start Ollama

```bash
ollama serve
ollama pull qwen2.5:3b
```

---

## Running the app

### Start the backend

```bash
python run_backend.py
```

Or directly:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API docs (Swagger): `http://localhost:8000/docs`

### Start the Streamlit frontend

In a second terminal:

```bash
streamlit run frontend/app.py
```

UI: `http://localhost:8501`

---

## Try it yourself (5-minute walkthrough)

### Step 1 — Upload a contract

1. Open `http://localhost:8501`
2. You are on the **Document Explorer** page by default
3. In **Step 1: Upload**, drag in a PDF or DOCX contract (employment agreement, NDA, MSA, or similar)
4. Click **Upload & Ingest** — the file is parsed, chunked, embedded, and indexed

### Step 2 — Classify

5. The system auto-detects the contract type (employment / NDA / MSA / other)
6. Confirm or override the detected type using the dropdown
7. Select a jurisdiction (optional; used for GCC statutory references in the report)
8. Click **Continue**

### Step 3 — Explore the document

After classification you land in the exploration panel. Two things happen:

**Contract Review (runs automatically for contracts)**

- The system extracts clauses, scores risks, detects contradictions, and resolves uncertain clauses via Ollama
- Results show: risk score + label, clause coverage matrix, per-clause evidence with verbatim snippets, contradiction flags, statutory references
- Download the PDF report (English or Arabic) using the download buttons

**Conversational Q&A**

- Ask questions like:
  - *"What is the notice period?"*
  - *"Does this contract have a non-compete clause?"*
  - *"What are the termination conditions?"*
  - *"ما هي مدة العقد؟"* (Arabic works too)
- Each answer includes cited page references and evidence chunks
- The chat session remembers established facts (parties, jurisdiction, doc type) across turns

### What to look for

| Thing to try | Where to find it |
|---|---|
| Risk summary (score + label) | Top of contract review results |
| Clause coverage matrix | Below the risk summary |
| Verbatim evidence per clause | Expandable evidence sections |
| Contradiction warnings | Flagged inline in the review |
| GCC statutory references | Statutory notes section of the review + PDF |
| Download PDF report | Download buttons below the review |
| Arabic Q&A | Type a question in Arabic in the chat box |

---

## API endpoints

Full interactive docs at `http://localhost:8000/docs`.

**Core**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check + config snapshot |
| GET | `/api/health/ai` | LLM + embedding health |
| GET | `/api/stats` | Vector store + config info |

**Documents**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload and ingest a document/template |
| GET | `/api/documents` | List all ingested documents |
| GET | `/api/documents/{id}/classification` | Get auto-classification for a document |
| PUT | `/api/documents/{id}/rename` | Rename a document |
| DELETE | `/api/documents/{id}` | Delete a document and its index entries |

**Clauses**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/extract-clauses` | Extract and store structured clauses |
| GET | `/api/clauses/{document_id}` | List clauses for a document |
| GET | `/api/clauses/single/{clause_id}` | Fetch a single clause |
| POST | `/api/clauses/query` | Filter clauses by metadata |

**Search & RAG**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/search` | RAG search with optional LLM answer |
| POST | `/api/search-bilingual` | Bilingual search (Arabic/English) |
| POST | `/api/translate` | Translate text |
| POST | `/api/explore` | Document exploration (guided RAG) |
| POST | `/api/explore-evidence` | Retrieve evidence for a query |
| POST | `/api/explore-answer` | Generate grounded answer from evidence |

**Chat (conversational sessions)**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat/session` | Create a new chat session |
| POST | `/api/chat/{session_id}` | Send a message in a session |
| GET | `/api/chat/{session_id}` | Get session history |
| DELETE | `/api/chat/{session_id}` | Delete a session |

**Contract workflows**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/contract-review` | Full contract review (risk + clauses + PDF) |
| POST | `/api/compare` | Compare contract against a template |
| GET | `/api/workflow/{workflow_id}/state` | Poll a running workflow state |

**Case analysis** *(backend only — no UI page)*

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/summarize` | Generate structured case summary |
| POST | `/api/summarize/stream` | Streaming case summary |
| POST | `/api/due-diligence-memo` | Generate due diligence memo |

**Admin**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/admin/clear-all` | Wipe all documents, vectors, and clause data |

---

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

---

## Troubleshooting

- **Backend not reachable from UI**
  - Confirm FastAPI is running on `http://localhost:8000`
  - Set `BACKEND_URL=http://127.0.0.1:8000` before starting Streamlit if needed

- **Ollama features fail (contract review, Q&A generation)**
  - Run `ollama serve` then `ollama list` to confirm the model is present
  - If using a different tag, set `OLLAMA_MODEL` in `.env`

- **OCR not working**
  - Install Tesseract and confirm it is on PATH (`tesseract --version`)
  - Install Poppler and confirm `pdf2image` can find it
  - Text-based PDFs do not require OCR — skip if your documents are not scanned

- **Slow contract review**
  - The review runs clause extraction, risk scoring, contradiction detection, and an LLM pass — expect 30–90 seconds on a mid-range machine
  - Use `standard` depth (default) rather than `deep` for faster results

- **Timeouts on large documents**
  - Try splitting very large files before ingesting
  - Backend timeouts are in `backend/config.py`

---

## Limitations and safety notes

This is a demo MVP. It explicitly does not provide:
- Legal advice or interpretation
- Outcome prediction
- Contract drafting
- Internet access
- Enterprise security or compliance certifications

**Privacy**: all processing is local. Documents are retrieved at query time (RAG). Model weights are never updated with client data.
