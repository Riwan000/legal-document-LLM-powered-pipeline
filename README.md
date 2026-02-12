## Legal Document Intelligence (LLM-powered pipeline)

A demo-ready, private Legal Document Intelligence system built with:
- FastAPI backend + Streamlit UI
- RAG (Retrieval-Augmented Generation) with citations
- Multilingual embeddings + FAISS vector store (persistent)
- Clause extraction + clause store
- Contract comparison (template vs contract)
- Case file summarization
- Bilingual search (Arabic/English)

This repository is designed for local, auditable workflows where documents are retrieved at query time (not used for training).

---

## Features implemented (so far)

- **Document upload + ingestion**
  - Supports **PDF / DOCX / DOC**
  - Splits documents into chunks with overlap, tracks page numbers where possible
  - Uses a deterministic **content-hash** document ID (SHA-256)
- **OCR support for scanned PDFs**
  - If a PDF has poor/empty extractable text, the pipeline can fall back to OCR
  - Uses a lower similarity threshold for OCR-derived text (configurable)
- **Vector search (multilingual)**
  - Sentence-Transformers multilingual embeddings
  - FAISS vector index persisted in `data/vector_store/`
- **RAG semantic search with citations**
  - Returns top matching chunks with score + page references
  - Optional: generate an LLM answer using **Ollama** (local)
- **Clause extraction (structured) + storage**
  - Extracts structured clauses, validates them, stores JSON in `data/clause_store/`
  - API to retrieve clauses by document, by clause ID, and via filters
- **Contract comparison**
  - Compares a contract against a template and reports matched/modified/missing/extra clauses
- **Case file summarization**
  - Produces a structured summary: executive summary, timeline, key arguments, open issues, citations
- **Bilingual search**
  - Detects query language (Arabic/English)
  - Can return answers in Arabic or English (configurable per request)

---

## Architecture (high level)

- **Frontend (Streamlit)**: `frontend/app.py` on `http://localhost:8501`
- **Frontend (Young Counsel UI)**: Next.js + React + Tailwind in `frontend/young-counsel-ui` on `http://localhost:3000`
- **Backend**: FastAPI (`backend/main.py`) on `http://localhost:8000`
- **Storage**
  - Raw files: `data/documents/` and `data/templates/`
  - Vector index: `data/vector_store/`
  - Extracted clauses: `data/clause_store/`
- **LLM**: Ollama (local) at `http://localhost:11434` (optional, required only for LLM-powered features)

---

## Requirements

- **Python**: 3.9+
- **Ollama (optional but recommended)**: required for LLM answers, clause extraction, and case summarization
- **OCR dependencies (optional)**: only needed if you ingest scanned/image-based PDFs
  - Tesseract OCR
  - Poppler (for `pdf2image`)

---

## Setup

### 1) Create and activate a virtual environment

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

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) (Optional) Configure environment variables

This repo uses sensible defaults (see `backend/config.py`). If you want to override settings, create a `.env` file in the project root.

Common overrides:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

### 4) (Optional) Start Ollama

```bash
ollama serve
ollama pull llama3.2
```

Note: the default model configured in `backend/config.py` is `llama3.2:latest`. If you use a different model tag, update `OLLAMA_MODEL`.

---

## Run the app

### Start the FastAPI backend

From the project root:

```bash
python run_backend.py
```

Or:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

### Start the Streamlit frontend

In a second terminal:

```bash
streamlit run frontend/app.py
```

UI (Streamlit): `http://localhost:8501`

### Start the Young Counsel Next.js frontend

In a separate terminal:

```bash
cd frontend/young-counsel-ui
npm install
npm run dev
```

By default the Young Counsel UI will be available at `http://localhost:3000` and will talk to the FastAPI backend at `http://localhost:8000` (configurable via `NEXT_PUBLIC_API_BASE_URL` in `frontend/young-counsel-ui/.env`).

---

## Quick demo workflow (UI)

- **Upload Document**: upload a PDF/DOCX as `document` or `template`
- **RAG Search**: ask questions; enable/disable “Generate LLM response”
- **Clause Extraction**: extract and browse clauses grouped by type
- **Contract Comparison**: compare contract vs template
- **Case Summary**: generate structured case summary
- **Bilingual Search**: query in Arabic or English, choose response language

For a fuller walkthrough, see `START_COMPLETE_SYSTEM.md`.

---

## API endpoints (backend)

You can test everything from Swagger at `http://localhost:8000/docs`.

- **GET** `/api/health`: health + config snapshot
- **POST** `/api/upload`: upload & ingest a document/template
- **POST** `/api/search`: RAG search (optional LLM answer)
- **POST** `/api/search-bilingual`: bilingual search (Arabic/English)
- **POST** `/api/extract-clauses`: extract & (optionally) store structured clauses
- **GET** `/api/clauses/{document_id}`: list clauses for a document
- **GET** `/api/clauses/single/{clause_id}`: fetch a specific clause
- **POST** `/api/clauses/query`: filter clauses by metadata
- **POST** `/api/compare`: compare contract vs template
- **POST** `/api/summarize`: case file summarization
- **GET** `/api/documents`: list ingested docs (from vector store metadata)
- **GET** `/api/stats`: vector store + config info

---

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run step scripts (manual/regression style):

```bash
python test_ingestion.py
python test_embeddings.py
python test_rag.py
python test_clause_extraction.py
python test_contract_comparison.py
python test_case_summarization.py
python test_fastapi_backend.py
python test_streamlit_frontend.py
python test_api_endpoints.py
```

More details: `TESTING_GUIDE.md` and `tests/README.md`.

---

## Troubleshooting

- **Backend not reachable from UI**
  - Ensure FastAPI is running on `http://localhost:8000`
  - Check `API_BASE_URL` in `frontend/app.py`
- **Ollama-dependent features fail**
  - Start Ollama: `ollama serve`
  - Ensure the configured model exists: `ollama list`
  - If your tag differs, set `OLLAMA_MODEL` in `.env`
- **OCR not working**
  - Install Tesseract and ensure it is on PATH
  - Install Poppler and ensure `pdf2image` can find it
  - If your PDFs are text-based (not scanned), you can ignore OCR setup
- **Timeouts on large documents**
  - Clause extraction and ingestion can take time; try smaller files first
  - Backend timeouts are configured in `backend/config.py` (see `API_TIMEOUT`)

---

## Limitations and safety notes

This is a demo MVP and explicitly does not provide:
- Legal advice or interpretation
- Outcome prediction
- Contract drafting
- Internet browsing
- Enterprise security/compliance guarantees

Privacy note:
- Documents are processed locally and retrieved at query time (RAG).
- This system is intended to avoid “training on client data”; answers are generated from retrieved context.
