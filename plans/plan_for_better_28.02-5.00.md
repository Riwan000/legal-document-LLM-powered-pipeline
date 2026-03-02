# Plan: Repo & Codebase Improvements
**Date:** 2026-02-28

---

## 1. Repo Presentation

### README Improvements
- Add badges (build status, Python version, license, coverage) at the top
- Add a system architecture diagram (even a simple ASCII flowchart) — condense `ARCHITECTURE.md` content here
- Add a "Quick Demo" GIF or screenshot of the Streamlit UI
- Clearly mark what requires Ollama vs. what works offline
- Add a **Troubleshooting** section (common startup errors, Tesseract path issues, etc.)

### Structure Cleanup
- Move `diagnose_document_search.py`, `manual_test_OCR.py`, `test_api_endpoints.py`, `test_ingestion_pipeline.py`, `run_backend.py` from root into a `scripts/` or `tools/` folder — root is currently messy
- Merge `review/` and `reviews/` — they appear duplicated
- Merge `START_BACKEND.md` and `START_COMPLETE_SYSTEM.md` into a single `CONTRIBUTING.md` or `docs/SETUP.md`
- Move `deployment-plan.pdf` into `docs/`
- Add a `CHANGELOG.md` — helps collaborators and future-you understand what changed

### GitHub Hygiene
- Add a `.github/` folder with:
  - `ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`
  - `PULL_REQUEST_TEMPLATE.md`
  - A basic `CODEOWNERS` file
- Add GitHub Actions for at minimum: `pytest` on PRs, `ruff` lint check

---

## 2. Codebase Quality & Efficiency

### Split `backend/main.py` (currently 1,607 lines)

```
backend/
├── main.py              # App factory only (~80 lines)
├── routers/
│   ├── documents.py     # /upload, /documents, /ingest
│   ├── search.py        # /search, /rag-query
│   ├── clauses.py       # /extract-clauses, /clause-store
│   ├── sessions.py      # /sessions, /chat
│   ├── review.py        # /contract-review, /compare
│   └── health.py        # /health, /status
```

FastAPI's `APIRouter` makes this clean. Each router registers itself with the app in `main.py`.

### Dependency Injection with `Depends()`

All services are instantiated at module level in `main.py`. Move to FastAPI's dependency injection:

```python
# backend/dependencies.py
from functools import lru_cache

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

@lru_cache()
def get_vector_store(emb=Depends(get_embedding_service)) -> VectorStore:
    return VectorStore(emb)
```

Benefits: makes testing dramatically easier (override `Depends` in tests), avoids global state.

### Add `pyproject.toml` + `ruff` for Consistent Formatting

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I"]  # pyflakes + isort

[tool.ruff.isort]
known-first-party = ["backend"]
```

### Pydantic v2 Migration

`requirements.txt` pins `pydantic 2.5.0` but some models may still use v1 patterns (`class Config:` instead of `model_config`). Audit and migrate — v2 is 5–50x faster for validation.

### Cache `established_facts` to Avoid Redundant LLM Calls

In `session_manager.py` and `chat_orchestrator.py`, if facts are already extracted for a session, skip re-extraction:

```python
if "doc_type" not in session.established_facts:
    # call LLM to extract
```

### FAISS — Save After Every Ingestion

If the app crashes mid-session, newly ingested vectors are lost. Call `vector_store.save()` right after each document's vectors are added, not just on shutdown.

### SQLite WAL Mode

Both `documents.db` and `sessions.db` will block under concurrent reads. Add on connection open:

```python
conn.execute("PRAGMA journal_mode=WAL")
```

### Pin `torch` to CPU-Only Build

Currently `torch 2.2.0+` pulls the CUDA build (~2.5 GB). For CPU deployment:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.2.0+cpu
```

---

## 3. Privacy-Safe Logging

### Core Principle: Log Operations, Not Content

| Log this | Never log this |
|---|---|
| `"Document DOC-0021 ingested, 47 chunks"` | Document text or filename if it's a real person's name |
| `"Query classified as: clause_lookup"` | The raw query string (may contain names, case facts) |
| `"Session abc123 started"` | Session messages or conversation turns |
| `"Contract review completed: 3 risks found"` | Risk descriptions or clause text |
| `"Embedding generated in 1.2s"` | The text being embedded |

### Centralized Structured Logging

Create `backend/logging_config.py`:

```python
import logging
import logging.config
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "format": "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "structured",
            "level": "INFO",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10_000_000,   # 10 MB
            "backupCount": 5,
            "formatter": "structured",
            "level": "DEBUG",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    # Quiet noisy third-party libs
    "loggers": {
        "httpx": {"level": "WARNING"},
        "sentence_transformers": {"level": "WARNING"},
        "faiss": {"level": "WARNING"},
        "urllib3": {"level": "WARNING"},
    },
}

def configure_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
```

Call `configure_logging()` once at the top of `main.py`'s lifespan.

### PII Scrub Helper

```python
# backend/utils/log_scrub.py
import re

_PII_PATTERNS = [
    re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),        # Proper names (heuristic)
    re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b'),       # Emails
    re.compile(r'\b(?:\+?\d[\d\s\-]{7,}\d)\b'),          # Phone numbers
    re.compile(r'\b\d{9,}\b'),                            # Long ID numbers
]

def scrub(text: str, replacement: str = "[REDACTED]") -> str:
    for pattern in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
```

Use selectively where content must appear in logs:

```python
logger.debug("Query rewrite input: %s", scrub(original_query))
```

### Log Level Discipline

| Level | Use for |
|---|---|
| `DEBUG` | Internal state, timing, engine selection — never goes to production console |
| `INFO` | Operation outcomes: "ingested", "session started", "review completed" |
| `WARNING` | Degraded mode: fallback used, threshold not met, cache miss |
| `ERROR` | Caught exceptions with context (not the user's data) |
| `CRITICAL` | Service startup failure |

### Auto-Expire Old Logs

Add a startup sweep in lifespan:

```python
import time
for log_file in Path("logs").glob("*.log.*"):
    if time.time() - log_file.stat().st_mtime > 30 * 86400:
        log_file.unlink()
```

### Separate Audit Log from Application Log

Legal/document apps need an audit trail (who uploaded what, when a review ran) separate from debug logs:

```python
audit_logger = logging.getLogger("audit")
# separate handler → logs/audit.log (append-only, never rotated away)
audit_logger.info("document_ingested doc_id=%s pages=%d ip_hash=%s", doc_id, pages, ip_hash)
```

Hash the IP to detect anomalies without storing the raw value:
```python
import hashlib
ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16]
```

---

## Priority Order

| # | Task | Impact |
|---|---|---|
| 1 | Logging scrub + centralized config | Highest risk — do first |
| 2 | Split `main.py` into routers | Biggest quality-of-life improvement |
| 3 | Dependency injection with `Depends()` | Unlocks proper test isolation |
| 4 | `ruff` + `pyproject.toml` | One-time setup, pays off across all 45 services |
| 5 | README badges + diagram + root folder cleanup | Repo presentation |
