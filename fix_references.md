# Troubleshooting Reference

## RAG returns "info not found" on every question

**First check:** Is the vector store empty?
```bash
python -c "
import pickle
with open('data/vector_store/legal_documents.metadata.pkl','rb') as f:
    m = pickle.load(f)
print(len(m['metadata']), 'chunks indexed')
print(set(x['document_id'] for x in m['metadata']))
"
```
Expected: non-zero chunk count covering the documents in use.

**If empty:** Re-ingest all active documents via the API:
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@data/documents/<DOC-ID>_v1.pdf" \
  -F "force_reingest=true"
```
Run once per document listed in `data/documents.db` where `is_latest = 1`.

**Root cause (known bug — fixed):** `VectorStore.save()` previously had no guard against saving an empty in-memory state. On server restart the store starts unloaded; if the server shuts down again before any query or ingestion triggers `_ensure_loaded()`, the shutdown hook overwrites the valid on-disk index with zeros. Fixed by returning early from `save()` when `_loaded` is `False`.

---

## Retrieval returns 0 chunks for one specific document

**Check the debug log** (`.cursor/debug.log`) for lines like:
```json
{"location": "chat_orchestrator.after_retrieve", "data": {"len_chunks": 0}}
```

If `len_chunks` is consistently 0 for one `document_id` but not others, that document's vectors are missing from the index. Re-ingest only that document with `force_reingest=true`.

---

## Guardrail always fails ("evidence_score": "none")

Occurs when retrieval returns 0 chunks. The guardrail has nothing to grade and fails immediately. Fix retrieval first (see above) — the guardrail result will recover on its own.

---

## General pipeline health checklist

| Check | Command / File |
|---|---|
| Vector store chunk count | `data/vector_store/legal_documents.metadata.pkl` → `metadata` list length |
| Active documents | `data/documents.db` → `documents` table, `is_latest = 1` |
| Backend health | `GET http://localhost:8000/api/health` |
| Recent pipeline trace | `.cursor/debug.log` — last 50 lines |
| Ollama running | `ollama list` |
