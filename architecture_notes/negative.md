# Negative / Worst-Case Architecture Notes

---

## 2026-02-25 — legal-document-LLM-powered-pipeline

- **Append-only index without deletion support**: Using an append-only vector index (FAISS/NumPy) while also exposing a delete-document endpoint creates a permanent index/metadata misalignment after the first deletion. The metadata list and the positional index diverge silently; searches return wrong citations forever. Any system combining an append-only index with mutable document sets must either (a) rebuild the index on deletion, (b) use soft-deletes with query-time filtering, or (c) switch to an index that supports removal.

- **No thread safety on shared mutable data structures**: A shared in-memory data structure (list, dict) accessed concurrently by a writer (upload) and multiple readers (search requests) without locking will produce intermittent data corruption. The symptom — "search returns results from wrong document" — appears only under load and is almost impossible to reproduce deterministically.

- **Debug code committed to a production code path**: A hardcoded absolute filesystem path (developer's personal machine path) inside an exception handler in a production API endpoint will silently fail on all other machines. Debug logging must always use the application logger, never hardcoded paths. Review CI pipelines should reject `os.path.join("C:\\Users\\...")` or equivalent patterns in exception handlers.

- **Blocking synchronous operations in async handlers**: Calling CPU-intensive or blocking I/O operations (document parsing, embedding generation, LLM inference) directly inside an `async def` handler starves the event loop. With a single-threaded event loop, one slow request (30s document ingestion) blocks all other requests for its full duration. All blocking work must be offloaded to a thread pool executor.

- **CORS wildcard origins + credentials enabled**: Setting `allow_origins=["*"]` together with `allow_credentials=True` is explicitly prohibited by the CORS spec. Browsers silently reject all credentialed preflight responses. This misconfiguration appears "working" in local development (same-origin) but silently breaks all cross-origin deployments.

- **Exception message pass-through in HTTP error responses**: Returning `str(exception)` directly in HTTP error detail fields exposes internal library errors, file paths, SQL queries, and stack-trace fragments to API clients. Always log internally and return a generic message.

- **Duplicate-check outside the serialization lock**: Checking for existing records and then performing an insert/update under a separate lock creates a TOCTOU (time-of-check, time-of-use) window. Two concurrent requests can both pass the existence check, then both proceed to insert duplicate records. The check must be inside the same lock as the insert.
