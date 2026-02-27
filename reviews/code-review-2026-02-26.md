# Code Review: Legal Document LLM-Powered Pipeline
**Date:** 2026-02-26
**Reviewer:** Senior Code Review (Claude Sonnet 4.6)
**Scope:** 15 backend files — full structured analysis

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Findings by Severity](#findings-by-severity)
   - [Critical](#critical)
   - [High](#high)
   - [Medium](#medium)
   - [Low](#low)
3. [Positive Aspects](#positive-aspects)
4. [File-by-File Notes](#file-by-file-notes)
5. [Prioritized Fix Order](#prioritized-fix-order)

---

## Executive Summary

The codebase is a sophisticated FastAPI-based RAG pipeline for legal document analysis. It has clearly evolved over multiple iterations with thoughtful architecture decomposition (service layer, workflow context, guardrails, audit logging). However, several findings — ranging from production-breaking to maintainability — need to be addressed before this can be considered production-ready. Two critical issues stand out: an infinite recursion bug in `rag_service.py` and an open CORS policy in `main.py`. Security posture is generally reasonable for an internal tool but insufficient for external exposure. The audit and guardrail layers are well-designed and above-average for a prototype.

**Counts:** 2 Critical · 5 High · 13 Medium · 12 Low

---

## Findings by Severity

---

### CRITICAL

---

#### FINDING C-01
**Category:** Logic Bug
**Severity:** Critical
**File:** `backend/services/rag_service.py`
**Lines:** ~199–261 (the `_search_without_translation` method body)

**Description:**
`_search_without_translation` contains a recursive call to itself (`self._search_without_translation(query, ...)`) on lines 205, 231, and a final fallback on line 261. When `ENABLE_QUERY_TRANSLATION` is `True` and the translation service is present, the method enters an infinite recursion: it calls itself for the "initial pass", which again checks the flag, calls itself again, and so on. There is no base case that skips the translation block.

The method is labeled "original implementation without translation logic" in its docstring, but the body is identical to the outer `search()` path and re-enters the translation block on every call. Python's default recursion limit (~1000 frames) will cause a `RecursionError` for any translated-query flow in production.

**Evidence:**
```python
# Line ~199 in _search_without_translation:
if settings.ENABLE_QUERY_TRANSLATION and self.translation_service:
    ...
    original_results = self._search_without_translation(   # <-- calls SELF
        query, initial_top_k, document_id_filter, priority_clause_types
    )
    ...
# Line ~261 (fallback at bottom of method):
return self._search_without_translation(query, top_k, ...)  # <-- SELF again
```

**Suggested Fix:**
The private method should contain only the non-translation search logic (embedding + vector store call). Extract that logic into a true base method (e.g., `_search_core`) and call it from both the translation path and the no-translation fallback. The public `search()` method already wraps the translation decision; `_search_without_translation` should just be the raw search path.

---

#### FINDING C-02
**Category:** Security
**Severity:** Critical
**File:** `backend/main.py`
**Lines:** 103 (`allow_origins=["*"]`)

**Description:**
The CORS middleware is configured with `allow_origins=["*"]` combined with `allow_credentials=True`. This combination is rejected by browsers (CORS spec forbids credentialed requests to wildcard origins), but more critically it signals the intent to eventually accept credentialed cross-origin requests from anywhere. For a legal document system that processes sensitive contracts, this is a production security risk. Even during development, it masks the actual frontend origin that should be whitelisted.

**Evidence:**
```python
app.user_middleware.append(
    _CORSMiddlewareMarker(
        CORSMiddleware,
        allow_origins=["*"],   # <-- wildcard
        allow_credentials=True,
        ...
    )
)
```

**Suggested Fix:**
Replace `allow_origins=["*"]` with an explicit list loaded from config/environment (e.g., `settings.ALLOWED_ORIGINS`). For development, default to `["http://localhost:8501"]` (Streamlit default). For production, require the env var to be set.

---

### HIGH

---

#### FINDING H-01
**Category:** Security
**Severity:** High
**File:** `backend/services/vector_store.py`
**Lines:** 604–630 (`load()` method — `pickle.load`)

**Description:**
The metadata file is deserialized using `pickle.load` with no integrity check. If an attacker can write to the `data/vector_store/` directory (or substitute the `.metadata.pkl` file), they can achieve arbitrary code execution when the server restarts and loads the index. This is a standard pickle deserialization vulnerability.

The numpy index file is also loaded via `pickle.load` (line 639) for the non-FAISS path, presenting the same attack surface.

**Evidence:**
```python
with open(metadata_path, 'rb') as f:
    payload = pickle.load(f)   # No HMAC, no checksum
```

**Suggested Fix:**
For the metadata file, replace pickle with JSON (metadata is a list of dicts with simple types). For the numpy vectors file, use `numpy.save`/`numpy.load` with `allow_pickle=False`. If pickle is retained for compatibility, add an HMAC signature stored alongside the file and verify it before loading.

---

#### FINDING H-02
**Category:** Logic Bug
**Severity:** High
**File:** `backend/services/chat_orchestrator.py`
**Lines:** 53–61 (module-level audit logger setup)

**Description:**
The audit log path is hardcoded as a relative path `Path("logs/chat.log")` at module import time. This is resolved relative to the current working directory at the moment the module is imported, not relative to the project root. In production deployments (e.g., systemd service, Docker), the CWD may differ from the project root, causing the audit log to be written to an unexpected location or failing silently if the directory does not exist. The `mkdir` call makes the directory, but the path could end up inside `/` or wherever the server process starts.

Additionally, the `debug.log` writes throughout the orchestrator use a `.cursor/debug.log` path (lines 248–258, 273–277, 641–647, 681–691). These are leftover agent-debugging artifacts that write to a Cursor IDE directory and should not be present in production code.

**Evidence:**
```python
_chat_log_path = Path("logs/chat.log")      # relative — CWD-dependent
_chat_log_path.parent.mkdir(parents=True, exist_ok=True)
# ...
_DEBUG_LOG_G = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
```

**Suggested Fix:**
Derive paths from `Path(__file__).resolve()` anchoring to the project root, consistent with how `.cursor/debug.log` is resolved. Remove all `.cursor/debug.log` write blocks (they are clearly development artifacts). Move the debug logging calls to `logger.debug(...)` using the standard `logging` infrastructure.

---

#### FINDING H-03
**Category:** Logic Bug
**Severity:** High
**File:** `backend/services/evidence_guardrail_service.py`
**Lines:** 38–47 (module-level debug logger)

**Description:**
The same `.cursor/debug.log` pattern appears as module-level code in `evidence_guardrail_service.py`. The `_dbg()` function is called on every guardrail `check()` invocation (multiple times per chat turn, including at entry and at each decision branch). This opens and appends to the file on every call with no buffering, causing O(N) file open/close operations per conversation turn and I/O contention under concurrent load. In a multi-user scenario this will degrade performance noticeably.

**Evidence:**
```python
_DEBUG_LOG = (Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log")
def _dbg(payload: dict) -> None:
    ...
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:   # opens file every call
        f.write(json.dumps(payload, ...) + "\n")
```

**Suggested Fix:**
Remove all `_dbg` / `#region agent log` blocks. These are IDE-specific debugging artifacts. If structured debug traces are needed in production, route them through the standard `logger.debug()` call with a structured formatter.

---

#### FINDING H-04
**Category:** Edge Case / Logic
**Severity:** High
**File:** `backend/services/session_manager.py`
**Lines:** 258–277 (`enforce_limits`)

**Description:**
`enforce_limits` triggers `self.summarizer.summarize(session)` synchronously in-band during the chat request when turn or token limits are approached. The `ConversationSummarizer` presumably calls the Ollama LLM, which can take several seconds. This means every (MAX_SESSION_TURNS)th user message will experience a multi-second delay before the response pipeline even starts. There is no timeout on this call, no async offloading, and no user feedback.

Additionally, if `summarizer.summarize()` raises an exception, the outer `try/except` in `ChatOrchestratorService.chat()` (line 597) silently swallows it with `logger.warning`, meaning the session state after a failed summarization is undefined.

**Evidence:**
```python
if (near_turn_limit or near_token_limit) and self.summarizer is not None:
    self.summarizer.summarize(session)   # blocking LLM call, no timeout
    self.store.save_session(session)
```

**Suggested Fix:**
Run summarization asynchronously (background task or separate thread) so it does not block the user's next response. Pass the session copy to summarize; when complete, update the store. If the summarizer fails, log the error and continue without summarization (graceful degradation). Consider adding a per-session flag `summarization_in_progress` to prevent double-triggers.

---

#### FINDING H-05
**Category:** Security / Logic
**Severity:** High
**File:** `backend/services/structured_clause_extraction.py`
**Lines:** 298–302 (chunk/embedding/token detection logic)

**Description:**
The "fail-closed" guard that rejects chunk-like input (line 299–302) checks whether the word "embedding", "chunk", or "token" appears in the page text. These are common words in legal documents (e.g., "token" appears in IP clauses about blockchain, "chunk" can appear in data processing agreements). This guard will silently return an empty clause list for documents containing these terms, making the extraction pipeline non-deterministic and producing a silent failure with no document-level error surfaced to the caller.

**Evidence:**
```python
if text and ('embedding' in text.lower() or 'chunk' in text.lower() or 'token' in text.lower()):
    logger.error("Input appears to be chunks/embeddings, not page-level text. Fail-closed.")
    return []
```

**Suggested Fix:**
Remove this heuristic entirely. The guard was designed to detect accidental pipeline misuse (passing embedding metadata as page text), but the actual structural contract — that the input comes from `FileParser.parse_file()` — is enforced by construction. If this guard is kept, it should be far more specific (e.g., require all three words AND numeric vector content) and should emit a warning rather than a silent empty return that propagates as "no clauses found".

---

### MEDIUM

---

#### FINDING M-01
**Category:** Logic Bug
**Severity:** Medium
**File:** `backend/services/query_classifier.py`
**Lines:** 107–134 (`_normalize` method)

**Description:**
The `_normalize` method uses plain `str.replace()` on the full query string, not word-boundary-aware substitution. This causes incorrect replacements: "termination" → "terminate" is fine, but "determination" → "deteterminate" because "termination" is a substring of "determination". Similarly, "cancellation" → "cancel" would corrupt "cancellation fee" before other parts of the classifier see it.

**Evidence:**
```python
for variant, base in self._normalization_map.items():
    normalized = normalized.replace(variant, base)  # no word boundaries
```

**Suggested Fix:**
Replace `str.replace` with `re.sub(r'\b' + re.escape(variant) + r'\b', base, normalized)` so that only whole-word matches are normalized. Pre-compile these patterns in `__init__` for performance.

---

#### FINDING M-02
**Category:** Logic Bug
**Severity:** Medium
**File:** `backend/services/retrieval_router.py`
**Lines:** 82–90 (`_merge` inner function)

**Description:**
The `engines_used` list accumulation in `_merge` has a subtle bug. When a chunk is already in `fused` and the new score is NOT higher, the code correctly appends the engine name (line 90). But when the score IS higher and the entry is replaced (line 84–88), the new entry's `engines_used` is built from `fused.get(cid, {}).get("engines_used", []) + [engine_name]`. The problem: `fused.get(cid, {})` is read before `fused[cid] = entry` is written, so the first time a chunk is seen, this correctly starts with `[]`. However, if a chunk appears multiple times in the same engine's results (which can happen if `_definition_engine`'s semantic fallback returns the same chunk that the primary scan already scored), the engines_used list will correctly grow. This is actually fine — but the condition `fused_score > fused[cid]["score"]` only fires when the new score strictly exceeds the old. Equal scores from different engines do not merge engine names into the winning entry.

More seriously: the score produced by `page_fallback_engine` (weight 0.4) and `clause_semantic_engine` (weight 1.0) for the same chunk may differ, and only the higher score entry is kept — but the lower-scoring engine's name is still appended via line 90. However, because the losing engine's name is appended to the existing entry without updating the score, the `engines_used` list on the retained entry grows but the score does not update. This is correct behavior, but the comment "highest wins" is subtly misleading — it's the highest weighted score, not the engine-local score.

A real bug: if the initial `fused` dict does not yet contain `cid` (first time this chunk is seen), `fused.get(cid, {}).get("engines_used", [])` returns `[]` correctly. But `entry["engines_used"]` is then `[] + [engine_name]`. On the very next iteration, if the same chunk appears again with a higher score, `fused.get(cid, {}).get("engines_used", [])` reads the OLD entry's engines_used. The new entry's engines_used therefore accumulates the old list. This is correct. But the NEW entry is then written to `fused[cid]`, discarding the OLD entry's other fields (like the old `engines_used` that was already in fused). Actually re-reading carefully: `entry = dict(r)` copies only the current result `r`, then `entry["engines_used"] = fused.get(cid, {}).get("engines_used", []) + [engine_name]` correctly pulls from the existing fused entry. This is fine logic but fragile; the next finding is more concrete.

**Concrete bug:** `_chunk_key` at the bottom of the file (line 379) and `RetrievalStrategy._chunk_key` in `chat_orchestrator.py` (line 152) produce keys in different formats:
- Router: `f"{entry.get('document_id')}:{entry.get('page_number')}:{entry.get('chunk_index')}"`
- Orchestrator: `f"{c.get('document_id', '')}::{c.get('chunk_index', '')}"`

These two key formats are incompatible. If a chunk from the router is passed through the orchestrator's deduplication, the same chunk will not deduplicate against itself because the key schemas differ (different separators, different fields).

**Suggested Fix:**
Standardize on one `_chunk_key` format across both files. Use `chunk_id` from metadata when available; fall back to a consistent composite key. Extract this to a shared utility function in `backend/utils/`.

---

#### FINDING M-03
**Category:** Logic Bug
**Severity:** Medium
**File:** `backend/services/evidence_guardrail_service.py`
**Lines:** 173–179 (evidence score decision logic)

**Description:**
The evidence score decision tree has a dead code branch. When `coverage_ratio >= GUARDRAIL_STRONG_THRESHOLD` but `citation_valid` is False, the score is set to `"moderate"` (line 176). Then when `coverage_ratio >= GUARDRAIL_STRONG_THRESHOLD` AND `citation_valid` is True, it is set to `"strong"` (line 174). This ordering is correct. However, the condition on line 177 (`elif coverage_ratio >= settings.GUARDRAIL_STRONG_THRESHOLD`) can never be reached when `citation_valid` is True because line 174 already captured that case. The `elif` on line 177 only fires when `citation_valid` is False AND `coverage_ratio >= STRONG_THRESHOLD`. Both lines 176 and 177 produce `"moderate"` for the same condition (high coverage but invalid citations), making one branch unreachable.

**Evidence:**
```python
if coverage_ratio >= settings.GUARDRAIL_STRONG_THRESHOLD and citation_valid:
    evidence_score = "strong"
elif coverage_ratio >= settings.GUARDRAIL_STRONG_THRESHOLD:   # only if citation_valid=False
    evidence_score = "moderate"
elif coverage_ratio >= settings.GUARDRAIL_WEAK_THRESHOLD:     # this is the same as above for citaiton_valid=False case
    evidence_score = "moderate"
```

Lines 176 and 178 both produce `"moderate"` for overlapping conditions. The intent was likely to have line 177 catch the case where `coverage_ratio` is between `WEAK` and `STRONG` thresholds regardless of `citation_valid`. The current code collapses that case correctly but the branch structure is misleading.

**Suggested Fix:**
Rewrite the block with explicit conditions:
```python
if coverage_ratio >= settings.GUARDRAIL_STRONG_THRESHOLD:
    evidence_score = "strong" if citation_valid else "moderate"
elif coverage_ratio >= settings.GUARDRAIL_WEAK_THRESHOLD:
    evidence_score = "moderate"
elif coverage_ratio > 0.0:
    evidence_score = "weak"
else:
    evidence_score = "none"
```

---

#### FINDING M-04
**Category:** Logic Bug
**Severity:** Medium
**File:** `backend/services/chat_orchestrator.py`
**Lines:** 693–698 (refusal logic)

**Description:**
The refusal decision has a fragile special case: when `evidence_score == "none"` AND `len(final_chunks) == 0`, the RAG answer is preserved as-is (the `pass` branch). This means that if a guardrail `decision == "fail"` is triggered by a zero-chunk result, the LLM's raw answer (which may say "I couldn't find this in the document" or, worse, a hallucinated answer if the LLM chose to generate one despite empty context) is passed through unchecked.

The comment says "answer came from RAG (e.g. not_specified)" but this is an assumption — the RAG service could return any string in that scenario, including a hallucination. The `_REFUSAL_ANSWER` constant exists for exactly this reason.

**Evidence:**
```python
if guardrail_result.evidence_score == "none" and len(final_chunks) == 0:
    pass  # keep final_answer (RAG's message) — potentially untrusted content
else:
    final_answer = _REFUSAL_ANSWER
```

**Suggested Fix:**
Instead of preserving an arbitrary RAG answer in the zero-chunk case, check the RAG answer against a whitelist of expected "not found" messages, or always substitute `_REFUSAL_ANSWER` when chunks are zero. Alternatively, have the RAG service set a flag on its response when it returns a "not found" message vs. a generated answer.

---

#### FINDING M-05
**Category:** Edge Case
**Severity:** Medium
**File:** `backend/services/vector_store.py`
**Lines:** 282–283 (index search result indexing)

**Description:**
In the `search()` method, after calling `self.index.search(query_embedding, top_k)`, the code iterates over `zip(distances[0], similarities)` and accesses `self.metadata[indices[0][idx]]`. If `document_id_filter` causes all top-k results to be filtered out, the method returns an empty list — which is correct. However, if the FAISS index returns indices greater than `len(self.metadata)` due to a corruption or mismatch that slipped past the consistency check on load, this will raise an `IndexError`. The consistency check in `load()` resets to empty on mismatch, but `add_chunks()` does not atomically add both the vector and the metadata entry — if an exception occurs between `self.index.add(embeddings)` and `self.metadata.append(...)` in the loop, the index and metadata will permanently diverge for that document.

**Evidence:**
```python
self.index.add(embeddings)   # vectors added
for chunk in chunks:         # metadata added in loop — exception here = divergence
    ...
    self.metadata.append(metadata_entry)
```

**Suggested Fix:**
Build the complete `new_metadata` list first, then add vectors and extend metadata in an atomic-as-possible sequence. Wrap both operations in try/except and roll back (remove added vectors if metadata append fails). At minimum, add a post-add assertion: `assert self.index.ntotal == len(self.metadata)`.

---

#### FINDING M-06
**Category:** Edge Case
**Severity:** Medium
**File:** `backend/services/session_manager.py`
**Lines:** 36–51 (regex patterns for fact extraction)

**Description:**
`_JURISDICTION_PATTERN` requires the exact phrase "governed by" (with optional "the" and "laws? of"). Many GCC contract phrasings use "subject to the laws of", "in accordance with the laws of", or simply state the jurisdiction without the governed-by phrasing. If the LLM paraphrases rather than using the exact trigger phrase, `established_facts["jurisdiction"]` is never populated, silently breaking the cross-turn consistency feature without any indication to the user or developer.

Similarly, `_DOC_TYPE_PATTERN` requires "this is a/an [X]" — a pattern that a model trained on concise answers often does not produce verbatim. The feature will frequently fail to extract facts in practice.

**Evidence:**
```python
_JURISDICTION_PATTERN = re.compile(
    r"governed\s+by\s+(?:the\s+)?(?:laws?\s+of\s+)?"
    r"(KSA|Saudi Arabia|UAE|...)[^.]*\.",
    re.IGNORECASE,
)
```

**Suggested Fix:**
Expand the jurisdiction pattern to cover "subject to", "in accordance with", "applicable law is", and bare country-name mentions with surrounding legal context. Alternatively, extract facts via a dedicated LLM classification call on the first assistant answer (one-time cost), which is more robust than regex-on-paraphrased-text. Document the current limitation clearly so callers know `established_facts` may be empty even when the document type is known.

---

#### FINDING M-07
**Category:** Logic Bug
**Severity:** Medium
**File:** `backend/services/contract_review_service.py`
**Lines:** 1047–1057 (`_load_risk_explanations`)

**Description:**
`_load_risk_explanations` uses a relative path `Path("backend/contract_profiles/risk_explanations.yaml")` resolved at call time from the CWD. If the server is started from a directory other than the project root (a common scenario in Docker containers, CI systems, or when running tests), the file will not be found and the method silently returns `{}`. The `_attach_explanations` then silently skips all explanation enrichment, producing a ContractReviewResponse with `severity_reason=None` and `recommendation=None` on all risks, which degrades the output quality invisibly.

**Evidence:**
```python
path = Path("backend/contract_profiles/risk_explanations.yaml")
if path.exists():  # silently skips if CWD is wrong
    ...
return {}
```

**Suggested Fix:**
Use `Path(__file__).resolve().parent.parent / "contract_profiles" / "risk_explanations.yaml"` to anchor the path to the file's own location, regardless of CWD. Log a warning at startup (not per-call) if the file does not exist.

---

#### FINDING M-08
**Category:** Maintainability
**Severity:** Medium
**File:** `backend/services/contract_review_service.py`
**Lines:** 1353–1920 (`run()` method)

**Description:**
The `ContractReviewService.run()` method is 568 lines long and contains 10 numbered workflow steps plus multiple post-passes (G1–G7), each of which involves non-trivial logic. The method violates the Single Responsibility Principle and makes it very difficult to:
- Unit test individual steps in isolation
- Trace which post-pass caused a presence status change
- Add new post-passes without regression risk

Each post-pass (G1 through G7) is a separate detection strategy with its own logic, and they are all inlined into the same giant method.

**Suggested Fix:**
Extract each logical step into a private method: `_load_profile`, `_gather_evidence`, `_detect_clause_presence_for_all`, `_run_implicit_postpasses`, `_run_page_fallback`, `_build_risks`, `_build_executive_summary`, `_build_response`. Each method should return a structured result. The `run()` method becomes a coordinator of ~15 lines.

---

#### FINDING M-09
**Category:** Logic Bug
**Severity:** Medium
**File:** `backend/services/structured_clause_extraction.py`
**Lines:** 892–902 (`detect_document_type`)

**Description:**
`detect_document_type` uses a fragile heuristic that checks for "this agreement" or "party" in the first 3 pages to detect a contract. The word "party" appears in almost every legal document type (including court judgments, statutes, and correspondence). "this agreement" is more specific but still appears in recitals and preambles of non-contract documents.

More critically, `DocumentType.UNKNOWN` is detected when none of the patterns match, but `UNKNOWN` documents go through the same clause extraction pipeline as contracts because `operative_sections` in `__init__` includes `DocumentSection.UNKNOWN`. A statute or administrative letter that doesn't match the judgment or statute patterns will be treated as `UNKNOWN` and all its pages may be extracted as contractual clauses.

**Evidence:**
```python
if "this agreement" in sample or "party" in sample:
    return DocumentType.CONTRACT   # "party" matches too broadly
```

**Suggested Fix:**
Strengthen the contract detection pattern: require "this agreement" AND at least one party-definition phrase ("hereinafter referred to as", "first party", "second party") or at least two of the contract_entry_patterns. Add a minimum evidence threshold before classifying as CONTRACT.

---

#### FINDING M-10
**Category:** Edge Case
**Severity:** Medium
**File:** `backend/services/query_classifier.py`
**Lines:** 281 (`_missing_context`)

**Description:**
`_missing_context` checks whether "law" appears in the query to decide if jurisdiction is already specified. The word "law" is extremely common in legal queries and will suppress the jurisdiction missing-context flag for almost every query (e.g., "what does the termination clause say about notice period under the law?" incorrectly suppresses the clarification request). This makes `requires_clarification` effectively always `False`, defeating the purpose of the clarification mechanism.

**Evidence:**
```python
jurisdiction_keywords = ["jurisdiction", "saudi", "saudi arabia", "ksa", "country", "law"]
if not any(kw in query_lower for kw in jurisdiction_keywords):
    ...  # "law" in any query kills this branch
```

**Suggested Fix:**
Remove "law" from the `jurisdiction_keywords` list. The list should contain only explicit jurisdiction identifiers: "jurisdiction", "saudi", "saudi arabia", "ksa", "uae", "country", "qatar", etc. — not the generic word "law".

---

#### FINDING M-11
**Category:** Maintainability
**Severity:** Medium
**File:** `backend/services/contract_review_service.py`
**Lines:** 41–153 (`EXPECTED_CLAUSE_PATTERNS`)

**Description:**
`EXPECTED_CLAUSE_PATTERNS` contains duplicate and overlapping entries between `"probation"` and `"probation_period"` (both use the same keywords), and between `"remedies"` and `"remedies_injunction"` (significant keyword overlap). When both are in a profile's `expected_clauses`, both will match the same evidence, resulting in duplicate risk items and confusing executive summary entries for the same contractual concept.

**Evidence:**
```python
"probation": ["probation", "probation period", "probationary period", "trial period", "on probation"],
"probation_period": ["probation", "trial period", "probationary period"],  # complete subset
```

**Suggested Fix:**
Consolidate `probation` and `probation_period` into a single entry. Use `CANONICAL_CLAUSE_GROUPS` to group them at the display level. Apply the same to `remedies`/`remedies_injunction`. Audit all profile YAML files to ensure no profile lists both a canonical key and its duplicate.

---

#### FINDING M-12
**Category:** Security
**Severity:** Medium
**File:** `backend/services/chat_orchestrator.py`
**Lines:** 73–78 (`_INJECTION_STRIP` regex)

**Description:**
The prompt injection filter strips a small set of hardcoded phrases ("ignore all instructions", "act as", etc.). This is a best-effort filter that is easily bypassed by minor variations: "ign0re all instructions", "Ignore\nall\ninstructions", Unicode homoglyphs, or phrases not in the list ("override your constraints", "new system prompt", "DAN mode"). The filter provides a false sense of security.

**Evidence:**
```python
_INJECTION_STRIP = re.compile(
    r"(ignore (all |previous )?instructions?|"
    r"disregard|forget (your |all )?instructions?|"
    r"you are now|act as|pretend you|system prompt)",
    re.IGNORECASE,
)
```

**Suggested Fix:**
The chunk delimiter approach (`<<DOCUMENT_EXCERPT_START>>`) is actually more robust than substring stripping because it allows the prompt to instruct the model that everything between delimiters is untrusted user content. Ensure the system prompt explicitly tells the model to treat delimited content as potentially adversarial. Consider adding a semantic injection detector using the embedding service to flag chunks that are semantically similar to known injection patterns. Document that the current filter is partial.

---

#### FINDING M-13
**Category:** Edge Case
**Severity:** Medium
**File:** `backend/utils/file_parser.py`
**Lines:** 73–79 (OCR fallback exception handling)

**Description:**
In `parse_pdf_pages`, when selective OCR fails (line 74), the code silently falls through to a full-document OCR pass. If the full-document OCR also fails (line 79), the error is printed to stdout (`print(f"Warning: OCR extraction failed: ...")`) rather than logged via `logging`. For a service that needs audit trails and centralized log aggregation, `print()` statements are not captured by log handlers. The same pattern appears throughout `file_parser.py` and `structured_clause_extraction.py`.

**Evidence:**
```python
except Exception as full_ocr_error:
    print(f"Warning: OCR extraction failed: {str(full_ocr_error)}")
    print("Note: Install Tesseract OCR for scanned PDF support.")
```

**Suggested Fix:**
Replace all `print()` calls in service code with `logger.warning()` or `logger.error()` as appropriate. Create a module-level `logger = logging.getLogger(__name__)` in `file_parser.py`.

---

### LOW

---

#### FINDING L-01
**Category:** Maintainability
**Severity:** Low
**File:** `backend/config.py`
**Lines:** 53–56 (`SIMILARITY_THRESHOLDS`)

**Description:**
`SIMILARITY_THRESHOLDS` is defined as a dict with keys `"workflow"` and `"explorer"` but is never referenced in any of the reviewed files. The operational threshold used throughout the codebase is `settings.SIMILARITY_THRESHOLD` (singular). The dict is dead configuration that creates confusion about which threshold is actually in use.

**Suggested Fix:**
Either remove `SIMILARITY_THRESHOLDS` or wire it into the services that were presumably intended to use it.

---

#### FINDING L-02
**Category:** Maintainability
**Severity:** Low
**File:** `backend/services/retrieval_router.py`
**Lines:** 106 (comment says "Engine 4" but Engine 3 has not run yet)

**Description:**
The engine comments in `route()` are numbered 1, 2, 4, 6, 7, 3, 5 — out of order. Engine 3 runs after Engines 4, 6, and 7 in the code, but the comment at line 123 says "Engine 3 — Clause semantic (always, primary)". While the execution order is intentional (primary engine runs last to fill any gaps), the numbering is confusing to maintainers and the comment at line 106 ("Engine 4") follows line 100 ("Engine 2") with no Engine 3 in between.

**Suggested Fix:**
Renumber the engines to match execution order, or add a comment block at the top of `route()` explaining that engines are numbered by priority, not execution order, and that the "always-run" engines intentionally run last.

---

#### FINDING L-03
**Category:** Maintainability
**Severity:** Low
**File:** `backend/services/session_manager.py`
**Lines:** 169–171 (`_extract_defined_terms_from_answer`)

**Description:**
The defined-term extraction regex `r'["\u201c]([A-Z][A-Za-z\s]+)["\u201d]'` uses `\s` without limiting repetition. A multi-paragraph response with curly quotes but no actual defined terms (e.g., a quotation from a contract) will cause excessive backtracking or return very long "terms" spanning multiple sentences.

**Suggested Fix:**
Limit the inner group: `r'["\u201c]([A-Z][A-Za-z ]{1,50})["\u201d]'` and require the matched text to be title-case (no lowercase words longer than 3 chars).

---

#### FINDING L-04
**Category:** Maintainability
**Severity:** Low
**File:** `backend/services/query_rewriter.py`
**Lines:** 82 (comment says "last 6 messages")

**Description:**
The `QueryRewriter` uses `history[-6:]` regardless of `settings.SESSION_CONTEXT_WINDOW`. If `SESSION_CONTEXT_WINDOW` is changed in config, the rewriter's context window remains hardcoded at 6. This creates a subtle inconsistency where the orchestrator might inject 10 history messages but the rewriter only sees the last 6.

**Suggested Fix:**
Replace `history[-6:]` with `history[-settings.SESSION_CONTEXT_WINDOW * 2:]` to stay consistent with the orchestrator's windowing.

---

#### FINDING L-05
**Category:** Maintainability
**Severity:** Low
**File:** `backend/models/session.py`
**Lines:** 32, 44 (`datetime.utcnow`)

**Description:**
`datetime.utcnow()` is deprecated as of Python 3.12 and produces a naive (timezone-unaware) datetime. Pydantic v2 stores these without timezone info, which can cause comparison issues and is inconsistent with ISO 8601 conventions expected by API consumers.

**Suggested Fix:**
Replace `datetime.utcnow` with `datetime.now(timezone.utc)` and ensure the Pydantic model field type is `datetime` (Pydantic v2 handles tz-aware datetimes correctly in JSON serialization).

---

#### FINDING L-06
**Category:** Edge Case
**Severity:** Low
**File:** `backend/services/vector_store.py`
**Lines:** 247–252 (sentinel value -1.0 for threshold)

**Description:**
The `similarity_threshold` parameter uses `-1.0` as a sentinel meaning "use config default". This is a magic number that must be documented and remembered by every caller. It also prevents callers from intentionally passing a very low positive threshold close to 0. The parameter signature shows `Optional[float] = -1.0` which looks like a legitimate threshold value to a new developer.

**Suggested Fix:**
Replace the sentinel with an explicit `use_default: bool = True` parameter, or use Python's `sentinel = object()` pattern. Alternatively, rename the parameter to `similarity_threshold_override` and use `None` as the "use default" signal consistently (which is already how the RetrievalRouter calls it — passing `None`).

---

#### FINDING L-07
**Category:** Maintainability
**Severity:** Low
**File:** `backend/services/contract_review_service.py`
**Lines:** 311–328 (`DISPLAY_NAME_MAP`)

**Description:**
`DISPLAY_NAME_MAP` and `EXPECTED_CLAUSE_DISPLAY_NAMES` serve the same purpose (mapping clause key → display name) but use different key formats. `DISPLAY_NAME_MAP` uses camelCase-stripped slug keys (e.g., `"confidentialinformation"`), while `EXPECTED_CLAUSE_DISPLAY_NAMES` uses underscore-separated keys (e.g., `"confidentiality"`). The `_resolve_clause_display_name` function applies regex to extract the slug from a `clause_id`, then looks it up in `DISPLAY_NAME_MAP`. Any clause_id whose slug does not exactly match a key in `DISPLAY_NAME_MAP` (after regex cleanup) falls through to a less readable fallback. This dual-map approach is fragile and hard to maintain.

**Suggested Fix:**
Consolidate into a single display name lookup. `_resolve_clause_display_name` should try `EXPECTED_CLAUSE_DISPLAY_NAMES` first (using the clause type key extracted from the clause_id prefix), then fall back to humanizing the slug.

---

#### FINDING L-08
**Category:** Edge Case
**Severity:** Low
**File:** `backend/services/retrieval_router.py`
**Lines:** 280–300 (`_title_engine`)

**Description:**
`_title_engine` selects chunks where `page_number <= 1` OR `chunk_index <= 2`. For multi-document stores, this will include the first few chunks of every document in the store if `document_id` is `None`. The engine is only triggered for `summary`/`classification` queries where a `document_id` should always be provided, but there is no assertion or guard. If called without a `document_id`, it would inject title chunks from all documents into the context.

**Suggested Fix:**
Add `if not document_id: return []` at the top of `_title_engine` (and `_binary_engine`) since these engines make no sense cross-document. Log a warning if called without a document_id when these engines are triggered.

---

#### FINDING L-09
**Category:** Maintainability
**Severity:** Low
**File:** `backend/services/chat_orchestrator.py`
**Lines:** 97–98 (class `_CORSMiddlewareMarker`)

**Description:**
The `_CORSMiddlewareMarker` class in `main.py` exists purely to satisfy a test assertion that checks `type(m)` is a subclass of `CORSMiddleware`. This is a test-driven hack that makes the production code depend on test-specific assumptions. It also creates a confusing class definition that inherits from both `StarletteMiddleware` and `CORSMiddleware`, which are not designed to be combined this way.

**Suggested Fix:**
Fix the test to check the middleware's class attribute (e.g., `m.cls is CORSMiddleware`) rather than hacking the production code to satisfy the test's incorrect assumption. The test expectation is the problem, not the middleware registration.

---

#### FINDING L-10
**Category:** Edge Case
**Severity:** Low
**File:** `backend/services/structured_clause_extraction.py`
**Lines:** 628 (Roman numeral regex)

**Description:**
The Roman numeral clause detection pattern `r'^(I{1,4}|IV|IX|V|VI{0,3}|X{1,3})\.'` will match the word "IV" at the start of a line — which could be a clause number but could also be a drug name, abbreviation (Intravenous), or other text in a medical/pharma contract. Priority 3 has higher priority than Priority 5 (ALL CAPS), so a line beginning with "IV." will be detected as a Roman numeral clause even if it is a non-clause heading.

**Suggested Fix:**
Require the Roman numeral to be followed by at least one word character and a space before the heading text: `r'^(I{1,4}|IV|IX|V|VI{0,3}|X{1,3})\.\s+\w'`. Also add a minimum heading word count (e.g., at least 2 words) before triggering clause detection.

---

#### FINDING L-11
**Category:** Maintainability
**Severity:** Low
**File:** `backend/services/contract_review_service.py`
**Lines:** 1049 (relative path in `_load_risk_explanations`)

**Description:**
`logging.warning` is called (line 1055) via the module-level `logging` import, not via the `_log` logger defined at line 34. This bypasses the module logger's name and may produce log entries without the correct module prefix in structured logging systems.

**Suggested Fix:**
Replace `logging.warning(...)` with `_log.warning(...)` consistently throughout the file. The same issue appears at line 1829 (`logging.debug(...)`).

---

#### FINDING L-12
**Category:** Edge Case
**Severity:** Low
**File:** `backend/services/evidence_guardrail_service.py`
**Lines:** 318–325 (`_holistic_fallback`)

**Description:**
In `_holistic_fallback`, if any single chunk embedding fails, the method immediately returns `0.0` (line 318: `return 0.0` inside the except block). This means one bad chunk (e.g., a corrupted or empty chunk) causes the entire holistic fallback to report zero similarity, potentially triggering an incorrect "fail" decision even when the remaining chunks provide strong evidence.

**Evidence:**
```python
try:
    emb = self.embedding_service.embed_text(text)
    vectors.append(np.asarray(emb))
except Exception as exc:
    logger.warning("Guardrail holistic fallback: failed to embed chunk: %s", exc)
    return 0.0   # <-- bails out entirely on one chunk failure
```

**Suggested Fix:**
Replace `return 0.0` with `continue` so that failed chunk embeddings are skipped and the holistic score is computed from successfully embedded chunks. Only return `0.0` if ALL chunks fail to embed (which is already handled by the `if not vectors: return 0.0` guard below).

---

## Positive Aspects

The following elements represent genuinely good engineering decisions worth preserving:

1. **Audit log architecture** (`chat_orchestrator.py:AuditService`): The structured JSON audit log with `prompt_hash`, `answer_hash`, `chunk_hashes`, threshold snapshots, and latency is production-grade and well-designed. Full reconstructability of each LLM turn is excellent.

2. **Two-pass guardrail with regeneration** (`ValidationPipeline`): The weak-evidence → stricter regeneration → re-validate pattern is a thoughtful approach to hallucination reduction. The chunk-exact traceability (passing the same chunks to both generation and validation) is the right design.

3. **Dual deduplication key fallback** (`RetrievalRouter._merge`): The fallback to `_chunk_key(r)` when `chunk_id` is absent shows defensive coding for backward-compatible chunk metadata.

4. **Lazy vector store loading** (`VectorStore._ensure_loaded`): The lazy disk load pattern avoids I/O blocking at startup and correctly handles the first-use initialization.

5. **Embedding dimension mismatch check** (`VectorStore.add_chunks`): Raising `ValueError` on shape mismatch before adding to the index prevents silent data corruption.

6. **Score normalization in dual retrieval** (`RetrievalStrategy._normalize_scores`): Min-max normalization before fusion ensures that scores from two independent retrieval passes (original query vs. rewritten query) are comparable, which is a non-obvious but correct approach.

7. **Fail-closed clause extraction** (`StructuredClauseExtractionService`): The explicit `return []` with logging on every failure path ensures that extraction failures are visible and never silently produce partial results.

8. **Contract entry gate** (`_meets_contract_entry_gate`): The two-pattern requirement before starting clause extraction is a good heuristic to avoid extracting clauses from preamble/cover-letter pages.

9. **Legal hierarchy ranking** (`hierarchy_service.rank_by_authority`): Applying a domain-specific authority ordering (law > contract > policy) before returning search results is a sound RAG design decision for legal documents.

10. **`established_facts` cross-turn consistency** (`SessionManager._extract_established_facts` + `ChatOrchestratorService.chat`): The pattern of extracting high-confidence document-level facts from LLM answers and injecting them into subsequent generation queries is a sound approach to avoiding re-derivation errors across turns.

11. **Profile-driven contract review** (YAML contract profiles): Separating the "what clauses to look for" configuration from the service logic via YAML profiles makes the review engine configurable without code changes.

12. **Page-level fallback scan** (`_page_fallback_scan`): The last-resort raw-page keyword scan ensures that clauses in Schedules/Annexures (which the structured extractor might miss) are still detectable for presence checks.

---

## File-by-File Notes

| File | Key Concern |
|------|-------------|
| `main.py` | CORS wildcard (C-02); relative path for log; `_CORSMiddlewareMarker` test-hack (L-09) |
| `config.py` | Dead `SIMILARITY_THRESHOLDS` dict (L-01); generally clean |
| `query_classifier.py` | `_normalize` no word boundaries (M-01); "law" in jurisdiction check (M-10) |
| `retrieval_router.py` | Chunk key format mismatch with orchestrator (M-02); engine numbering (L-02); title/binary engine cross-document risk (L-08) |
| `rag_service.py` | CRITICAL: infinite recursion in `_search_without_translation` (C-01) |
| `session_manager.py` | Regex too narrow for fact extraction (M-06); `\s+` in term regex (L-03) |
| `chat_orchestrator.py` | Debug log artifacts (H-02); relative audit log path (H-02); refusal edge case (M-04); injection filter weakness (M-12) |
| `models/session.py` | `datetime.utcnow` deprecated (L-05); otherwise clean model definitions |
| `models/contract_review.py` | Clean, well-structured Pydantic models |
| `contract_review_service.py` | 568-line `run()` method (M-08); duplicate clause pattern keys (M-11); relative path for YAML (M-07); dual display name maps (L-07); `logging` vs `_log` inconsistency (L-11) |
| `structured_clause_extraction.py` | Token/chunk/embedding word guard too broad (H-05); `detect_document_type` fragile (M-09); Roman numeral regex (L-10) |
| `evidence_guardrail_service.py` | Debug log module-level artifact (H-03); dead code branch in score logic (M-03); holistic fallback bail-out (L-12) |
| `query_rewriter.py` | Hardcoded window size 6 (L-04); otherwise clean |
| `vector_store.py` | Pickle deserialization (H-01); index/metadata atomicity (M-05); sentinel `-1.0` (L-06) |
| `file_parser.py` | `print()` instead of logging (M-13); no issues with parse logic itself |

---

## Prioritized Fix Order

Recommended order based on severity and blast radius:

1. **C-01** — Fix infinite recursion in `rag_service._search_without_translation` (production-breaking)
2. **C-02** — Restrict CORS origins to explicit allowlist (security)
3. **H-01** — Replace pickle with JSON + numpy.save for vector store persistence (security)
4. **H-02/H-03** — Remove all `.cursor/debug.log` write blocks and fix audit log path (maintainability + production correctness)
5. **H-05** — Remove the chunk/token/embedding word guard from clause extraction (silent data loss)
6. **H-04** — Async-offload session summarization (latency / UX)
7. **M-01** — Add word-boundary matching to `_normalize` in query classifier (correctness)
8. **M-04** — Harden zero-chunk refusal path in orchestrator (hallucination risk)
9. **M-07** — Anchor risk_explanations.yaml path to `__file__` (silent feature degradation)
10. **M-08** — Decompose `ContractReviewService.run()` into sub-methods (maintainability)
11. **M-10** — Remove "law" from jurisdiction keyword list (correctness)
12. **M-02** — Standardize chunk key format across router and orchestrator (deduplication correctness)
13. **M-03** — Rewrite guardrail evidence score logic to remove dead branch (clarity)
14. **M-05** — Make `add_chunks` atomic with respect to index + metadata (data integrity)
15. **M-06**, **M-09**, **M-11**, **M-12**, **M-13** — Remaining medium findings
16. **L-01** through **L-12** — Low findings, address in normal development cycle

---

*End of review. Total findings: 2 Critical, 5 High, 13 Medium, 12 Low.*
