# Evidence Explorer & RAG Answer Explorer — Contract & Guardrails

This document is the **authoritative spec** for Evidence retrieval and RAG answers in the Document Explorer. Implementations and future optimizations **must** respect these rules.

---

## 1. Semantic vs lexical (Evidence Explorer)

**Rule: Lexical matches are always eligible results, even if semantic score is low or zero.**

- Semantic search **helps ranking, not eligibility**. OCR breaks embeddings; FAISS similarity is not recall-safe. Evidence Explorer is **recall-first**.
- Implementation must **union** semantic and lexical results. Any chunk or clause that matches lexically (or by heading) is included regardless of semantic score.
- Do **not** use semantic score to exclude a result that has a lexical or heading match.

---

## 2. No score cutoff (Evidence Explorer)

**Rule: Evidence Explorer does not apply a hard score cutoff once a lexical or heading match exists.**

- Scores may be used for **ranking** only. Once a chunk/clause is eligible (e.g. lexical or heading match), it must not be filtered out by a similarity threshold.
- Adding a threshold here would reintroduce false negatives (e.g. termination clause disappearing).

---

## 3. Debug block versioning

**Rule: All explorer debug payloads include a version field for schema evolution and tests.**

- Include: `"debug_version": "v1"` (or current version) in every Evidence Explorer and RAG Answer Explorer debug block.
- Debug schemas will evolve; versioning allows tests and tooling to rely on a known contract.

---

## 4. Evidence Explorer as authority for “absence”

**Rule: RAG Answer Explorer cannot return `status="not_specified"` unless Evidence Explorer would also return `status="not_found"`.**

- Evidence Explorer is the **authority** for whether evidence exists. RAG may say “not specified” only after Evidence Explorer has also failed to find anything.
- Implementation options:
  - **Option A:** In `/api/explore-answer` (or the single-document RAG path), call Evidence Explorer first; if it returns `not_found`, then RAG may return `not_specified`. If Evidence returns results, RAG must not return `not_specified`.
  - **Option B:** Reuse the same retrieval primitives and assert equivalence.
- Invariant: **RAGAnswerExplorer cannot return status="not_specified" unless evidence_explorer would also return status="not_found".**

---

## 5. Clause extraction version pinning

**Rule: Extracted clause storage and API responses include an extraction version for explainability and regression debugging.**

- Stored payload must include:
  - `document_id`
  - `extraction_version`: e.g. `"v1.0"` (pinned so Explorer behavior is explainable when extraction logic changes)
  - `clauses`: array of clause objects
- Enables regression debugging and clear contract when extraction logic is updated.

---

## 6. Evidence Explorer never merges snippets

**Rule: Evidence Explorer returns each chunk or clause independently; it never merges, summarizes, or stitches text.**

- Each result is one chunk or one clause. No server-side merging of snippets, no summarization, no stitching of multiple chunks into one string. This prevents accidental summarization and preserves verbatim evidence.

---

## Summary table

| # | Rule | Purpose |
|---|------|--------|
| 1 | Lexical always eligible | OCR/FAISS recall; avoid false negatives |
| 2 | No score cutoff for eligible results | Avoid threshold-based false negatives |
| 3 | `debug_version` in debug block | Schema evolution; tests |
| 4 | Evidence = authority for absence | RAG not_specified only when evidence not_found |
| 5 | `extraction_version` in clause store | Explainability; regression debugging |
| 6 | Never merge/summarize/stitch evidence | Verbatim recall-only |
