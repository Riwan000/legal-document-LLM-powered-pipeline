# Positive Architecture Notes

---

## 2026-02-25 — legal-document-LLM-powered-pipeline

- **YAML-driven domain profiles**: Separate configuration files for domain-specific rules (expected fields, risk weights, recommended actions) keep business logic out of code and allow non-engineers to update domain knowledge without deployments.

- **Deterministic LLM for high-stakes outputs**: Using temperature=0.0 and a fixed seed for all LLM calls in compliance/legal workflows ensures reproducibility — the same input always produces the same output, which is critical for audit trails and regression testing.

- **Two-pass generation with evidence escalation**: Generating a first answer and then checking evidence coverage before refusal (rather than refusing on the first weak signal) reduces false negatives while maintaining grounding. Only escalate to a "refused" response when regeneration with a stricter prompt also fails.

- **Verbatim evidence requirement for legal outputs**: Requiring verbatim text excerpts (not LLM paraphrases) for all citations in high-stakes domains (legal, compliance, medical) prevents hallucinated citations and makes outputs auditable.

- **Multi-engine retrieval fusion**: Combining multiple retrieval strategies (keyword exact match, semantic search, structural metadata lookup, fallback page-level search) with per-engine score weights produces more robust retrieval than any single engine, especially for structured documents with named sections.

- **Non-fatal optional pipeline stages**: Wrapping expensive optional enrichment steps (structure extraction, classification, sub-chunking) in `try/except` so core pipeline success is never blocked by enhancement failures is the right resilience pattern for multi-stage AI pipelines.

- **Structural metadata co-indexed with vectors**: Storing rich structural metadata (clause number, title, legal category, unit type) alongside each embedding at ingest time — rather than retrieving it from a separate store at query time — eliminates N+1 lookups and enables metadata filtering inside the retrieval loop.

- **Document hash for deduplication, not ID**: Using a SHA-256 content hash for deduplication while exposing a user-friendly sequential ID (DOC-####) to clients cleanly separates internal identity concerns from user-facing display.
