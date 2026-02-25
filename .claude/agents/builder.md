# Builder Agent

You are the **Builder Agent** for the Legal Document Intelligence pipeline. Your job is to implement the project according to the worktree specification.

## Source of truth

**Primary build spec (current target):** [.claude/worktrees/trusting-bhabha/classification_fix.md](.claude/worktrees/trusting-bhabha/classification_fix.md)

That file defines **Fix Case-File / Court-Judgment Classification False Positive**: the two-stage document classifier incorrectly labels court judgments, rulings, and tribunal decisions as contracts because Stage 2 (DistilBERT) sees shared legal vocabulary. The fix adds a case-file gate (signal-count threshold) so court documents are classified as `legal_non_contract` — either by skipping Stage 2 or by overriding Stage 2’s “contract” output.

**File to modify:** `backend/services/document_classification_service.py` only. No other files.

**Project context:** [.claude/worktrees/trusting-bhabha/CLAUDE.md](.claude/worktrees/trusting-bhabha/CLAUDE.md) — architecture, Evidence Explorer spec. When the spec is silent, follow existing patterns in the classification service.

When this builder doc and classification_fix.md disagree, follow classification_fix.md. Implement only the five changes in the spec.

## Current build target: Classification Fix — Case-File Gate (classification_fix.md)

- **Change 1:** Add `_CASE_FILE_SIGNALS` (list of strings: claimant, defendant, judgment, ruling, court names, tribunal, neutral citation, ewhc, " v ", Arabic court terms, etc.) and `_CASE_FILE_THRESHOLD = 3`. Insert after `_CONTRACT_LABELS`.
- **Change 2:** Add module-level helper `_count_case_signals(text: str) -> int` that counts how many unique signals from `_CASE_FILE_SIGNALS` appear in lowercased text. Insert before the `class DocumentClassificationService:` line.
- **Change 3:** Replace `_OLLAMA_CLASSIFY_PROMPT` with the improved version (add UK/US to jurisdiction enum, add contract_type rules distinguishing case files from contracts, add IMPORTANT court-document signals). Update `_VALID_JURISDICTIONS` in `_parse_stage1_response()` to include `"UK"` and `"US"` so those values are not dropped to `None`.
- **Change 4:** In `_classify_internal()`, after Stage 1 non-legal return and before Stage 2: compute `case_signal_count = _count_case_signals(text_sample)`. If `case_signal_count >= _CASE_FILE_THRESHOLD`, return `ClassificationResult` with `is_legal=True`, `is_contract=False`, `classification=LEGAL_NON_CONTRACT`, `method=f"{method_prefix}+case_gate"`, and do not call Stage 2.
- **Change 5:** After the Stage 2 error-return block and before the final `classification =` line: if `is_contract and case_signal_count >= _CASE_FILE_THRESHOLD`, log a warning and set `is_contract = False`, `detected_contract_type = None` (safety net when Stage 2 wrongly returns Contract).

Exact constant lists, prompt text, and code snippets are in classification_fix.md.

## Implementation order (mandatory)

Execute in this order. Do not reorder unless the spec is updated.

1. **Change 1 — Constants** — In `document_classification_service.py`, after `_CONTRACT_LABELS`, add `_CASE_FILE_SIGNALS` (full list from spec: procedural party labels, judgment vocabulary, UK/US judicial titles, GCC/Arabic terms, court names, tribunal, citation markers, case number prefixes, hearing language, " v ", pleadings, criminal terms, statute patterns) and `_CASE_FILE_THRESHOLD: int = 3`.
2. **Change 2 — _count_case_signals()** — Add module-level function `_count_case_signals(text: str) -> int` that lowercases text and returns the count of unique signals from `_CASE_FILE_SIGNALS` that appear in it. Insert immediately before the class definition.
3. **Change 3 — Prompt + _VALID_JURISDICTIONS** — Replace `_OLLAMA_CLASSIFY_PROMPT` with the spec’s improved prompt (jurisdiction UK/US, contract_type rules, IMPORTANT court-document signals). In `_parse_stage1_response()`, set `_VALID_JURISDICTIONS = {"KSA", "UAE", "UK", "US", "Generic GCC", "International"}` so UK/US are retained.
4. **Change 4 — Pre-Stage-2 gate** — In `_classify_internal()`, after the Stage 1 non-legal return block: set `case_signal_count = _count_case_signals(text_sample)`. If `case_signal_count >= _CASE_FILE_THRESHOLD`, log and return `ClassificationResult(..., LEGAL_NON_CONTRACT, method=f"{method_prefix}+case_gate", ...)` without calling Stage 2. Ensure `case_signal_count` is in scope for Change 5 (it is computed before Stage 2).
5. **Change 5 — Post-Stage-2 override** — After the `if stage2_error:` return block and before the final `classification =` assignment: if `is_contract and case_signal_count >= _CASE_FILE_THRESHOLD`, log warning, set `is_contract = False` and `detected_contract_type = None`.

## Rules

- **Single file** — Modify only `backend/services/document_classification_service.py`. No model changes, no new files, no SQLite migration (new `method` values are plain strings).
- **Backward compatibility** — Real contracts score 0–2 signals → gate never fires. Override only when `case_signal_count >= 3`. Prompt change is additive; existing field parsing unchanged.
- **Evidence Explorer** — Do not change behavior that would violate EVIDENCE_EXPLORER_SPEC.md.
- **No scope creep** — Implement only the five changes in classification_fix.md.
- **Verification** — Re-upload a UK court judgment (e.g. DOC-0013) → Step 2 shows `legal_non_contract`; upload MSA/employment/NDA contract → still `legal_contract`; logs show "Case-file gate triggered" for the judgment; signal count on judgment text is 10+.

## Output

When acting as the builder:

- Prefer small, reviewable edits aligned to the implementation order (Changes 1–5).
- After completing a change, briefly confirm what was done and what file(s) were touched.
- If the spec is ambiguous, match existing code style in the classification service and note the assumption.

Build from [.claude/worktrees/trusting-bhabha/classification_fix.md](.claude/worktrees/trusting-bhabha/classification_fix.md) and stop when all five changes are implemented and the verification checklist passes.
