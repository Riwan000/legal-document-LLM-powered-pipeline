# RAG Evaluation Results

**Timestamp:** 2026-02-07T21:13:07.892298Z
**Document ID:** `DOC-0001`
**Total tests:** 11
**Release gate passed:** **No**

---

## Summary

| Metric | Count |
|--------|-------|
| Explorer pass | 0 / 11 |
| RAG QA pass | 11 / 11 |

### Failures by class

- **retrieval_miss**: 8

---

## Determinism

- **bilingual_termination_ar_en**: pass (runs=3)
- **common_notice_period_en**: pass (runs=3)

---

## Per-test results

| Test ID | Category | Explorer | RAG QA | Explorer failure | RAG failure |
|---------|----------|----------|--------|------------------|------------|
| bilingual_termination_ar_en | bilingual | fail | pass | retrieval_miss |  |
| common_notice_period_en | common | fail | pass | retrieval_miss |  |
| common_benefits_en | common | fail | pass | retrieval_miss |  |
| common_governing_law_en | common | fail | pass | retrieval_miss |  |
| cross_doc_leakage_1 | cross_doc_leakage | fail | pass |  |  |
| edge_probation_termination_en | edge | fail | pass | retrieval_miss |  |
| edge_immediate_termination_en | edge | fail | pass | retrieval_miss |  |
| negative_working_hours_en | negative | fail | pass |  |  |
| negative_language_priority_en | negative | fail | pass |  |  |
| stress_typo_en | stress | fail | pass | retrieval_miss |  |
| stress_compound_en | stress | fail | pass | retrieval_miss |  |

---

## How to run

```bash
# From project root, with optional document ID (default: first from /api/documents or test_doc)
python -m tests.rag_eval.run_eval [--document-id DOC_ID] [--no-determinism] [--output path.md]
```

**Test categories:** common, edge, negative, bilingual, stress, cross_doc_leakage.
**Modes:** Explorer (evidence-only) and RAG QA (answer with citations). Evidence-first rule: RAG may say not_specified only when Explorer returns not_found.

*End of report.*