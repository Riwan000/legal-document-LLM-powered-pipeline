# PageIndex RAG — Detailed Explanation, Implementation & System Impact

---

## 1. What Is PageIndex RAG?

PageIndex RAG is a **reasoning-based retrieval** approach that replaces semantic similarity search (embeddings + vector database) with an LLM that **reasons over a structured document index** to decide what to read, just like a human would.

Traditional RAG asks: *"Which chunks look semantically similar to this query?"*
PageIndex asks: *"Given this document's structure, which sections should I navigate to in order to answer this query?"*

### Core Idea

At ingestion time, instead of (or in addition to) embedding chunks into FAISS, the system builds a **JSON hierarchical index** — essentially a machine-readable Table of Contents:

```json
{
  "document_id": "DOC-0021",
  "title": "Master Service Agreement",
  "nodes": [
    {
      "id": "node_1",
      "title": "Definitions",
      "page": 1,
      "description": "Defines key terms used throughout the agreement",
      "children": [
        { "id": "node_1_1", "title": "Service Provider", "page": 1 },
        { "id": "node_1_2", "title": "Confidential Information", "page": 2 }
      ]
    },
    {
      "id": "node_2",
      "title": "Term and Termination",
      "page": 5,
      "description": "Duration of agreement, termination conditions, notice periods",
      "children": [...]
    }
  ]
}
```

At query time, the LLM:
1. Reads the entire index (fits in context window — it's just titles and descriptions, not full text)
2. Decides which `node_ids` are relevant to the query
3. Retrieves the full content of those sections
4. Determines if it has enough to answer, or loops back for more nodes
5. Generates a final answer

---

## 2. Does It Use Storage?

**Yes.** The "vectorless" claim only means no vector database. PageIndex still requires:

| What Is Stored | Our System (FAISS) | PageIndex |
|---|---|---|
| Embedding vectors | Yes (`.index` file, ~1.5 KB/chunk × dim) | No |
| FAISS metadata pickle | Yes (`.metadata.pkl`) | No |
| JSON hierarchical index | No | Yes (per document, ~10–50 KB) |
| Raw page/section text | Yes (inside metadata.pkl) | Yes (needed to retrieve section content) |
| Original document file | Yes (`data/documents/`) | Yes (same) |

**Net storage effect:** Slightly less total storage (no high-dimensional float32 vectors), but still requires persistent JSON index files and the original document text.

---

## 3. Latency Impact

This is where PageIndex has its biggest cost. More LLM calls = more latency, especially running **Ollama locally** with no batching.

### Per-Query Call Breakdown

| Stage | Current System | PageIndex |
|---|---|---|
| Embedding query | ~50ms (nomic-embed-text) | Not needed |
| FAISS search | ~5–20ms | Not needed |
| Index reasoning (LLM picks nodes) | Not needed | ~1,500–3,000ms per call |
| Section retrieval | Not needed | ~100ms disk read |
| Sufficiency check + loop | Not needed | 0–3 more LLM calls |
| Generation (LLM answer) | ~2,000–5,000ms | ~2,000–5,000ms |
| **Total** | **~2–6 seconds** | **~5–20+ seconds** |

### Why Ollama Makes This Worse

Cloud APIs (GPT-4, Claude) can handle multiple calls with low inter-call latency because of:
- GPU farms with request batching
- Low network overhead from co-located services

Our Ollama setup (local CPU/GPU inference) has:
- Full model load per call (no batching)
- Sequential inference — each iteration blocks the next
- No parallelism across reasoning steps

A 3-iteration PageIndex query on Ollama can take **15–25 seconds** where FAISS would take 3–5 seconds.

---

## 4. Where PageIndex Wins Over Our Current System

### Problems PageIndex Solves That FAISS Cannot

**a) Query-Knowledge Space Mismatch**

FAISS finds text that *looks similar* to the query, not text that *answers* the query.

Example: Query = *"Can the service provider subcontract work?"*
- FAISS retrieves chunks containing "subcontract", "third party", "work" — might miss the governing clause if it uses different vocabulary like "assignment of obligations"
- PageIndex reasons: "subcontracting → look at Assignment and Delegation clauses" and navigates directly there

**b) Cross-Clause References**

Legal documents are full of internal references:
- *"Subject to clause 14.3..."*
- *"As defined in Schedule 2..."*
- *"Notwithstanding the provisions of Section 8..."*

FAISS treats each chunk independently. PageIndex can follow these references by re-navigating the index to fetch Schedule 2 after reading the clause that references it.

**c) Structural Awareness**

FAISS has no concept of document hierarchy. Our retrieval engines (clause_title, title_engine) partially address this, but they're heuristic. PageIndex *understands* that a document has a hierarchy and reasons about it.

**d) Multi-Part Queries**

Query: *"What are the payment terms and what happens if payment is late?"*
- FAISS: retrieves payment-adjacent chunks, may get partial answer
- PageIndex: navigates to Payment clause AND Late Payment/Penalties clause in one reasoning step

---

## 5. Current System Architecture (Before PageIndex)

```
User Query
    │
    ▼
QueryClassifier  ───────────────────────────────────────────────────────
(8 intents: definition_lookup, binary, summary, classification, etc.)   │
    │                                                                     │
    ▼                                                                     │
QueryRewriter                                                            │
(reformulates query for retrieval clarity)                              │
    │                                                                     │
    ▼                                                                     │
RetrievalRouter (7-engine fusion)                                        │
    ├── Engine 1: definition_engine     (keyword scan, unit_type=def)   │
    ├── Engine 2: clause_title_engine   (BM25-simplified, clause ref)   │
    ├── Engine 3: clause_semantic       (FAISS cosine, weight=1.0)      │
    ├── Engine 4: category_engine       (legal_category metadata scan)  │
    ├── Engine 5: page_fallback_engine  (FAISS on page_chunks, w=0.4)   │
    ├── Engine 6: title_engine          (first-page chunks, w=0.9)      │
    └── Engine 7: binary_engine         (keyword scan + page diversity) │
    │                                                                     │
    ▼                                                                     │
Score Fusion (weighted, deduplicated by chunk_id)                       │
    │                                                                     │
    ▼                                                                     │
Top-K Chunks (text + metadata)                                          │
    │                                                                     │
    ▼                                                                     │
RAGService.generate()                                                    │
(prompt assembly + Ollama LLM call)                                     │
    │                                                                     │
    ▼                                                                     │
EvidenceGuardrailService (2-pass validation)                            │
    │                                                                     │
    ▼                                                                     │
Answer + Citations to User                                              │
```

**Storage footprint per document:**
- FAISS index vectors: `n_chunks × embedding_dim × 4 bytes` (float32)
- metadata.pkl: all chunk text + metadata (typically 2–10 MB per 50-page doc)

---

## 6. What PageIndex Would Look Like in This System

### Architecture Change

```
Ingestion Pipeline (changes)
    │
    ├── [Existing] FAISS embedding + storage
    └── [NEW] PageIndexBuilder → JSON index → data/pageindex/{document_id}.json

Query Pipeline (changes)
    │
    ▼
QueryClassifier (unchanged)
    │
    ▼
Intent Router
    ├── Simple factual / definition     → [Existing] RetrievalRouter (FAISS, fast)
    ├── Binary yes/no                   → [Existing] binary_engine (fast)
    └── Complex / cross-clause / multi  → [NEW] PageIndexReasoner
            │
            ├── Step 1: Load JSON index into LLM context
            ├── Step 2: LLM selects relevant node_ids
            ├── Step 3: Fetch section text for selected nodes
            ├── Step 4: LLM checks sufficiency, loops if needed (max 3 iters)
            └── Step 5: Pass retrieved sections to RAGService.generate()
```

### New Files Required

```
backend/
├── services/
│   ├── pageindex_builder.py       # Builds JSON index from ingested pages
│   ├── pageindex_reasoner.py      # LLM-guided iterative section retrieval
│   └── retrieval_router.py        # Add pageindex routing branch (modified)
├── models/
│   └── pageindex.py               # PageIndexNode, PageIndexDocument models
data/
└── pageindex/                     # Persisted JSON indexes (one per document)
    ├── DOC-0021.json
    ├── DOC-0024.json
    └── ...
```

### PageIndexBuilder Logic

During ingestion (`document_ingestion.py`), after chunking, a new step:
1. Collect all chunks with `unit_type in ("clause", "title", "definition")`
2. Group by `parent_clause_id` to reconstruct hierarchy
3. For each clause node, generate a short description (via LLM or heuristic keyword extraction)
4. Serialize to JSON and save to `data/pageindex/{document_id}.json`

### PageIndexReasoner Logic

```
load_index(document_id)
    → read JSON from disk

iteration = 0
selected_content = []

while iteration < MAX_ITERATIONS:
    prompt = build_index_navigation_prompt(query, index_json, selected_content)
    response = ollama.generate(model, prompt)
    node_ids = parse_node_ids(response)

    if node_ids == [] or response.is_sufficient:
        break

    for node_id in node_ids:
        section_text = fetch_section_text(document_id, node_id)
        selected_content.append(section_text)

    iteration += 1

return selected_content  # passed to RAGService.generate()
```

---

## 7. Metric-by-Metric Impact on Our System

### 7.1 Retrieval Accuracy

| Query Type | Current FAISS Accuracy | PageIndex Accuracy | Winner |
|---|---|---|---|
| Exact clause reference ("clause 5.2") | High (clause_title_engine) | High | Tie |
| Definition lookups | High (definition_engine) | High | Tie |
| Simple keyword factual | Medium-High | Medium (extra reasoning overhead) | FAISS |
| Cross-clause reasoning | Low | High | PageIndex |
| "Does this contract have X?" | Medium (binary_engine) | High | PageIndex |
| Schedule/annex content | Low (often missed) | High (explicit navigation) | PageIndex |
| Multi-part questions | Medium | High | PageIndex |
| Arabic/bilingual docs | Medium (embedding quality varies) | High (structure-guided) | PageIndex |

### 7.2 Latency

| Scenario | Current | PageIndex | Delta |
|---|---|---|---|
| Simple factual (1 engine) | ~2–3s | N/A (routes to FAISS) | 0 |
| Complex legal question | ~3–5s | ~8–15s | +5–10s |
| Contract review (full doc) | ~30–60s | ~60–120s | +30–60s |
| Multi-turn chat | ~3–5s/turn | ~8–15s/turn | +5–10s/turn |

### 7.3 Storage

| Artifact | Current Size (50-page doc) | With PageIndex |
|---|---|---|
| FAISS index | ~2–5 MB | ~2–5 MB (kept) |
| metadata.pkl | ~3–8 MB | ~3–8 MB (kept) |
| pageindex JSON | None | ~20–80 KB (tiny) |
| **Total** | **~5–13 MB** | **~5–13 MB + 80 KB** |

Storage overhead is negligible. The JSON index is small because it only stores titles, IDs, and short descriptions — not full text.

### 7.4 Memory (RAM)

- FAISS index is loaded once at startup and kept in RAM (currently ~50–200 MB for a loaded index)
- PageIndex JSON is loaded per query and is tiny (~50 KB in RAM per doc during reasoning)
- No significant change to RAM footprint

### 7.5 LLM Call Count per Query

| Pipeline Stage | Current | PageIndex (complex query) |
|---|---|---|
| OCR correction (ingestion only) | 1 | 1 |
| Query rewriting | 1 | 1 |
| Index navigation (new) | 0 | 1–4 |
| Sufficiency check (new) | 0 | 1–3 |
| Generation | 1 | 1 |
| Evidence guardrail | 2 | 2 |
| **Total per query** | **4** | **6–11** |

### 7.6 Cost (Local Ollama)

No direct API cost since we use Ollama locally, but each extra LLM call = more:
- CPU/GPU utilization
- Electric consumption
- Thermal load
- Context token usage (the full JSON index is in context for reasoning calls)

---

## 8. Recommended Integration Strategy

Given the latency cost of PageIndex and our Ollama-local setup, a **hybrid routing strategy** is recommended: use PageIndex only where its accuracy advantage justifies the latency cost.

### Routing Decision Table

| Query Intent (from QueryClassifier) | Recommended Engine | Reason |
|---|---|---|
| `definition_lookup` | FAISS (definition_engine) | Fast, already accurate |
| `binary` | FAISS (binary_engine) | Already keyword-driven, fast |
| `clause_lookup` | FAISS (clause_title_engine) | Exact reference, fast |
| `summary` | FAISS (title_engine + semantic) | Page-level retrieval sufficient |
| `classification` | FAISS (title_engine) | First-page scan sufficient |
| `clause_comparison` | **PageIndex** | Cross-clause reasoning required |
| `multi_part` | **PageIndex** | Multiple sections needed |
| `schedule_annex` | **PageIndex** | FAISS consistently misses these |
| `cross_reference` | **PageIndex** | "Subject to clause X" chains |

### Where It Helps Most in This Project

1. **Contract Review Engine** (`contract_review_service.py`)
   - Currently uses `_page_fallback_scan` as last resort for missing clauses
   - PageIndex could replace this fallback with targeted section navigation
   - Latency is already high here (30–60s), so extra 10–20s is acceptable

2. **Chat Queries About Schedules / Appendices**
   - Our current engines frequently miss content in Schedule 1, Schedule 2, Annexures
   - `ANNEXURES_SCHEDULES` was previously excluded from `operative_sections` (fixed, but retrieval is still weak)
   - PageIndex navigation would directly target schedule nodes

3. **Arabic/Bilingual Documents**
   - Embedding quality for Arabic text with nomic-embed-text is lower than English
   - Structural reasoning is language-agnostic — the index uses headings that are often bilingual

---

## 9. What We Would NOT Change

- **FAISS + VectorStore** remains the primary retrieval engine for the majority of queries
- **RetrievalRouter** 7-engine fusion is kept and used for fast/simple queries
- **EvidenceGuardrailService** remains unchanged — applies regardless of retrieval method
- **SessionManager / ChatOrchestrator** — PageIndex is a retrieval layer change, not a session logic change
- **Embedding pipeline** — still needed for FAISS engines and as a fallback

---

## 10. Implementation Effort Estimate

| Component | Description | Complexity |
|---|---|---|
| `pageindex.py` models | `PageIndexNode`, `PageIndexDocument` Pydantic models | Low |
| `pageindex_builder.py` | Build JSON index from ingested chunks (heuristic, no LLM needed) | Medium |
| Ingestion hook | Call builder after chunking in `document_ingestion.py` | Low |
| `pageindex_reasoner.py` | Iterative LLM navigation loop with max-iter guard | High |
| Routing branch | Add PageIndex path in `retrieval_router.py` or `rag_service.py` | Medium |
| Contract review hook | Replace `_page_fallback_scan` with PageIndex call | Medium |
| Prompt engineering | Navigation prompt + sufficiency check prompt | Medium |
| **Total** | | **High overall** |

---

## 11. Summary

| Dimension | Verdict |
|---|---|
| Does it use storage? | Yes — JSON index files, but much smaller than FAISS vectors |
| Does it increase latency? | Yes — significantly, especially on local Ollama (+5–15s per complex query) |
| Does it improve accuracy? | Yes — for cross-clause, schedule, and multi-part queries |
| Should we replace FAISS? | No — hybrid approach: FAISS for simple, PageIndex for complex |
| Best use case in this project | Contract review fallback + schedule/annex retrieval + cross-reference chains |
| Risk | High latency for conversational chat; needs per-intent routing to avoid degrading UX |
