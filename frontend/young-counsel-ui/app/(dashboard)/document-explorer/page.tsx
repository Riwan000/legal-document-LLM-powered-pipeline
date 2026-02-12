"use client";

import { useEffect, useState } from "react";
import { Card } from "@/src/components/ui/Card";
import {
  listDocuments,
  exploreEvidence,
  exploreAnswer,
  deleteDocument,
  type DocumentSummary,
  type ExploreEvidenceResult,
  type ExploreAnswerResult,
} from "@/src/lib/api";

const MODES = [
  { value: "both", label: "Both (clauses + text)" },
  { value: "text", label: "Text chunks" },
  { value: "clauses", label: "Extracted clauses" },
] as const;

export default function DocumentExplorerPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [mode, setMode] = useState<"both" | "text" | "clauses">("both");
  const [tab, setTab] = useState<"evidence" | "answer">("evidence");

  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceResult, setEvidenceResult] = useState<ExploreEvidenceResult | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);

  const [answerLoading, setAnswerLoading] = useState(false);
  const [answerResult, setAnswerResult] = useState<ExploreAnswerResult | null>(null);
  const [answerError, setAnswerError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    listDocuments()
      .then((list) => {
        const arr = Array.isArray(list) ? list : [];
        setDocuments(arr);
        if (arr.length > 0) {
          setSelectedId(arr[0].document_id);
        }
      })
      .catch(() => setDocuments([]));
  }, []);

  const handleSearchEvidence = async () => {
    if (!selectedId || !query.trim()) return;
    setEvidenceError(null);
    setEvidenceResult(null);
    setEvidenceLoading(true);
    try {
      const res = await exploreEvidence({
        document_id: selectedId,
        query: query.trim(),
        top_k: topK,
        mode,
      });
      setEvidenceResult(res);
    } catch (e) {
      setEvidenceError(e instanceof Error ? e.message : "Evidence search failed.");
    } finally {
      setEvidenceLoading(false);
    }
  };

  const handleGetAnswer = async () => {
    if (!selectedId || !query.trim()) return;
    setAnswerError(null);
    setAnswerResult(null);
    setAnswerLoading(true);
    try {
      const res = await exploreAnswer({
        document_id: selectedId,
        query: query.trim(),
        top_k: topK,
      });
      setAnswerResult(res);
    } catch (e) {
      setAnswerError(e instanceof Error ? e.message : "Answer search failed.");
    } finally {
      setAnswerLoading(false);
    }
  };

  const handleDeleteSelected = async () => {
    if (!selectedId) return;
    if (!window.confirm("Are you sure you want to permanently delete this document? This action cannot be undone.")) {
      return;
    }
    setDeleteError(null);
    setDeleting(true);
    try {
      await deleteDocument(selectedId);
      setDocuments((prev) => prev.filter((d) => d.document_id !== selectedId));
      setSelectedId("");
      setEvidenceResult(null);
      setAnswerResult(null);
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : "Delete failed.");
    } finally {
      setDeleting(false);
    }
  };

  const evidenceResults = evidenceResult?.results ?? [];
  const answerSources = answerResult?.sources ?? [];
  const selectedDoc = documents.find((d) => d.document_id === selectedId);

  return (
    <div className="space-y-5">
      <section className="flex flex-col gap-3 md:flex-row md:items-end md:flex-wrap">
        <div className="min-w-[260px]">
          <h2 className="text-sm font-semibold tracking-wide text-text">
            SELECT DOCUMENT
          </h2>
          <select
            className="mt-1 h-9 w-full rounded-lg border border-border bg-background-elevated px-3 text-sm text-text outline-none focus:border-accent"
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {(!Array.isArray(documents) || documents.length === 0) && (
              <option value="">No documents available</option>
            )}
            {Array.isArray(documents) &&
              documents.map((doc) => (
                <option key={doc.document_id} value={doc.document_id}>
                  {doc.filename}
                </option>
              ))}
          </select>
          <div className="mt-2">
            <button
              type="button"
              disabled={!selectedId || deleting}
              onClick={handleDeleteSelected}
              className="rounded-lg border border-risk-high/40 bg-background-elevated px-3 py-1.5 text-[11px] font-semibold text-risk-high disabled:opacity-50 hover:bg-risk-high/5"
            >
              {deleting ? "Deleting…" : "Delete this document"}
            </button>
          </div>
        </div>
        <div className="flex-1 min-w-[200px]">
          <label className="text-xs text-text-muted">Query</label>
          <input
            type="text"
            className="mt-1 h-9 w-full rounded-lg border border-border bg-background-elevated px-3 text-sm text-text placeholder:text-text-subtle outline-none focus:border-accent"
            placeholder="e.g. Where is termination notice?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="flex items-end gap-2">
          <div>
            <label className="text-xs text-text-muted">Max results</label>
            <select
              className="mt-1 h-9 rounded-lg border border-border bg-background-elevated px-3 text-sm text-text outline-none focus:border-accent"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            >
              {[5, 10, 15, 20, 25].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
        </div>
      </section>

      <p className="text-xs text-text-muted">
        Explore evidence (snippets only) or get a RAG answer with citations for the selected document.
      </p>

      <div className="rounded-xl border border-border bg-background-elevated-soft/60 p-2">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex rounded-lg bg-background-elevated p-0.5">
            <button
              type="button"
              onClick={() => setTab("evidence")}
              className={`rounded-md px-3 py-1.5 text-xs font-medium ${tab === "evidence" ? "bg-accent text-black" : "text-text-muted hover:text-text"}`}
            >
              Evidence (snippets)
            </button>
            <button
              type="button"
              onClick={() => setTab("answer")}
              className={`rounded-md px-3 py-1.5 text-xs font-medium ${tab === "answer" ? "bg-accent text-black" : "text-text-muted hover:text-text"}`}
            >
              Answer (RAG)
            </button>
          </div>
          {tab === "evidence" && (
            <div className="flex items-center gap-2">
              <span className="text-[11px] uppercase tracking-wide text-text-subtle">
                Mode
              </span>
              <select
                className="h-8 rounded border border-border bg-background-elevated px-2 text-xs text-text outline-none focus:border-accent"
                value={mode}
                onChange={(e) => setMode(e.target.value as "both" | "text" | "clauses")}
              >
                {MODES.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {tab === "evidence" && (
        <div className="space-y-4">
          <button
            type="button"
            disabled={evidenceLoading || !selectedId || !query.trim()}
            onClick={handleSearchEvidence}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black disabled:opacity-50 hover:bg-accent-soft"
          >
            {evidenceLoading ? "Searching…" : "Search evidence"}
          </button>
          {evidenceError && (
            <p className="text-sm text-risk-high">{evidenceError}</p>
          )}
          {evidenceResult?.status === "not_found" && (
            <p className="text-sm text-text-muted">
              {evidenceResult.reason ?? "No relevant text or clauses found."}
            </p>
          )}
          {evidenceResults.length > 0 && (
            <div className="space-y-3">
              {evidenceResults.map((hit, i) => (
                <Card
                  key={i}
                  title={`Page ${hit.page_number ?? "—"} — Score ${(hit.score ?? 0).toFixed(2)} — ${hit.source_type ?? "chunk"}`}
                  subtitle={hit.clause_id ? `Clause: ${hit.clause_id}` : undefined}
                >
                  <p className="text-sm leading-relaxed text-text-muted whitespace-pre-wrap">
                    {hit.text_snippet ?? ""}
                  </p>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {tab === "answer" && (
        <div className="space-y-4">
          <button
            type="button"
            disabled={answerLoading || !selectedId || !query.trim()}
            onClick={handleGetAnswer}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black disabled:opacity-50 hover:bg-accent-soft"
          >
            {answerLoading ? "Generating…" : "Get answer"}
          </button>
          {answerError && (
            <p className="text-sm text-risk-high">{answerError}</p>
          )}
          {answerResult && (
            <>
              <Card title="Answer">
                <p className="text-sm leading-relaxed text-text-muted whitespace-pre-wrap">
                  {answerResult.answer ?? "No answer generated."}
                </p>
                <p className="mt-2 text-xs text-text-subtle">
                  Status: {answerResult.status ?? "—"} | Confidence: {answerResult.confidence ?? "—"}
                </p>
              </Card>
              {answerSources.length > 0 && (
                <Card title="Sources / citations">
                  <div className="space-y-3">
                    {answerSources.slice(0, 10).map((src, i) => (
                      <div
                        key={i}
                        className="rounded-lg border border-border bg-background-elevated-soft/50 p-3"
                      >
                        <p className="text-[11px] text-text-subtle">
                          Source {i + 1} — Page {src.page_number ?? "—"} • {src.display_name ?? src.document_id ?? "—"}
                        </p>
                        <p className="mt-1 text-sm text-text-muted">
                          {(src.text ?? "").slice(0, 800)}
                          {(src.text?.length ?? 0) > 800 ? "…" : ""}
                        </p>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </>
          )}
        </div>
      )}

      {deleteError && (
        <p className="text-sm text-risk-high">{deleteError}</p>
      )}

      {selectedDoc && (
        <p className="text-[11px] text-text-subtle">
          Source list: {selectedDoc.filename} (Document ID: {selectedDoc.document_id})
        </p>
      )}
    </div>
  );
}
