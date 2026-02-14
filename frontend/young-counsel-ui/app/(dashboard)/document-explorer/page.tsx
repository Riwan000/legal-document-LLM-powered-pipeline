"use client";

import { useEffect, useRef, useState } from "react";
import { Card } from "@/src/components/ui/Card";
import {
  listDocuments,
  exploreEvidence,
  exploreAnswer,
  deleteDocument,
  createChatSession,
  sendChatMessage,
  deleteChatSession,
  type DocumentSummary,
  type ExploreEvidenceResult,
  type ExploreAnswerResult,
  type ChatMode,
  type ChatMessage,
  type ChatMessageResponse,
} from "@/src/lib/api";

const EVIDENCE_MODES = [
  { value: "both", label: "Both (clauses + text)" },
  { value: "text", label: "Text chunks" },
  { value: "clauses", label: "Extracted clauses" },
] as const;

// ---------------------------------------------------------------------------
// Chat mode badge
// ---------------------------------------------------------------------------
function ModeBadge({ mode }: { mode: ChatMode }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide ${
        mode === "conversational"
          ? "bg-accent/10 text-accent"
          : "bg-border/50 text-text-muted"
      }`}
    >
      {mode === "conversational" ? "CONVERSATIONAL" : "STRICT"}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Evidence badge for chat answers
// ---------------------------------------------------------------------------
function EvidenceBadge({ score }: { score?: string }) {
  if (!score) return null;
  const colours: Record<string, string> = {
    strong: "bg-risk-low/10 text-risk-low",
    moderate: "bg-risk-medium/10 text-risk-medium",
    weak: "bg-risk-high/10 text-risk-high",
    none: "bg-risk-high/10 text-risk-high",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide ${colours[score] ?? "bg-border/50 text-text-muted"}`}
    >
      {score.toUpperCase()}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function DocumentExplorerPage() {
  // ---- shared state ----
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);

  // ---- mode toggle (top-level) ----
  const [chatMode, setChatMode] = useState<ChatMode>("strict");

  // ---- strict mode state (unchanged from before) ----
  const [evidenceMode, setEvidenceMode] = useState<"both" | "text" | "clauses">("both");
  const [tab, setTab] = useState<"evidence" | "answer">("evidence");
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceResult, setEvidenceResult] = useState<ExploreEvidenceResult | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [answerLoading, setAnswerLoading] = useState(false);
  const [answerResult, setAnswerResult] = useState<ExploreAnswerResult | null>(null);
  const [answerError, setAnswerError] = useState<string | null>(null);

  // ---- conversational mode state ----
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [sessionLastActive, setSessionLastActive] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // ---- shared delete state ----
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    listDocuments()
      .then((list) => {
        const arr = Array.isArray(list) ? list : [];
        setDocuments(arr);
        if (arr.length > 0) setSelectedId(arr[0].document_id);
      })
      .catch(() => setDocuments([]));
  }, []);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatMode === "conversational") {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory, chatMode]);

  // Reset session when document changes
  const handleSelectDocument = (id: string) => {
    setSelectedId(id);
    if (sessionId) {
      deleteChatSession(sessionId).catch(() => {});
    }
    setSessionId(null);
    setChatHistory([]);
    setChatError(null);
    setSessionLastActive(null);
    setEvidenceResult(null);
    setAnswerResult(null);
  };

  // Reset session when switching modes
  const handleModeSwitch = async (mode: ChatMode) => {
    if (sessionId) {
      await deleteChatSession(sessionId).catch(() => {});
    }
    setSessionId(null);
    setChatHistory([]);
    setChatError(null);
    setSessionLastActive(null);
    setChatMode(mode);
  };

  // ---------------------------------------------------------------------------
  // Strict mode handlers (unchanged logic)
  // ---------------------------------------------------------------------------
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
        mode: evidenceMode,
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

  // ---------------------------------------------------------------------------
  // Conversational mode handler
  // ---------------------------------------------------------------------------
  const handleSendChatMessage = async () => {
    if (!selectedId || !query.trim()) return;
    const userText = query.trim();
    setQuery("");
    setChatError(null);

    // Optimistic UI: add user message immediately
    const userMsg: ChatMessage = {
      role: "user",
      content: userText,
      timestamp: new Date().toISOString(),
    };
    setChatHistory((prev) => [...prev, userMsg]);
    setChatLoading(true);

    try {
      let currentSessionId = sessionId;

      // Create session on first message
      if (!currentSessionId) {
        const session = await createChatSession({
          document_id: selectedId,
          mode: "conversational",
        });
        currentSessionId = session.session_id;
        setSessionId(currentSessionId);
      }

      const response: ChatMessageResponse = await sendChatMessage(currentSessionId, {
        message: userText,
        mode: "conversational",
      });

      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: response.answer,
        timestamp: new Date().toISOString(),
        trace: response.trace,
      };
      setChatHistory((prev) => [...prev, assistantMsg]);
      setSessionLastActive(new Date().toLocaleTimeString());
    } catch (e) {
      setChatError(e instanceof Error ? e.message : "Chat request failed.");
    } finally {
      setChatLoading(false);
    }
  };

  const handleClearSession = async () => {
    if (sessionId) {
      await deleteChatSession(sessionId).catch(() => {});
    }
    setSessionId(null);
    setChatHistory([]);
    setChatError(null);
    setSessionLastActive(null);
  };

  // ---------------------------------------------------------------------------
  // Delete document
  // ---------------------------------------------------------------------------
  const handleDeleteSelected = async () => {
    if (!selectedId) return;
    if (!window.confirm("Permanently delete this document? This action cannot be undone.")) return;
    setDeleteError(null);
    setDeleting(true);
    try {
      await deleteDocument(selectedId);
      if (sessionId) await deleteChatSession(sessionId).catch(() => {});
      setDocuments((prev) => prev.filter((d) => d.document_id !== selectedId));
      setSelectedId("");
      setSessionId(null);
      setChatHistory([]);
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
      {/* ---- Document selector ---- */}
      <section className="flex flex-col gap-3 md:flex-row md:items-end md:flex-wrap">
        <div className="min-w-[260px]">
          <h2 className="text-sm font-semibold tracking-wide text-text">SELECT DOCUMENT</h2>
          <select
            className="mt-1 h-9 w-full rounded-lg border border-border bg-background-elevated px-3 text-sm text-text outline-none focus:border-accent"
            value={selectedId}
            onChange={(e) => handleSelectDocument(e.target.value)}
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

        {/* Query input */}
        <div className="flex-1 min-w-[200px]">
          <label className="text-xs text-text-muted">
            {chatMode === "conversational" ? "Message" : "Query"}
          </label>
          <input
            type="text"
            className="mt-1 h-9 w-full rounded-lg border border-border bg-background-elevated px-3 text-sm text-text placeholder:text-text-subtle outline-none focus:border-accent"
            placeholder={
              chatMode === "conversational"
                ? "Ask a follow-up question…"
                : "e.g. Where is termination notice?"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && chatMode === "conversational") handleSendChatMessage();
            }}
          />
        </div>

        {/* Max results (strict mode only) */}
        {chatMode === "strict" && (
          <div className="flex items-end gap-2">
            <div>
              <label className="text-xs text-text-muted">Max results</label>
              <select
                className="mt-1 h-9 rounded-lg border border-border bg-background-elevated px-3 text-sm text-text outline-none focus:border-accent"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              >
                {[5, 10, 15, 20, 25].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
          </div>
        )}
      </section>

      {/* ---- Mode toggle ---- */}
      <div className="rounded-xl border border-border bg-background-elevated-soft/60 p-2">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[11px] uppercase tracking-wide text-text-subtle font-semibold">Mode</span>
          <div className="flex rounded-lg bg-background-elevated p-0.5">
            <button
              type="button"
              onClick={() => handleModeSwitch("strict")}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                chatMode === "strict" ? "bg-accent text-black" : "text-text-muted hover:text-text"
              }`}
            >
              Evidence (Strict)
            </button>
            <button
              type="button"
              onClick={() => handleModeSwitch("conversational")}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                chatMode === "conversational" ? "bg-accent text-black" : "text-text-muted hover:text-text"
              }`}
            >
              Conversational (Evidence-Grounded)
            </button>
          </div>
          <ModeBadge mode={chatMode} />
          {chatMode === "conversational" && sessionLastActive && (
            <span className="text-[10px] text-text-subtle">Last active: {sessionLastActive}</span>
          )}
        </div>
      </div>

      {/* ---- Strict mode tabs ---- */}
      {chatMode === "strict" && (
        <>
          <p className="text-xs text-text-muted">
            Explore evidence (snippets only) or get a RAG answer with citations for the selected document.
          </p>

          <div className="rounded-xl border border-border bg-background-elevated-soft/60 p-2">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex rounded-lg bg-background-elevated p-0.5">
                <button
                  type="button"
                  onClick={() => setTab("evidence")}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium ${
                    tab === "evidence" ? "bg-accent text-black" : "text-text-muted hover:text-text"
                  }`}
                >
                  Evidence (snippets)
                </button>
                <button
                  type="button"
                  onClick={() => setTab("answer")}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium ${
                    tab === "answer" ? "bg-accent text-black" : "text-text-muted hover:text-text"
                  }`}
                >
                  Answer (RAG)
                </button>
              </div>
              {tab === "evidence" && (
                <div className="flex items-center gap-2">
                  <span className="text-[11px] uppercase tracking-wide text-text-subtle">Source</span>
                  <select
                    className="h-8 rounded border border-border bg-background-elevated px-2 text-xs text-text outline-none focus:border-accent"
                    value={evidenceMode}
                    onChange={(e) => setEvidenceMode(e.target.value as "both" | "text" | "clauses")}
                  >
                    {EVIDENCE_MODES.map((m) => (
                      <option key={m.value} value={m.value}>{m.label}</option>
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
              {evidenceError && <p className="text-sm text-risk-high">{evidenceError}</p>}
              {evidenceResult?.status === "not_found" && (
                <p className="text-sm text-text-muted">{evidenceResult.reason ?? "No relevant text or clauses found."}</p>
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
              {answerError && <p className="text-sm text-risk-high">{answerError}</p>}
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
                          <div key={i} className="rounded-lg border border-border bg-background-elevated-soft/50 p-3">
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
        </>
      )}

      {/* ---- Conversational mode ---- */}
      {chatMode === "conversational" && (
        <div className="space-y-4">
          <p className="text-xs text-text-muted">
            Evidence-grounded conversation. Every answer is retrieved from the document and validated before delivery.
            {sessionId && (
              <span className="ml-2 text-text-subtle">
                Session: <span className="font-mono text-[10px]">{sessionId.slice(0, 8)}…</span>
              </span>
            )}
          </p>

          {/* Chat thread */}
          <div className="min-h-[200px] max-h-[520px] overflow-y-auto rounded-xl border border-border bg-background-elevated-soft/40 p-4 space-y-4">
            {chatHistory.length === 0 && !chatLoading && (
              <p className="text-sm text-text-subtle text-center mt-8">
                Start the conversation by typing a question below.
              </p>
            )}
            {chatHistory.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-xl px-4 py-2.5 ${
                    msg.role === "user"
                      ? "bg-accent/10 text-text"
                      : "bg-background-elevated border border-border text-text"
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  {msg.role === "assistant" && msg.trace && (
                    <div className="mt-2 flex flex-wrap gap-1.5 items-center">
                      <EvidenceBadge score={msg.trace.evidence_score} />
                      {msg.trace.rewritten_query && msg.trace.rewritten_query !== msg.trace.original_query && (
                        <span className="text-[10px] text-text-subtle">
                          Rewritten: &ldquo;{msg.trace.rewritten_query.slice(0, 60)}{msg.trace.rewritten_query.length > 60 ? "…" : ""}&rdquo;
                        </span>
                      )}
                    </div>
                  )}
                  <p className="mt-1 text-[10px] text-text-subtle">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex justify-start">
                <div className="rounded-xl bg-background-elevated border border-border px-4 py-2.5">
                  <p className="text-sm text-text-muted animate-pulse">Retrieving and validating…</p>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {chatError && <p className="text-sm text-risk-high">{chatError}</p>}

          {/* Send controls */}
          <div className="flex gap-2">
            <button
              type="button"
              disabled={chatLoading || !selectedId || !query.trim()}
              onClick={handleSendChatMessage}
              className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black disabled:opacity-50 hover:bg-accent-soft"
            >
              {chatLoading ? "Thinking…" : "Send"}
            </button>
            {chatHistory.length > 0 && (
              <button
                type="button"
                onClick={handleClearSession}
                className="rounded-lg border border-border px-3 py-2 text-xs text-text-muted hover:text-text hover:border-text-muted"
              >
                Clear session
              </button>
            )}
          </div>
        </div>
      )}

      {deleteError && <p className="text-sm text-risk-high">{deleteError}</p>}

      {selectedDoc && (
        <p className="text-[11px] text-text-subtle">
          {selectedDoc.filename} (Document ID: {selectedDoc.document_id})
        </p>
      )}
    </div>
  );
}
