const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {})
    }
  });

  if (!res.ok) {
    throw new Error(`API request failed: ${res.status} ${res.statusText}`);
  }

  return (await res.json()) as T;
}

export type DocumentSummary = {
  document_id: string;
  filename: string;
};

/** Backend returns { documents: DocumentListItem[] }; we normalize to DocumentSummary[]. */
export async function listDocuments(): Promise<DocumentSummary[]> {
  const res = await apiFetch<{ documents: Array<{ document_id: string; original_filename?: string; display_name?: string }> }>("/api/documents");
  const list = Array.isArray(res?.documents) ? res.documents : [];
  return list.map((d) => ({
    document_id: d.document_id,
    filename: d.original_filename ?? d.display_name ?? d.document_id
  }));
}

export async function deleteDocument(document_id: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(document_id)}`, {
    method: "DELETE"
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail =
      typeof (err as any)?.detail === "string"
        ? (err as any).detail
        : (err as any)?.detail?.message ?? res.statusText;
    throw new Error(detail || `Delete failed: ${res.status}`);
  }
}

export async function getStats() {
  return apiFetch<Record<string, unknown>>("/api/stats");
}

export async function uploadDocument(
  file: File,
  options?: { document_type?: "document" | "template"; display_name?: string }
) {
  const form = new FormData();
  form.append("file", file);
  form.append("document_type", options?.document_type ?? "document");
  if (options?.display_name?.trim()) {
    form.append("display_name", options.display_name.trim());
  }

  const res = await fetch(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body: form
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = typeof err?.detail === "string" ? err.detail : err?.detail?.message ?? res.statusText;
    throw new Error(detail || `Upload failed: ${res.status}`);
  }

  return (await res.json()) as Record<string, unknown>;
}

/** Contract Review workflow. Form body. Returns WorkflowContext (intermediate_results.contract_review.response). */
export async function runContractReview(params: {
  contract_id: string;
  contract_type?: string;
  jurisdiction?: string | null;
  review_depth?: string;
}): Promise<ContractReviewWorkflowResult> {
  const body = new URLSearchParams();
  body.set("contract_id", params.contract_id);
  body.set("contract_type", params.contract_type ?? "employment");
  if (params.jurisdiction != null && params.jurisdiction !== "") {
    body.set("jurisdiction", params.jurisdiction);
  }
  body.set("review_depth", params.review_depth ?? "standard");

  const res = await fetch(`${API_BASE_URL}/api/contract-review`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString()
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = typeof err?.detail === "string" ? err.detail : err?.detail?.error?.message ?? res.statusText;
    throw new Error(detail || `Contract review failed: ${res.status}`);
  }

  const ctx = (await res.json()) as WorkflowContextEnvelope;
  const ir = ctx?.intermediate_results ?? {};
  const resp = ir["contract_review.response"] ?? ir.contract_review?.response ?? null;
  return { context: ctx, response: resp };
}

/** GET workflow state by workflow_id. Returns indeterminate on 404 or error. */
export async function getWorkflowState(
  workflow_id: string
): Promise<WorkflowStateFromAPI | { indeterminate: true }> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/workflow/${encodeURIComponent(workflow_id)}/state`);
    if (!res.ok) return { indeterminate: true };
    return (await res.json()) as WorkflowStateFromAPI;
  } catch {
    return { indeterminate: true };
  }
}

export type WorkflowStateFromAPI = {
  upload_extraction?: string;
  legal_analysis?: string;
  report_generation?: string;
};

type WorkflowContextEnvelope = {
  status: string;
  workflow_id?: string;
  workflow_state?: WorkflowStateFromAPI;
  error?: { code: string; message: string; step?: string; details?: unknown };
  intermediate_results?: Record<string, unknown>;
};

export type ContractReviewWorkflowResult = {
  context: WorkflowContextEnvelope;
  response: ContractReviewResponse | null;
};

export type ContractReviewResponse = {
  workflow_id?: string;
  document_id?: string;
  contract_type?: string;
  jurisdiction?: string;
  risks?: RiskItem[];
  not_detected_clauses?: string[];
  evidence?: ClauseEvidenceBlock[];
  executive_summary?: ExecutiveSummaryItem[];
  document_classification_warning?: string;
  used_implicit_or_distributed_logic?: boolean;
  disclaimer?: string;
};

export type RiskItem = {
  severity: string;
  status?: string;
  description: string;
  clause_types?: string[];
  clause_ids?: string[];
  display_names?: string[];
  page_numbers?: number[];
};

export type ClauseEvidenceBlock = {
  clause_id?: string;
  page_number?: number;
  raw_text?: string;
  clean_text?: string;
  display_name?: string;
  structure_class?: string;
  semantic_label?: string;
  is_non_contractual?: boolean;
};

export type ExecutiveSummaryItem = {
  text: string;
  severity?: string;
  related_risk_indices?: number[];
};

/** Evidence Explorer: snippets only (no LLM). Modes: text | clauses | both */
export async function exploreEvidence(params: {
  document_id: string;
  query: string;
  top_k?: number;
  mode?: "text" | "clauses" | "both";
}): Promise<ExploreEvidenceResult> {
  const body = new URLSearchParams();
  body.set("document_id", params.document_id);
  body.set("query", params.query);
  if (params.top_k != null) body.set("top_k", String(params.top_k));
  body.set("mode", params.mode ?? "both");

  const res = await fetch(`${API_BASE_URL}/api/explore-evidence`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString()
  });

  if (res.status === 422) {
    const err = await res.json().catch(() => ({}));
    const msg = err?.detail?.error?.message ?? err?.detail ?? res.statusText;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  if (!res.ok) throw new Error(`Evidence search failed: ${res.status} ${res.statusText}`);

  return (await res.json()) as ExploreEvidenceResult;
}

export type ExploreEvidenceResult = {
  status?: string;
  results?: { page_number?: number; score?: number; source_type?: string; text_snippet?: string; clause_id?: string }[];
  reason?: string;
  debug?: Record<string, unknown>;
};

/** RAG Answer Explorer: answer + citations */
export async function exploreAnswer(params: {
  document_id: string;
  query: string;
  top_k?: number;
  response_language?: string | null;
}): Promise<ExploreAnswerResult> {
  const body = new URLSearchParams();
  body.set("document_id", params.document_id);
  body.set("query", params.query);
  if (params.top_k != null) body.set("top_k", String(params.top_k));
  if (params.response_language?.trim()) body.set("response_language", params.response_language.trim());

  const res = await fetch(`${API_BASE_URL}/api/explore-answer`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString()
  });

  if (!res.ok) throw new Error(`Answer search failed: ${res.status} ${res.statusText}`);
  return (await res.json()) as ExploreAnswerResult;
}

export type ExploreAnswerResult = {
  answer?: string;
  status?: string;
  confidence?: string;
  sources?: { page_number?: number; text?: string; display_name?: string; document_id?: string }[];
  citation?: string;
};

