"use client";

import { useEffect, useState } from "react";
import { ProcessingStages } from "@/src/components/dashboard/ProcessingStages";
import { Card } from "@/src/components/ui/Card";
import { RiskIndicator } from "@/src/components/dashboard/RiskIndicator";
import {
  listDocuments,
  runContractReview,
  type DocumentSummary,
  type ContractReviewResponse,
  type RiskItem,
  type WorkflowStateFromAPI,
} from "@/src/lib/api";

const CONTRACT_TYPES = ["employment", "nda", "msa"] as const;
const JURISDICTIONS = [
  { value: "", label: "—" },
  { value: "Generic GCC", label: "Generic GCC" },
  { value: "KSA", label: "KSA" },
  { value: "UAE", label: "UAE" },
];
const REVIEW_DEPTHS = ["standard", "quick"] as const;

function severityLevel(s: string): "high" | "medium" | "low" {
  const lower = (s || "").toLowerCase();
  if (lower === "high") return "high";
  if (lower === "medium") return "medium";
  return "low";
}

export default function ContractReviewPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [contractType, setContractType] = useState<string>("employment");
  const [jurisdiction, setJurisdiction] = useState<string>("");
  const [reviewDepth, setReviewDepth] = useState<string>("standard");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ContractReviewResponse | null>(null);
  const [workflowState, setWorkflowState] = useState<WorkflowStateFromAPI | null>(null);

  useEffect(() => {
    listDocuments()
      .then((docs) => {
        const list = Array.isArray(docs) ? docs : [];
        setDocuments(list);
        if (list.length > 0 && !selectedId) {
          setSelectedId(list[0].document_id);
        }
      })
      .catch(() => setDocuments([]));
  }, [selectedId]);

  const handleRunReview = async () => {
    if (!selectedId) {
      setError("Please select a contract.");
      return;
    }
    setError(null);
    setResult(null);
    setWorkflowState(null);
    setLoading(true);
    try {
      const { context, response } = await runContractReview({
        contract_id: selectedId,
        contract_type: contractType,
        jurisdiction: jurisdiction || undefined,
        review_depth: reviewDepth,
      });
      setWorkflowState(context.workflow_state ?? null);
      setResult(response ?? null);
      if (context.status === "failed" && context.error) {
        setError(context.error.message || context.error.code || "Contract review failed.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Contract review failed.");
    } finally {
      setLoading(false);
    }
  };

  const risks: RiskItem[] = result?.risks ?? [];
  const evidence = result?.evidence ?? [];
  const executiveSummary = result?.executive_summary ?? [];
  const notDetected = result?.not_detected_clauses ?? [];

  const failedStage =
    workflowState?.legal_analysis === "failed"
      ? "Legal Analysis"
      : workflowState?.upload_extraction === "failed"
        ? "Upload & Extraction"
        : workflowState?.report_generation === "failed"
          ? "Report Generation"
          : null;
  const showFailureFooter = !!failedStage;

  return (
    <div className="flex flex-col gap-6 lg:flex-row">
      <div className="flex-1 space-y-5">
        <section className="space-y-3">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:flex-wrap">
            <div className="min-w-[200px]">
              <h2 className="text-sm font-semibold tracking-wide text-text">
                SELECT CONTRACT
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
            </div>
            <div className="min-w-[140px]">
              <label className="text-xs text-text-muted">Contract type</label>
              <select
                className="mt-1 h-9 w-full rounded-lg border border-border bg-background-elevated px-3 text-sm text-text outline-none focus:border-accent"
                value={contractType}
                onChange={(e) => setContractType(e.target.value)}
              >
                {CONTRACT_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>
            <div className="min-w-[140px]">
              <label className="text-xs text-text-muted">Jurisdiction</label>
              <select
                className="mt-1 h-9 w-full rounded-lg border border-border bg-background-elevated px-3 text-sm text-text outline-none focus:border-accent"
                value={jurisdiction}
                onChange={(e) => setJurisdiction(e.target.value)}
              >
                {JURISDICTIONS.map((j) => (
                  <option key={j.value || "none"} value={j.value}>
                    {j.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="min-w-[120px]">
              <label className="text-xs text-text-muted">Review depth</label>
              <select
                className="mt-1 h-9 w-full rounded-lg border border-border bg-background-elevated px-3 text-sm text-text outline-none focus:border-accent"
                value={reviewDepth}
                onChange={(e) => setReviewDepth(e.target.value)}
              >
                {REVIEW_DEPTHS.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
            <button
              type="button"
              disabled={loading || !selectedId}
              onClick={handleRunReview}
              className="h-9 rounded-lg bg-accent px-4 text-sm font-semibold text-black disabled:opacity-50 hover:bg-accent-soft"
            >
              {loading ? "Running…" : "Run Contract Review"}
            </button>
          </div>
          <ProcessingStages
            workflowState={workflowState ?? undefined}
            indeterminate={!workflowState && !loading}
          />
        </section>

        {error && (
          <div className="rounded-lg border border-risk-high bg-risk-high/10 px-4 py-2 text-sm text-risk-high">
            {error}
          </div>
        )}

        {showFailureFooter && failedStage && (
          <div className="rounded-lg border border-failed-muted bg-failed-muted/20 px-4 py-3 text-sm text-text-muted">
            Analysis halted during {failedStage}. No partial conclusions should be relied upon.
          </div>
        )}

        {result?.document_classification_warning && (
          <div className="rounded-lg border border-risk-medium bg-risk-medium/10 px-4 py-2 text-sm text-risk-medium">
            {result.document_classification_warning}
          </div>
        )}

        <Card title="EXECUTIVE SUMMARY">
          {executiveSummary.length > 0 ? (
            <ul className="list-inside list-disc space-y-1 text-sm text-text-muted">
              {executiveSummary.map((item, i) => (
                <li key={i}>
                  {item.severity && (
                    <span className="font-medium text-text">{item.severity}: </span>
                  )}
                  {item.text}
                </li>
              ))}
            </ul>
          ) : result ? (
            <p className="text-sm text-text-muted">No executive summary items.</p>
          ) : (
            <p className="text-sm text-text-muted">
              Select a contract and run Contract Review to see the executive summary.
            </p>
          )}
        </Card>

        {notDetected.length > 0 && (
          <Card
            title={`Clauses not detected (${result?.contract_type ?? "profile"})`}
          >
            <ul className="list-inside list-disc text-sm text-text-muted">
              {notDetected.map((name, i) => (
                <li key={i}>{name}</li>
              ))}
            </ul>
          </Card>
        )}

        <Card title="EVIDENCE BLOCKS">
          {evidence.length > 0 ? (
            <div className="space-y-4">
              {evidence.map((ev, i) => (
                <div key={i}>
                  <p className="text-xs font-semibold text-text-subtle">
                    {ev.display_name || ev.clause_id || "Excerpt"} (Page{" "}
                    {ev.page_number ?? "—"})
                  </p>
                  <p className="mt-1 text-sm text-text-muted">
                    {ev.clean_text || ev.raw_text || "—"}
                  </p>
                </div>
              ))}
            </div>
          ) : result ? (
            <p className="text-sm text-text-muted">No clause evidence blocks.</p>
          ) : (
            <p className="text-sm text-text-muted">
              Run Contract Review to see evidence by clause.
            </p>
          )}
        </Card>
      </div>

      <div className="w-full space-y-4 lg:w-80">
        {risks.length > 0 ? (
          risks.map((r, i) => (
            <RiskIndicator
              key={i}
              level={severityLevel(r.severity)}
              title={
                (r.display_names?.length
                  ? r.display_names.join(", ")
                  : r.clause_ids?.join(", ")) || r.description.slice(0, 40)
              }
              description={r.description}
              location={
                r.page_numbers?.length
                  ? `${r.display_names?.join(", ") || r.clause_ids?.join(", ") ?? ""} • Page ${r.page_numbers.join(", ")}`
                  : undefined
              }
            />
          ))
        ) : result ? (
          <p className="text-sm text-text-muted">No risks identified.</p>
        ) : (
          <>
            <RiskIndicator
              level="high"
              title="Indemnification"
              description="Run Contract Review to see risks for the selected contract."
              location="—"
            />
          </>
        )}
      </div>
    </div>
  );
}
