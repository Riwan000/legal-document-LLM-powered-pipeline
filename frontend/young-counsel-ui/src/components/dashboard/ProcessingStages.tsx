/**
 * Processing stages: reflects backend workflow state only. No guessing, no animation.
 * Mapping: PENDING 0% grey | IN_PROGRESS 70% gold | COMPLETE 100% gold | FAILED 100% muted red
 */

export type StageStatusValue = "pending" | "in_progress" | "complete" | "failed";

export type WorkflowStateProp = {
  upload_extraction?: StageStatusValue;
  legal_analysis?: StageStatusValue;
  report_generation?: StageStatusValue;
};

const STAGE_LABELS = [
  { key: "upload_extraction" as const, title: "1. Upload & Extraction" },
  { key: "legal_analysis" as const, title: "2. Legal Analysis" },
  { key: "report_generation" as const, title: "3. Report Generation" },
];

function barWidth(status: StageStatusValue | undefined): string {
  if (!status || status === "pending") return "0%";
  if (status === "in_progress") return "70%";
  return "100%";
}

function barBg(status: StageStatusValue | undefined): string {
  if (!status || status === "pending") return "bg-border";
  if (status === "failed") return "bg-failed-muted";
  return "bg-accent";
}

function statusLabel(status: StageStatusValue | undefined): string {
  if (!status || status === "pending") return "PENDING";
  if (status === "in_progress") return "IN PROGRESS";
  if (status === "complete") return "COMPLETE";
  if (status === "failed") return "FAILED";
  return "PENDING";
}

type Props = {
  workflowState?: WorkflowStateProp | null;
  indeterminate?: boolean;
};

export function ProcessingStages({ workflowState, indeterminate }: Props) {
  if (indeterminate || !workflowState) {
    return (
      <div className="rounded-xl border border-border bg-background-elevated-soft/80 p-4">
        <p className="text-xs text-text-muted">
          Current Stage: Legal Analysis — Status: Indeterminate
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-background-elevated-soft/80 p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        {STAGE_LABELS.map(({ key, title }) => {
          const status = (workflowState[key] ?? "pending") as StageStatusValue;
          return (
            <div key={key} className="flex-1">
              <div className="mb-1 text-xs font-medium text-text-subtle">
                {title}
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-background-elevated">
                <div
                  className={`h-full rounded-full ${barBg(status)}`}
                  style={{ width: barWidth(status) }}
                />
              </div>
              <div className="mt-2 text-[10px] uppercase tracking-wide text-text-subtle">
                {statusLabel(status)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
