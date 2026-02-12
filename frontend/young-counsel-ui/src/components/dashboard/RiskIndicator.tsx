type RiskLevel = "high" | "medium" | "low";

type RiskIndicatorProps = {
  level: RiskLevel;
  title: string;
  description: string;
  location?: string;
};

const levelConfig: Record<
  RiskLevel,
  { label: string; color: string; bg: string }
> = {
  high: {
    label: "HIGH RISK",
    color: "text-risk-high",
    bg: "bg-risk-high/10"
  },
  medium: {
    label: "MEDIUM RISK",
    color: "text-risk-medium",
    bg: "bg-risk-medium/10"
  },
  low: {
    label: "LOW RISK",
    color: "text-risk-low",
    bg: "bg-risk-low/10"
  }
};

export function RiskIndicator({
  level,
  title,
  description,
  location
}: RiskIndicatorProps) {
  const cfg = levelConfig[level];

  return (
    <section className="rounded-xl border border-border bg-background-elevated-soft/80 p-4 shadow-card">
      <div className="mb-2 flex items-center justify-between">
        <div
          className={[
            "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-semibold tracking-wide",
            cfg.bg,
            cfg.color
          ].join(" ")}
        >
          {cfg.label}
        </div>
      </div>
      <h3 className="text-sm font-semibold text-text">{title}</h3>
      <p className="mt-1 text-xs text-text-muted">{description}</p>
      {location && (
        <p className="mt-2 text-[11px] text-text-subtle">Clause {location}</p>
      )}
    </section>
  );
}

