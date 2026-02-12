import type { ReactNode } from "react";

type BadgeProps = {
  variant?: "default" | "soft" | "outline";
  children: ReactNode;
  className?: string;
};

export function Badge({ variant = "default", children, className }: BadgeProps) {
  const base =
    "inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-medium";

  const variants: Record<typeof variant, string> = {
    default: "border-accent bg-accent/10 text-accent-soft",
    soft: "border-border bg-background-elevated-soft text-text-muted",
    outline: "border-border text-text-muted"
  };

  return (
    <span className={[base, variants[variant], className].filter(Boolean).join(" ")}>
      {children}
    </span>
  );
}

