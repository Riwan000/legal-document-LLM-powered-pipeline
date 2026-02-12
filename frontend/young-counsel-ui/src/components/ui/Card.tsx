import type { ReactNode } from "react";

type CardProps = {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
};

export function Card({ title, subtitle, children, className }: CardProps) {
  return (
    <section
      className={[
        "rounded-xl border border-border bg-background-elevated-soft/80 p-5 shadow-card",
        className
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {(title || subtitle) && (
        <header className="mb-3">
          {title && (
            <h2 className="text-sm font-semibold tracking-wide text-text">
              {title}
            </h2>
          )}
          {subtitle && (
            <p className="mt-1 text-xs text-text-muted">{subtitle}</p>
          )}
        </header>
      )}
      <div className="text-sm text-text-muted">{children}</div>
    </section>
  );
}

