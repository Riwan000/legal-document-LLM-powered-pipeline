"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/contract-review", label: "Contract Review" },
  { href: "/document-explorer", label: "Document Explorer" },
  { href: "/upload", label: "Upload Document" },
  { href: "/about", label: "About" }
];

function SidebarLink({ href, label }: { href: string; label: string }) {
  // `usePathname` is a client-only hook; keep this small wrapper client-side.
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const pathname = usePathname();
  const active = pathname === href;

  return (
    <Link
      href={href}
      className={[
        "flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors",
        active
          ? "bg-background-elevated-soft text-text"
          : "text-text-muted hover:bg-background-elevated-soft/60 hover:text-text"
      ].join(" ")}
    >
      {label}
    </Link>
  );
}

function Sidebar() {
  return (
    <aside className="flex h-full flex-col border-r border-border bg-background-elevated px-4 py-6">
      <div className="mb-8">
        <div className="text-xs font-semibold tracking-[0.25em] text-text-subtle">
          YOUNG COUNSEL
        </div>
        <div className="mt-1 text-lg font-semibold text-text">
          Legal Intelligence Systems
        </div>
      </div>
      <nav className="flex flex-1 flex-col gap-1">
        {navItems.map((item) => (
          <SidebarLink key={item.href} href={item.href} label={item.label} />
        ))}
      </nav>
      <div className="mt-6 border-t border-border pt-4 text-[11px] text-text-subtle">
        <div>Documents are processed locally.</div>
        <div>This system does not provide legal advice.</div>
      </div>
    </aside>
  );
}

function TopBar({ title }: { title: string }) {
  return (
    <header className="flex items-center justify-between border-b border-border bg-background-elevated px-8 py-4">
      <div>
        <h1 className="text-lg font-semibold text-text">{title}</h1>
      </div>
      <div className="rounded-full border border-border bg-background-elevated-soft px-4 py-1 text-xs text-text-subtle">
        Private Deployment • Local Environment
      </div>
    </header>
  );
}

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen bg-background">
      <div className="hidden w-64 shrink-0 lg:block">
        <Sidebar />
      </div>
      <div className="flex min-h-screen flex-1 flex-col">
        <TopBar title="Young Counsel" />
        <main className="flex-1 bg-background px-4 py-6 text-text md:px-8">
          <div className="mx-auto max-w-6xl">{children}</div>
        </main>
      </div>
    </div>
  );
}

