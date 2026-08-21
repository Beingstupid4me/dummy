"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useConsole } from "@/lib/store";
import { clockTime } from "@/lib/format";

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/analysis", label: "Analysis" },
  { href: "/registry", label: "Registry" },
] as const;

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { toast, p99, clock, active, selected, apiLive, theme, toggleTheme } = useConsole();

  return (
    <div className="sf-shell">
      {toast ? (
        <div className="sf-toast pointer-events-none fixed top-4 left-1/2 z-30 rounded-md px-3.5 py-2 text-[12px] font-medium shadow-xl">
          {toast}
        </div>
      ) : null}

      <header className="border-b border-[var(--line)] bg-[var(--bg-panel)] px-5 py-3 transition-colors">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Link href="/" className="min-w-0 no-underline">
            <div className="sf-kicker">Bank of India · Cybershield 2026</div>
            <div className="mt-0.5 flex items-baseline gap-3">
              <span className="text-[18px] font-bold tracking-tight text-[var(--text-primary)]">
                SentinelFlow
              </span>
              <span className="hidden text-[12px] font-medium text-slate sm:inline">
                Real-Time Mule Prevention
              </span>
            </div>
          </Link>

          <nav className="flex rounded-md border border-[var(--line)] bg-[var(--bg-card)] p-0.5">
            {NAV.map((item, idx) => {
              const on =
                item.href === "/"
                  ? pathname === "/"
                  : pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={`nav-${item.href}-${idx}`}
                  href={item.href === "/analysis" ? `/analysis/${selected.id}` : item.href}
                  className={`rounded px-3.5 py-1.5 text-[12px] font-medium no-underline transition-colors ${
                    on
                      ? "bg-[var(--navy)] text-white shadow-sm dark:bg-foam dark:text-navy-deep"
                      : "text-slate hover:text-[var(--text-primary)]"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-3.5 text-[12px]">
            {/* White / Dark Mode Toggle Button */}
            <button
              onClick={toggleTheme}
              title={`Switch to ${theme === "dark" ? "White / Light" : "Dark"} Mode`}
              className="flex items-center gap-1.5 rounded-md border border-[var(--line)] bg-[var(--bg-card)] px-2.5 py-1.5 text-[11px] font-medium text-[var(--text-primary)] shadow-sm transition-colors hover:border-slate"
            >
              {theme === "dark" ? (
                <>
                  <span className="text-[13px]">☀️</span>
                  <span>White Mode</span>
                </>
              ) : (
                <>
                  <span className="text-[13px]">🌙</span>
                  <span>Dark Mode</span>
                </>
              )}
            </button>

            <div className={`flex items-center gap-2 font-medium ${apiLive ? "text-good" : "text-slate"}`}>
              <span className={`sf-live h-2 w-2 rounded-full ${apiLive ? "bg-good" : "bg-slate"}`} />
              <span className="font-mono text-[11px]">{apiLive ? "API" : "Demo"} · p99 {p99.toFixed(1)} ms</span>
            </div>

            <div className="hidden text-right font-mono text-[11px] text-slate md:block">
              <div>{clockTime(clock)}</div>
              <div>
                {active.id} · {selected.route === "FIREWALL_FIRST" ? "firewall" : "ML"}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex flex-1 flex-col">{children}</main>
    </div>
  );
}
