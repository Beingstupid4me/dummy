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
  const { toast, p99, clock, active, selected, apiLive } = useConsole();

  return (
    <div className="sf-shell">
      {toast ? (
        <div className="sf-toast pointer-events-none fixed top-4 left-1/2 z-20 rounded-md border border-[var(--line)] bg-[#12233a] px-3 py-1.5 text-[12px] shadow-lg">
          {toast}
        </div>
      ) : null}

      <header className="border-b border-[var(--line)] px-5 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Link href="/" className="min-w-0 no-underline">
            <div className="sf-kicker">Bank of India · Cybershield 2026</div>
            <div className="mt-0.5 flex items-baseline gap-3">
              <span className="text-[18px] font-semibold tracking-tight text-foam">
                SentinelFlow
              </span>
              <span className="hidden text-[12px] text-slate sm:inline">
                mule-account prevention
              </span>
            </div>
          </Link>

          <nav className="flex rounded-md border border-[var(--line)] p-0.5">
            {NAV.map((item) => {
              const on =
                item.href === "/"
                  ? pathname === "/"
                  : pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={item.href}
                  href={item.href === "/analysis" ? `/analysis/${selected.id}` : item.href}
                  className={`rounded px-3 py-1.5 text-[12px] no-underline ${
                    on ? "bg-foam text-navy-deep" : "text-slate hover:text-foam"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-4 text-[12px]">
            <div className={`flex items-center gap-2 ${apiLive ? "text-good" : "text-slate"}`}>
              <span className={`sf-live h-1.5 w-1.5 rounded-full ${apiLive ? "bg-good" : "bg-slate"}`} />
              {apiLive ? "API · p99" : "Demo · p99"} {p99.toFixed(1)} ms
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
