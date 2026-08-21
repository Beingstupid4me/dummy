import type { AlertStatus, Transaction } from "@/lib/types";

export function Panel({
  title,
  hint,
  action,
  children,
  className = "",
}: {
  title?: string;
  hint?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`sf-panel rounded-lg ${className}`}>
      {title ? (
        <header className="flex items-center justify-between gap-3 border-b border-[var(--line)] px-4 py-2.5">
          <div>
            <h2 className="text-[13px] font-semibold tracking-tight text-[var(--text-primary)]">{title}</h2>
            {hint ? <p className="mt-0.5 text-[11px] leading-4 text-slate">{hint}</p> : null}
          </div>
          {action}
        </header>
      ) : null}
      <div className={title ? "p-4" : "p-4"}>{children}</div>
    </section>
  );
}

export function Kpi({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="sf-panel rounded-lg px-3.5 py-2.5">
      <div className="sf-kicker">{label}</div>
      <div className="mt-1 font-mono text-[22px] font-bold leading-none tabular-nums tracking-tight text-[var(--text-primary)]">
        {value}
      </div>
      <div className="mt-1.5 text-[11px] text-slate">{hint}</div>
    </div>
  );
}

export function StatusPill({ status }: { status: AlertStatus }) {
  const map = {
    ESCROW: "bg-crimson/15 text-crimson border border-crimson/30",
    QUEUE: "bg-warn/15 text-warn border border-warn/30",
    CLEAR: "bg-good/15 text-good border border-good/30",
  } as const;
  const label = status === "ESCROW" ? "15-min escrow" : status;
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] ${map[status]}`}
    >
      {label}
    </span>
  );
}

export function Tone({
  tone,
  children,
}: {
  tone: "good" | "warn" | "bad" | "mute";
  children: React.ReactNode;
}) {
  const c = {
    good: "text-good font-semibold",
    warn: "text-warn font-semibold",
    bad: "text-crimson font-bold",
    mute: "text-slate",
  }[tone];
  return <span className={c}>{children}</span>;
}

export function RiskMeter({ p }: { p: number }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const arc = c * 0.75;
  const filled = Math.max(0.015, Math.min(1, p)) * arc;
  const color = p >= 0.62 ? "#dc2626" : p >= 0.32 ? "#d97706" : "#059669";
  return (
    <div className="relative mx-auto h-[132px] w-[132px]">
      <svg viewBox="0 0 140 140" className="h-full w-full -rotate-[135deg]">
        <circle
          cx="70"
          cy="70"
          r={r}
          fill="none"
          stroke="currentColor"
          className="text-slate/20"
          strokeWidth="8"
          strokeDasharray={`${arc} ${c}`}
          strokeLinecap="round"
        />
        <circle
          cx="70"
          cy="70"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={`${filled} ${c}`}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="font-mono text-[28px] font-bold leading-none tabular-nums text-[var(--text-primary)]">
          {(p * 100).toFixed(1)}
        </div>
        <div className="mt-1 text-[10px] font-medium uppercase tracking-[0.18em] text-slate">
          calibrated
        </div>
      </div>
    </div>
  );
}

export function Switch({
  on,
  leaky,
  disabled,
  onChange,
}: {
  on: boolean;
  leaky?: boolean;
  disabled?: boolean;
  onChange: () => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={disabled}
      onClick={onChange}
      className={`relative h-[18px] w-[32px] shrink-0 rounded-full transition-colors ${
        on ? (leaky ? "bg-crimson" : "bg-good") : "bg-slate/30"
      } ${disabled ? "opacity-40" : ""}`}
    >
      <span
        className={`absolute top-[2px] left-[2px] h-[14px] w-[14px] rounded-full bg-white shadow-sm transition-transform ${
          on ? "translate-x-[14px]" : ""
        }`}
      />
    </button>
  );
}

export function ShapList({ tx, limit = 6 }: { tx: Transaction; limit?: number }) {
  const shapItems = tx.shap ?? [];
  return (
    <div className="space-y-2.5">
      {shapItems.slice(0, limit).map((s, idx) => {
        const mag = Math.min(48, Math.abs(s.value) * 220);
        return (
          <div key={`${s.feature}-${s.label}-${idx}`} className="grid grid-cols-[1fr_56px] items-center gap-3">
            <div>
              <div className="relative h-1.5 overflow-hidden rounded-full bg-[var(--bg-chip)]">
                <div
                  className={`absolute top-0 h-full ${s.value >= 0 ? "bg-crimson" : "bg-slate"}`}
                  style={{
                    width: `${mag}%`,
                    left: s.value >= 0 ? "50%" : undefined,
                    right: s.value < 0 ? "50%" : undefined,
                  }}
                />
              </div>
              <div className="mt-1 truncate text-[11px] font-medium text-slate">{s.label}</div>
            </div>
            <div className="font-mono text-right text-[11px] font-semibold tabular-nums text-[var(--text-primary)]">
              {s.value >= 0 ? "+" : ""}
              {s.value.toFixed(3)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function LatencyBars({ tx }: { tx: Transaction }) {
  const rows: [string, number][] = [
    ["Redis", tx.latency?.redis ?? 0],
    ["Reconstruct", tx.latency?.reconstruct ?? 0],
    ["GBDT", tx.latency?.gbdt ?? 0],
    ["PCHIP", tx.latency?.pchip ?? 0],
    ["TreeSHAP", tx.latency?.shap ?? 0],
  ];
  const max = Math.max(...rows.map(([, v]) => v), 1);
  return (
    <div className="space-y-2">
      {rows.map(([k, v], idx) => (
        <div key={`${k}-${idx}`} className="grid grid-cols-[92px_1fr_52px] items-center gap-2 text-[11px]">
          <span className="text-slate">{k}</span>
          <div className="h-1.5 overflow-hidden rounded-full bg-[var(--bg-chip)]">
            <div className="h-full rounded-full bg-[var(--slate)]" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <span className="text-right font-mono font-medium tabular-nums text-[var(--text-primary)]">{v.toFixed(1)} ms</span>
        </div>
      ))}
      <div className="flex justify-between border-t border-[var(--line)] pt-2 text-[11px]">
        <span className="font-medium text-slate">Total</span>
        <span className="font-mono font-bold tabular-nums text-[var(--text-primary)]">
          {(tx.latencyMs ?? 0).toFixed(1)} ms
          <span className="ml-2 font-normal text-slate">SLA &lt;100 ms</span>
        </span>
      </div>
    </div>
  );
}

export function PageHeader({
  kicker,
  title,
  children,
}: {
  kicker: string;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="sf-kicker">{kicker}</div>
        <h1 className="mt-1 text-[22px] font-bold tracking-tight text-[var(--text-primary)]">{title}</h1>
      </div>
      {children}
    </div>
  );
}
