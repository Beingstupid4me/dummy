"use client";

import { PageHeader, Panel, Switch } from "@/components/ui";
import { useConsole } from "@/lib/store";
import type { ModelId } from "@/lib/types";

export function RegistryView() {
  const {
    models,
    model,
    toggles,
    retrain,
    log,
    smote,
    p99,
    swapModel,
    toggleFeature,
    setSmote,
    startRetrain,
  } = useConsole();

  const safe = toggles.filter((t) => !t.leaky);
  const leaky = toggles.filter((t) => t.leaky);
  const m4empty = models.find((m) => m.id === "M4")?.tag === "EMPTY";
  const activeModelObj = models.find((m) => m.id === model);
  const notes = activeModelObj?.notes;

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PageHeader kicker="Orchestrator & Lifecycle" title="Model Registry & Zero-Downtime Hot-Swap">
        <p className="max-w-md text-right text-[12px] leading-5 text-slate">
          Zero-downtime pointer swapping across 4 memory slots (&lt;5 MB). Trigger on-demand retrains with full telemetry.
        </p>
      </PageHeader>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {[
          ["Inference SLA", p99 < 100 ? "Pass" : "Fail", `${p99.toFixed(1)} ms / 100 ms target`],
          ["Hot-Swap Cutover", "Pass", "Zero-downtime pointer update"],
          [
            "Retrain Monotonicity",
            retrain === "done" ? "Pass" : "Verified",
            "PCHIP f′(x) > 0 · 90-day buffer",
          ],
        ].map(([k, v, s], idx) => (
          <div
            key={`status-kpi-${k}-${idx}`}
            className="sf-panel flex items-center justify-between rounded-lg px-4 py-3"
          >
            <div>
              <div className="sf-kicker">{k}</div>
              <div className="mt-1 text-[12px] text-slate">{s}</div>
            </div>
            <span
              className={`rounded px-2.5 py-1 text-[12px] font-semibold ${
                v === "Pass"
                  ? "bg-good/15 text-good"
                  : v === "Fail"
                    ? "bg-crimson/15 text-crimson"
                    : "bg-warn/15 text-warn"
              }`}
            >
              {v}
            </span>
          </div>
        ))}
      </div>

      <Panel
        title="Production Model Slots"
        hint="Click any loaded slot to instantly swap active scoring model with zero downtime"
        action={
          <span className="font-mono text-[11px] text-slate">
            Active: <span className="font-bold text-good">{model}</span>
          </span>
        }
      >
        <div className="overflow-auto">
          <table className="w-full text-left text-[12px]">
            <thead className="text-[10px] uppercase tracking-[0.14em] text-slate">
              <tr>
                <th className="py-2.5 font-medium">Slot</th>
                <th className="font-medium">Model Designation</th>
                <th className="text-right font-medium">PR-AUC</th>
                <th className="text-right font-medium">ROC-AUC</th>
                <th className="text-right font-medium">Mule F1</th>
                <th className="text-right font-medium">Macro F1</th>
                <th className="text-right font-medium">p99 Latency</th>
                <th className="text-right font-medium">Footprint</th>
                <th className="text-right font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, idx) => {
                const empty = m.id === "M4" && m.tag === "EMPTY";
                const isActive = model === m.id;
                return (
                  <tr
                    key={`model-slot-${m.id}-${idx}`}
                    onClick={() => swapModel(m.id as ModelId)}
                    className={`border-t border-[var(--border-subtle)] transition-colors ${
                      empty ? "cursor-not-allowed opacity-40" : "cursor-pointer hover:bg-slate-500/10"
                    } ${isActive ? "bg-[var(--navy)]/10 font-medium" : ""}`}
                  >
                    <td className="py-3 font-mono font-bold text-[var(--text-primary)]">
                      {m.id}
                    </td>
                    <td>
                      <div className="font-medium text-[var(--text-primary)]">{m.name}</div>
                      <div className="text-[10px] font-semibold uppercase tracking-widest text-slate">{m.tag}</div>
                    </td>
                    <td className="text-right font-mono font-semibold text-[var(--text-primary)]">{m.prAuc ? m.prAuc.toFixed(3) : "—"}</td>
                    <td className="text-right font-mono text-slate">{m.rocAuc ? m.rocAuc.toFixed(3) : "—"}</td>
                    <td className="text-right font-mono font-semibold text-crimson">
                      {m.minorityF1 ? m.minorityF1.toFixed(3) : "—"}
                    </td>
                    <td className="text-right font-mono text-slate">
                      {m.macroF1 ? m.macroF1.toFixed(3) : "—"}
                    </td>
                    <td className="text-right font-mono text-slate">{m.p99Ms ? `${m.p99Ms.toFixed(1)} ms` : "—"}</td>
                    <td className="text-right font-mono text-slate">{m.sizeMb ? `${m.sizeMb.toFixed(2)} MB` : "—"}</td>
                    <td className="text-right">
                      {isActive ? (
                        <span className="rounded-full bg-good/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-good">
                          ACTIVE
                        </span>
                      ) : empty ? (
                        <span className="rounded-full bg-slate/15 px-2 py-0.5 text-[10px] uppercase text-slate">
                          EMPTY
                        </span>
                      ) : (
                        <span className="rounded-full bg-slate-500/15 px-2 py-0.5 text-[10px] uppercase text-slate hover:bg-slate-500/25">
                          STANDBY
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {notes ? (
          <div className="mt-3.5 rounded border border-[var(--line)] bg-[var(--bg-card)] p-3 text-[12px] leading-5 text-slate">
            <span className="font-semibold text-[var(--text-primary)]">Active Slot Profile ({model}): </span>
            {notes}
          </div>
        ) : null}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <Panel
          title="Feature Composition Board"
          hint="Control feature sets fed into the on-demand retrain worker"
        >
          <div className="grid gap-8 sm:grid-cols-2">
            <div>
              <div className="sf-kicker mb-3 font-semibold text-[var(--text-primary)]">Production Feature Tracks</div>
              <div className="space-y-2.5">
                {safe.map((f, idx) => (
                  <label key={`safe-feature-${f.id}-${idx}`} className="flex items-center justify-between gap-3 text-[12px]">
                    <span className="text-[var(--text-primary)]">{f.label}</span>
                    <Switch on={f.on} onChange={() => toggleFeature(f.id)} />
                  </label>
                ))}
              </div>
            </div>
            <div>
              <div className="sf-kicker mb-3 font-semibold text-crimson">Audited Leaky Tracks (Demo Only)</div>
              <div className="space-y-2.5">
                {leaky.map((f, idx) => (
                  <label key={`leaky-feature-${f.id}-${idx}`} className="flex items-center justify-between gap-3 text-[12px]">
                    <span className="text-slate">{f.label}</span>
                    <Switch on={f.on} leaky onChange={() => toggleFeature(f.id)} />
                  </label>
                ))}
              </div>
              <p className="mt-4 rounded border border-crimson/25 bg-crimson/[0.04] p-2 text-[11px] leading-4 text-slate">
                <span className="font-semibold text-crimson">Strict Zero-Leakage Policy:</span> F3912, F2230, and post-alert status variables are confirmed target leaks. Keep these OFF for verified production deployment.
              </p>
            </div>
          </div>
        </Panel>

        <Panel
          title="Autonomous Retrain Pipeline"
          hint="Trigger on-demand GBDT retraining with PCHIP monotonicity validation"
        >
          <div className="space-y-3">
            <label className="block">
              <div className="flex justify-between text-[12px]">
                <span className="sf-kicker font-medium">SMOTE Minority Oversampling Ratio</span>
                <span className="font-mono text-[11px] font-bold text-[var(--text-primary)]">
                  {smote === 0 ? "Disabled (scale_pos_weight)" : smote.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={0.3}
                step={0.05}
                value={smote}
                onChange={(e) => setSmote(Number(e.target.value))}
                className="mt-2 w-full accent-[#457b9d]"
              />
            </label>
            <p className="text-[11px] leading-4 text-slate">
              Enforces 90-day label delay buffer to prevent label noise from chargeback lag. Pipeline validates monotone spline derivative \(f′(x) &gt; 0\) before committing weights.
            </p>
            <button
              onClick={startRetrain}
              disabled={retrain === "run"}
              className="w-full rounded-md bg-crimson py-2.5 text-[13px] font-semibold text-white shadow-sm transition hover:bg-[#c4313d] disabled:opacity-60"
            >
              {retrain === "run" ? "Executing Worker Retrain & PCHIP Spline…" : m4empty ? "Trigger Automated Retrain → Slot M4" : "Re-execute Retrain Pipeline → Slot M4"}
            </button>
            <div className="sf-kicker mt-3">Live Worker Progress Stream</div>
            <div className="h-40 overflow-auto rounded border border-[var(--line)] bg-[var(--bg-card)] p-2.5 font-mono text-[11px] leading-5 text-slate">
              {log.length === 0 ? (
                <span className="font-sans text-slate/70">Ready. Retrain latency SLA is &lt;15s. Progress events stream via SSE.</span>
              ) : (
                log.map((l, i) => <div key={`retrain-log-${i}-${l.slice(0, 12)}`} className="text-[var(--text-primary)]">› {l}</div>)
              )}
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
