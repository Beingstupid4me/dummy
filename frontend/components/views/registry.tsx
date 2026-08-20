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
  const notes = models.find((m) => m.id === model)?.notes;

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PageHeader kicker="Orchestrator" title="Model registry">
        <p className="max-w-md text-right text-[12px] leading-5 text-slate">
          Four slots under 5 MB. Click a row to hot-swap. Retrain writes M4.
        </p>
      </PageHeader>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {[
          ["API latency SLA", p99 < 100 ? "Pass" : "Fail", `${p99.toFixed(1)} ms / 100 ms`],
          ["Registry hot-swap", "Pass", "M1 ↔ M3 · 0 ms downtime"],
          [
            "Retrain integrity",
            retrain === "done" ? "Pass" : "Pending",
            "PCHIP f′(x)>0 · < 15 s · M4",
          ],
        ].map(([k, v, s]) => (
          <div
            key={k}
            className="sf-panel flex items-center justify-between rounded-lg px-3.5 py-2.5"
          >
            <div>
              <div className="sf-kicker">{k}</div>
              <div className="mt-1 text-[12px] text-slate">{s}</div>
            </div>
            <span
              className={`text-[12px] font-medium ${
                v === "Pass" ? "text-good" : v === "Fail" ? "text-crimson" : "text-warn"
              }`}
            >
              {v}
            </span>
          </div>
        ))}
      </div>

      <Panel title="Registry" hint="Filesystem slots · click to activate">
        <div className="overflow-auto">
          <table className="w-full text-left text-[12px]">
            <thead className="text-[10px] uppercase tracking-[0.14em] text-slate">
              <tr>
                <th className="py-2 font-medium">Slot</th>
                <th className="font-medium">Name</th>
                <th className="text-right font-medium">PR-AUC</th>
                <th className="text-right font-medium">ROC</th>
                <th className="text-right font-medium">Mule F1</th>
                <th className="text-right font-medium">Cost</th>
                <th className="text-right font-medium">p99</th>
                <th className="text-right font-medium">MB</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const empty = m.id === "M4" && m.tag === "EMPTY";
                return (
                  <tr
                    key={m.id}
                    onClick={() => swapModel(m.id as ModelId)}
                    className={`border-t border-white/[0.04] ${
                      empty ? "cursor-not-allowed opacity-45" : "cursor-pointer hover:bg-white/[0.03]"
                    } ${model === m.id ? "bg-foam/[0.06]" : ""}`}
                  >
                    <td className="py-2.5 font-mono">
                      {m.id}
                      {model === m.id ? (
                        <span className="ml-2 text-[10px] uppercase tracking-widest text-good">
                          active
                        </span>
                      ) : null}
                    </td>
                    <td>
                      <div>{m.name}</div>
                      <div className="text-[10px] uppercase tracking-widest text-slate">{m.tag}</div>
                    </td>
                    <td className="text-right font-mono">{m.prAuc ? m.prAuc.toFixed(3) : "—"}</td>
                    <td className="text-right font-mono">{m.rocAuc ? m.rocAuc.toFixed(3) : "—"}</td>
                    <td className="text-right font-mono">
                      {m.minorityF1 ? m.minorityF1.toFixed(3) : "—"}
                    </td>
                    <td className="text-right font-mono">{m.cost || "—"}</td>
                    <td className="text-right font-mono">{m.p99Ms ? m.p99Ms.toFixed(1) : "—"}</td>
                    <td className="text-right font-mono">{m.sizeMb || "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {notes ? <p className="mt-3 text-[12px] leading-5 text-slate">{notes}</p> : null}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
        <Panel title="Feature control board" hint="Sent to POST /retrain">
          <div className="grid gap-8 sm:grid-cols-2">
            <div>
              <div className="sf-kicker mb-3">Production tracks</div>
              <div className="space-y-2.5">
                {safe.map((f) => (
                  <label key={f.id} className="flex items-center justify-between gap-3 text-[12px]">
                    <span>{f.label}</span>
                    <Switch on={f.on} onChange={() => toggleFeature(f.id)} />
                  </label>
                ))}
              </div>
            </div>
            <div>
              <div className="sf-kicker mb-3 text-crimson">Leaky · demo only</div>
              <div className="space-y-2.5">
                {leaky.map((f) => (
                  <label key={f.id} className="flex items-center justify-between gap-3 text-[12px]">
                    <span>{f.label}</span>
                    <Switch on={f.on} leaky onChange={() => toggleFeature(f.id)} />
                  </label>
                ))}
              </div>
              <p className="mt-3 text-[11px] leading-5 text-slate">
                F3912 and F2230 are confirmed leaks. Leave the six audited tags off for a
                deployable M4.
              </p>
            </div>
          </div>
        </Panel>

        <Panel title="On-demand retrain" hint="POST /retrain · FastAPI 202">
          <label className="block">
            <div className="flex justify-between text-[12px]">
              <span className="sf-kicker">SMOTE ratio</span>
              <span className="font-mono text-[11px] text-slate">
                {smote === 0 ? "off" : smote.toFixed(2)}
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
          <p className="mt-2 text-[11px] leading-5 text-slate">
            Locked Phase 1 used no SMOTE. 90-day label-delay and PCHIP monotonicity stay on.
            Ensemble: XGB+LGB 500 / depth 5 / lr 0.05.
          </p>
          <button
            onClick={startRetrain}
            disabled={retrain === "run"}
            className="mt-4 w-full rounded-md bg-crimson py-2 text-[13px] font-medium text-foam hover:bg-[#c4313d] disabled:opacity-60"
          >
            {retrain === "run" ? "Retraining…" : m4empty ? "Trigger /retrain → M4" : "Retrain M4 again"}
          </button>
          <div className="mt-3 h-36 overflow-auto font-mono text-[11px] leading-5 text-slate">
            {log.length === 0 ? (
              <span className="font-sans">Log appears here. Typical wall time ~4 s, SLA 15 s.</span>
            ) : (
              log.map((l) => <div key={l}>› {l}</div>)
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
