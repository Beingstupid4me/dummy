"use client";

import { useEffect } from "react";
import { EgoGraph } from "@/components/ego-graph";
import { C_INFERENCE, C_SEARCH, PIPELINE } from "@/lib/demo";
import { holdLeft, inr, utcStamp } from "@/lib/format";
import { useConsole } from "@/lib/store";
import {
  LatencyBars,
  PageHeader,
  Panel,
  RiskMeter,
  ShapList,
  StatusPill,
} from "@/components/ui";

export function AnalysisView({ id }: { id?: string }) {
  const { feed, selected, selectTx, clock, gov, hooks, unitCost } = useConsole();
  const tx = (id && feed.find((t) => t.id === id)) || selected;

  useEffect(() => {
    if (id) selectTx(id);
  }, [id, selectTx]);

  const p = tx.profile;
  const left = holdLeft(tx.holdUntil, clock);
  const missing = Boolean(id) && !feed.some((t) => t.id === id);

  if (missing) {
    return (
      <div className="flex flex-1 flex-col gap-5 p-5">
        <PageHeader kicker="Case file" title={id ?? "Unknown"} />
        <p className="text-[13px] text-slate">
          This transaction is no longer in the rolling live window. Return to the dashboard
          and open a current row.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PageHeader kicker="Case file" title={tx.id}>
        <div className="flex items-center gap-3">
          <StatusPill status={tx.status} />
          <span className="font-mono text-[12px] text-slate">{tx.account}</span>
        </div>
      </PageHeader>

      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--line)] bg-[#0e1a2c] px-3 py-2">
        {PIPELINE.map((step, i) => (
          <span key={step} className="flex items-center gap-2 text-[11px] text-slate">
            {i > 0 ? <span className="text-slate/40">→</span> : null}
            <span className="rounded-full border border-[var(--line)] px-2 py-0.5 text-foam/80">
              {step}
            </span>
          </span>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.9fr_1.15fr_0.95fr]">
        <div className="flex flex-col gap-4">
          <Panel title="Calibrated risk" hint={`${tx.channel} · ${inr(tx.amount)}`}>
            <RiskMeter p={tx.pCalib} />
            <div className="mt-4 grid grid-cols-2 gap-2 text-[12px]">
              <div className="rounded-md border border-[var(--line)] px-2.5 py-2">
                Raw {tx.pRaw.toFixed(3)}
              </div>
              <div className="rounded-md border border-[var(--line)] px-2.5 py-2">
                {tx.route.replaceAll("_", " ")}
              </div>
              <div className="rounded-md border border-[var(--line)] px-2.5 py-2">
                {tx.govHit ? "I4C blacklist hit" : "No gov match"}
              </div>
              <div className="rounded-md border border-[var(--line)] px-2.5 py-2">
                TMS {tx.tmsFlags.length ? tx.tmsFlags.join(" · ") : "quiet"}
              </div>
              {tx.status === "ESCROW" ? (
                <div className="col-span-2 rounded-md border border-crimson/30 bg-crimson/5 px-2.5 py-2 font-mono text-[12px]">
                  Escrow remaining {left ?? "—"}
                </div>
              ) : null}
            </div>
          </Panel>
          <Panel title="Taylor-scaled TreeSHAP" hint="Local reason codes">
            <ShapList tx={tx} />
          </Panel>
        </div>

        <Panel title="Ego graph" hint="1-hop / 2-hop · Redis TTL 24h / 7d">
          <EgoGraph tx={tx} className="h-[320px]" />
          <div className="mt-3 overflow-auto">
            <table className="w-full text-left text-[11px]">
              <thead className="text-[10px] uppercase tracking-[0.14em] text-slate">
                <tr>
                  <th className="py-1.5 font-medium">From</th>
                  <th className="font-medium">To</th>
                  <th className="font-medium">Ch</th>
                  <th className="text-right font-medium">Amount</th>
                  <th className="text-right font-medium">TTL</th>
                </tr>
              </thead>
              <tbody>
                {tx.graph.edges.map((e, i) => (
                  <tr key={`${e.from}-${e.to}-${i}`} className="border-t border-white/[0.04]">
                    <td className="py-1 font-mono">{e.from}</td>
                    <td className="font-mono">{e.to}</td>
                    <td className="text-slate">{e.channel}</td>
                    <td className="text-right font-mono">{inr(e.amount)}</td>
                    <td className="text-right text-slate">{e.ttl}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="Redis profile" hint="Cross-channel L7">
            <div className="grid grid-cols-3 gap-2 text-center">
              {[
                ["UPI L7", p.upiL7],
                ["ATM L7", p.atmL7],
                ["Card L7", p.cardL7],
                ["Net L7", p.netL7],
                ["V_cross", p.vCross],
                ["Accel", p.accel],
              ].map(([k, v]) => (
                <div key={String(k)} className="rounded-md border border-[var(--line)] px-2 py-2">
                  <div className="sf-kicker">{k}</div>
                  <div className="mt-1 font-mono text-[16px] tabular-nums">{v}</div>
                </div>
              ))}
            </div>
          </Panel>
          <Panel title="Latency waterfall" hint="Must stay under 100 ms">
            <LatencyBars tx={tx} />
          </Panel>
          <Panel title="Routing economics" hint="C_search vs C_inference">
            <div className="grid grid-cols-3 gap-2 text-[12px]">
              <div>
                <div className="sf-kicker">C_search</div>
                <div className="mt-1 font-mono">{C_SEARCH.toFixed(1)}</div>
              </div>
              <div>
                <div className="sf-kicker">C_inference</div>
                <div className="mt-1 font-mono">{C_INFERENCE.toFixed(1)}</div>
              </div>
              <div>
                <div className="sf-kicker">This txn</div>
                <div className="mt-1 font-mono">{unitCost.toFixed(1)}</div>
              </div>
            </div>
          </Panel>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="I4C / NCRP tickets" hint="IP · device · account">
          <div className="max-h-[200px] space-y-1 overflow-auto">
            {gov.slice(0, 8).map((g) => (
              <div
                key={g.id}
                className="flex items-center justify-between gap-2 border-b border-white/[0.04] py-1.5 text-[11px]"
              >
                <div>
                  <div className="font-mono">{g.value}</div>
                  <div className="text-slate">
                    {g.src} · {g.kind.replaceAll("_", " ")}
                  </div>
                </div>
                <span className="font-mono text-slate">{utcStamp(g.ts)}</span>
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="Escrow webhooks" hint="15-minute outbound hold">
          <div className="max-h-[200px] space-y-1 overflow-auto font-mono text-[11px]">
            {hooks.length === 0 ? (
              <p className="font-sans text-[12px] text-slate">
                Webhooks fire when a transaction enters escrow.
              </p>
            ) : (
              hooks
                .slice()
                .reverse()
                .map((h) => (
                  <div key={h.id} className="flex justify-between gap-2 text-slate">
                    <span>{h.txId}</span>
                    <span>
                      {h.holdMin}m · {utcStamp(h.ts)}
                    </span>
                  </div>
                ))
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
