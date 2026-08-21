"use client";

import { useEffect, useState } from "react";
import { EgoGraph } from "@/components/ego-graph";
import { C_INFERENCE, C_SEARCH, PIPELINE } from "@/lib/demo";
import { holdLeft, inr, utcStamp } from "@/lib/format";
import { useConsole } from "@/lib/store";
import { postGovTicket } from "@/lib/api";
import {
  LatencyBars,
  PageHeader,
  Panel,
  RiskMeter,
  ShapList,
  StatusPill,
} from "@/components/ui";

export function AnalysisView({ id }: { id?: string }) {
  const { feed, selected, selectTx, clock, gov, hooks, unitCost, apiLive } = useConsole();
  const tx = (id && feed.find((t) => t.id === id)) || selected;

  const [ticketKind, setTicketKind] = useState<"IP_SUBNET" | "DEVICE" | "ACCOUNT">("ACCOUNT");
  const [ticketVal, setTicketVal] = useState("");
  const [ticketSrc, setTicketSrc] = useState<"I4C" | "NCRP">("I4C");
  const [submittingTicket, setSubmittingTicket] = useState(false);
  const [ticketMsg, setTicketMsg] = useState<string | null>(null);

  useEffect(() => {
    if (id) selectTx(id);
  }, [id, selectTx]);

  const p = tx.profile;
  const left = holdLeft(tx.holdUntil, clock);
  const missing = Boolean(id) && !feed.some((t) => t.id === id);

  async function handleAddTicket(e: React.FormEvent) {
    e.preventDefault();
    if (!ticketVal.trim()) return;
    setSubmittingTicket(true);
    setTicketMsg(null);
    try {
      const ticketId = `${ticketSrc}-${Math.floor(1000 + Math.random() * 9000)}`;
      await postGovTicket({
        id: ticketId,
        kind: ticketKind,
        value: ticketVal.trim(),
        src: ticketSrc,
      });
      setTicketMsg(`Added ${ticketId} to regulatory feed.`);
      setTicketVal("");
    } catch {
      setTicketMsg("Failed to add ticket to live API.");
    } finally {
      setSubmittingTicket(false);
    }
  }

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
      <PageHeader kicker="Incident Dossier" title={`Case File: ${tx.id}`}>
        <div className="flex items-center gap-3">
          <StatusPill status={tx.status} />
          <div className="flex items-center gap-2 rounded border border-[var(--line)] bg-[var(--bg-chip)] px-2.5 py-1 font-mono text-[12px]">
            <span className="text-slate">Target Account:</span>
            <span className="font-bold text-crimson">{tx.account}</span>
          </div>
        </div>
      </PageHeader>

      {/* Pipeline Navigation */}
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--bg-header)] px-3.5 py-2.5">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate">Pipeline Flow:</span>
        {PIPELINE.map((step, i) => (
          <span key={`pipeline-step-${step}-${i}`} className="flex items-center gap-2 text-[11px] text-slate">
            {i > 0 ? <span className="text-slate/40">→</span> : null}
            <span className="rounded-full border border-[var(--line)] bg-[var(--bg-card)] px-2.5 py-0.5 font-medium text-[var(--text-primary)]">
              {step}
            </span>
          </span>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.95fr_1.15fr_0.9fr]">
        {/* Left Column: Risk and TreeSHAP */}
        <div className="flex flex-col gap-4">
          <Panel title="Calibrated Risk Assessment" hint={`${tx.channel} · ${inr(tx.amount)}`}>
            <RiskMeter p={tx.pCalib} />
            <div className="mt-4 grid grid-cols-2 gap-2 text-[12px]">
              <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] px-2.5 py-2">
                <div className="sf-kicker">Raw GBDT Score</div>
                <div className="mt-0.5 font-mono font-medium text-[var(--text-primary)]">{tx.pRaw.toFixed(4)}</div>
              </div>
              <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] px-2.5 py-2">
                <div className="sf-kicker">Routing Stage</div>
                <div className="mt-0.5 font-mono font-medium text-[var(--text-primary)]">{tx.route.replaceAll("_", " ")}</div>
              </div>
              <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] px-2.5 py-2">
                <div className="sf-kicker">Gov Intelligence</div>
                <div className={`mt-0.5 font-mono ${tx.govHit ? "text-crimson font-bold" : "text-slate"}`}>
                  {tx.govHit ? "I4C/NCRP Blacklist" : "Clear (No Hits)"}
                </div>
              </div>
              <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] px-2.5 py-2">
                <div className="sf-kicker">TMS Ingest Flags</div>
                <div className="mt-0.5 font-mono font-medium text-[var(--text-primary)]">
                  {tx.tmsFlags.length ? tx.tmsFlags.join(" · ") : "None Fired"}
                </div>
              </div>
              {tx.status === "ESCROW" ? (
                <div className="col-span-2 rounded border border-crimson/30 bg-crimson/10 px-3 py-2 text-[12px]">
                  <div className="font-semibold text-crimson">15-Minute Autonomous Escrow Lock Active</div>
                  <div className="mt-0.5 font-mono font-medium text-[var(--text-primary)]">Time Remaining: {left ?? "Calculating…"}</div>
                </div>
              ) : null}
            </div>
          </Panel>

          <Panel title="Taylor-Scaled TreeSHAP Explanations" hint="Top Risk Drivers (Local Feature Attributions)">
            <ShapList tx={tx} />
          </Panel>
        </div>

        {/* Center Column: Ego-Graph */}
        <Panel title="Ego-Network Graph Decomposition" hint="1-Hop & 2-Hop Layering Topology · TTL Managed">
          <EgoGraph tx={tx} className="h-[320px]" />
          <div className="mt-4 overflow-auto">
            <div className="sf-kicker mb-1.5">Network Transaction Edges</div>
            <table className="w-full text-left text-[11px]">
              <thead className="text-[10px] uppercase tracking-[0.14em] text-slate">
                <tr>
                  <th className="py-1.5 font-medium">Origin</th>
                  <th className="font-medium">Destination</th>
                  <th className="font-medium">Channel</th>
                  <th className="text-right font-medium">Volume</th>
                  <th className="text-right font-medium">TTL</th>
                </tr>
              </thead>
              <tbody>
                {(tx.graph.edges ?? []).map((e, i) => (
                  <tr key={`edge-${e.from}-${e.to}-${i}`} className="border-t border-[var(--border-subtle)]">
                    <td className="py-1.5 font-mono font-medium text-[var(--text-primary)]">{e.from}</td>
                    <td className="font-mono font-medium text-[var(--text-primary)]">{e.to}</td>
                    <td className="text-slate">{e.channel}</td>
                    <td className="text-right font-mono font-medium text-[var(--text-primary)]">{inr(e.amount)}</td>
                    <td className="text-right">
                      <span className="rounded bg-[var(--bg-chip)] px-1.5 py-0.5 font-mono text-[10px] font-medium text-slate">
                        {e.ttl}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        {/* Right Column: Profile, Latency, Economics, Gov Ingest */}
        <div className="flex flex-col gap-4">
          <Panel title="Redis Historical Profile" hint="Account-Level L7 Rollup Metrics">
            <div className="grid grid-cols-3 gap-2 text-center">
              {[
                ["UPI Inbound L7", p.upiL7],
                ["ATM Cash L7", p.atmL7],
                ["Card Debit L7", p.cardL7],
                ["NetBank L7", p.netL7],
                ["V_cross Ratio", p.vCross],
                ["Txn Acceleration", p.accel],
              ].map(([k, v], idx) => (
                <div key={`profile-${k}-${idx}`} className="rounded border border-[var(--line)] bg-[var(--bg-card)] px-2 py-2">
                  <div className="sf-kicker text-[9px]">{k}</div>
                  <div className="mt-1 font-mono text-[15px] font-bold tabular-nums text-[var(--text-primary)]">{v}</div>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Inference Latency Waterfall" hint="Per-Component Execution Latency">
            <LatencyBars tx={tx} />
          </Panel>

          <Panel title="Routing Loss Economics" hint="Asymmetric Loss Function Assessment">
            <div className="grid grid-cols-3 gap-2 text-[12px]">
              <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] p-2 text-center">
                <div className="sf-kicker">C_search ($)</div>
                <div className="mt-1 font-mono font-semibold text-[var(--text-primary)]">{C_SEARCH.toFixed(2)}</div>
              </div>
              <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] p-2 text-center">
                <div className="sf-kicker">C_inference ($)</div>
                <div className="mt-1 font-mono font-semibold text-[var(--text-primary)]">{C_INFERENCE.toFixed(2)}</div>
              </div>
              <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] p-2 text-center">
                <div className="sf-kicker">This Decision</div>
                <div className="mt-1 font-mono font-bold text-crimson">{unitCost.toFixed(2)}</div>
              </div>
            </div>
          </Panel>
        </div>
      </div>

      {/* Bottom Section: Gov Feeds and Escrow Webhooks */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="I4C / NCRP Cyber Intelligence Feeds"
          hint="Central Regulatory Watchlist & Blacklist Feeds"
          action={
            <span className="rounded bg-[var(--navy)] px-2 py-0.5 text-[10px] font-semibold text-white dark:bg-foam dark:text-navy-deep">
              {gov.length} Active Feeds
            </span>
          }
        >
          {/* Add Ticket Form */}
          <form onSubmit={handleAddTicket} className="mb-3 flex flex-wrap items-center gap-2 border-b border-[var(--line)] pb-3">
            <select
              value={ticketSrc}
              onChange={(e) => setTicketSrc(e.target.value as "I4C" | "NCRP")}
              className="sf-select py-1 text-[11px]"
            >
              <option value="I4C">I4C</option>
              <option value="NCRP">NCRP</option>
            </select>
            <select
              value={ticketKind}
              onChange={(e) => setTicketKind(e.target.value as "IP_SUBNET" | "DEVICE" | "ACCOUNT")}
              className="sf-select py-1 text-[11px]"
            >
              <option value="ACCOUNT">ACCOUNT</option>
              <option value="IP_SUBNET">IP_SUBNET</option>
              <option value="DEVICE">DEVICE</option>
            </select>
            <input
              type="text"
              placeholder="e.g. XXXX2203 or 103.21.0.0/24"
              value={ticketVal}
              onChange={(e) => setTicketVal(e.target.value)}
              className="sf-input min-w-[140px] flex-1 py-1 text-[11px]"
            />
            <button
              type="submit"
              disabled={submittingTicket || !ticketVal.trim()}
              className="rounded bg-[var(--navy)] px-3 py-1 text-[11px] font-semibold text-white transition hover:opacity-90 disabled:opacity-40 dark:bg-foam dark:text-navy-deep"
            >
              {submittingTicket ? "Adding…" : "Ingest Flag"}
            </button>
          </form>
          {ticketMsg ? <p className="mb-2 text-[11px] font-medium text-good">{ticketMsg}</p> : null}

          <div className="max-h-[180px] space-y-1.5 overflow-auto">
            {gov.length === 0 ? (
              <p className="text-[12px] text-slate">No intelligence tickets active.</p>
            ) : (
              gov.slice(0, 10).map((g, idx) => (
                <div
                  key={`gov-${g.id}-${g.value}-${idx}`}
                  className="flex items-center justify-between gap-2 rounded border border-[var(--line)] bg-[var(--bg-card)] px-3 py-1.5 text-[11px]"
                >
                  <div>
                    <div className="font-mono font-medium text-[var(--text-primary)]">{g.value}</div>
                    <div className="text-slate">
                      <span className="font-semibold text-warn">{g.src}</span> · {g.kind.replaceAll("_", " ")} ({g.id})
                    </div>
                  </div>
                  <span className="font-mono text-[10px] text-slate">{utcStamp(g.ts)}</span>
                </div>
              ))
            )}
          </div>
        </Panel>

        <Panel
          title="Autonomous Escrow Outbound Webhooks"
          hint="15-Minute Automated Quarantine & External Payment Holds"
          action={
            <span className="rounded bg-crimson/15 px-2 py-0.5 text-[10px] font-bold text-crimson">
              {hooks.length} Events
            </span>
          }
        >
          <div className="max-h-[220px] space-y-1.5 overflow-auto font-mono text-[11px]">
            {hooks.length === 0 ? (
              <p className="py-6 text-center font-sans text-[12px] text-slate">
                No outbound webhook events recorded yet. Webhooks fire automatically when suspicious funds are placed in Escrow.
              </p>
            ) : (
              hooks
                .slice()
                .reverse()
                .map((h, idx) => (
                  <div
                    key={`hook-${h.id}-${h.txId}-${idx}`}
                    className="flex items-center justify-between rounded border border-crimson/20 bg-crimson/[0.04] px-3 py-2 text-slate"
                  >
                    <div>
                      <div className="font-semibold text-[var(--text-primary)]">{h.txId}</div>
                      <div className="text-[10px] text-slate">{h.endpoint}</div>
                    </div>
                    <div className="text-right">
                      <span className="rounded bg-crimson/20 px-1.5 py-0.5 text-[10px] font-bold text-crimson">
                        {h.holdMin}m Hold
                      </span>
                      <div className="mt-0.5 text-[10px] text-slate">{utcStamp(h.ts)}</div>
                    </div>
                  </div>
                ))
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
