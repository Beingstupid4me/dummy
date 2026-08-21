"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Kpi, PageHeader, Panel, StatusPill, Tone } from "@/components/ui";
import { SAMPLE_ACCOUNTS } from "@/lib/demo";
import { inr, utcStamp } from "@/lib/format";
import { useConsole } from "@/lib/store";
import type { Transaction } from "@/lib/types";

const CHANNELS: Transaction["channel"][] = ["UPI", "ATM", "IMPS", "NEFT", "NETBANK"];

export function DashboardView() {
  const {
    feed,
    selected,
    queue,
    holds,
    p99,
    active,
    sim,
    simOut,
    scoring,
    apiLive,
    selectTx,
    setSim,
    runScore,
  } = useConsole();
  const router = useRouter();

  const [filterChannel, setFilterChannel] = useState<string>("ALL");
  const [filterStatus, setFilterStatus] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  function openCase(id: string) {
    selectTx(id);
    router.push(`/analysis/${id}`);
  }

  const filteredFeed = useMemo(() => {
    return feed.filter((t) => {
      if (filterChannel !== "ALL" && t.channel !== filterChannel) return false;
      if (filterStatus !== "ALL" && t.status !== filterStatus) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase().trim();
        const matchesAcct = t.account.toLowerCase().includes(q);
        const matchesId = t.id.toLowerCase().includes(q);
        const matchesBene = t.beneficiary.toLowerCase().includes(q);
        if (!matchesAcct && !matchesId && !matchesBene) return false;
      }
      return true;
    });
  }, [feed, filterChannel, filterStatus, searchQuery]);

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PageHeader kicker="SecOps Decision Console" title="Live Threat Dashboard">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-md border border-[var(--line)] bg-[var(--bg-header)] px-3 py-1.5 text-[12px]">
            <span className={`h-2 w-2 rounded-full ${apiLive ? "bg-good animate-pulse" : "bg-slate"}`} />
            <span className="font-semibold text-[var(--text-primary)]">{apiLive ? "Live API Connected" : "Demo Stream (Simulated)"}</span>
          </div>
          <p className="hidden max-w-xs text-right text-[12px] leading-4 text-slate md:block">
            Real-time inference & asymmetric economic risk routing
          </p>
        </div>
      </PageHeader>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi
          label="Chrono PR-AUC"
          value={active.prAuc ? active.prAuc.toFixed(3) : "—"}
          hint="Model score · Step 5 time windows"
        />
        <Kpi
          label="Mule F1 (Honest)"
          value={active.minorityF1 ? active.minorityF1.toFixed(3) : "—"}
          hint={`${active.id} (${active.tag}) · Out-of-fold`}
        />
        <Kpi
          label="Active Escrow Holds"
          value={String(holds)}
          hint="15-min autonomous fund freeze"
        />
        <Kpi
          label="p99 Scoring Latency"
          value={`${p99.toFixed(1)} ms`}
          hint="Production SLA < 100.0 ms"
        />
      </div>

      <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(340px,0.85fr)]">
        <Panel
          title="Core Transaction Stream"
          hint={apiLive ? "Real-time SSE event stream · Redis profile overlay" : "Mock feed fallback · Real-time simulation"}
          action={
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-medium text-slate">Total: {feed.length} txns</span>
            </div>
          }
        >
          {/* Controls Bar */}
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--line)] pb-3">
            <div className="flex flex-1 items-center gap-2">
              <input
                type="text"
                placeholder="Search account, TX-ID, beneficiary..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="sf-input min-w-[180px] flex-1 py-1 text-[12px]"
              />
            </div>
            <div className="flex items-center gap-2">
              <select
                value={filterChannel}
                onChange={(e) => setFilterChannel(e.target.value)}
                className="sf-select py-1 text-[11px]"
              >
                <option value="ALL">All Channels</option>
                {CHANNELS.map((c, idx) => (
                  <option key={`filter-channel-${c}-${idx}`} value={c}>
                    {c}
                  </option>
                ))}
              </select>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="sf-select py-1 text-[11px]"
              >
                <option value="ALL">All Status</option>
                <option value="CLEAR">CLEAR</option>
                <option value="QUEUE">QUEUE (Review)</option>
                <option value="ESCROW">ESCROW (Freeze)</option>
              </select>
            </div>
          </div>

          <div className="max-h-[min(620px,calc(100vh-320px))] overflow-auto">
            <table className="w-full text-left text-[12px]">
              <thead className="sticky top-0 bg-[var(--bg-header)] text-[10px] uppercase tracking-[0.14em] text-slate">
                <tr>
                  <th className="py-2.5 font-medium">Time (UTC)</th>
                  <th className="font-medium">Tx ID</th>
                  <th className="font-medium">Account</th>
                  <th className="font-medium">Channel</th>
                  <th className="text-right font-medium">Amount</th>
                  <th className="text-right font-medium">Risk P</th>
                  <th className="font-medium">Route</th>
                  <th className="font-medium">Decision</th>
                </tr>
              </thead>
              <tbody>
                {filteredFeed.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-slate">
                      No transactions match the current filter.
                    </td>
                  </tr>
                ) : (
                  [...filteredFeed].reverse().map((t, idx) => (
                    <tr
                      key={`feed-${t.id}-${idx}`}
                      onClick={() => openCase(t.id)}
                      className={`cursor-pointer border-t border-[var(--border-subtle)] transition-colors ${
                        t.id === selected.id ? "bg-crimson/[0.12]" : "hover:bg-slate-500/10"
                      }`}
                    >
                      <td className="py-2 font-mono text-[11px] text-slate">{utcStamp(t.ts)}</td>
                      <td className="font-mono text-[11px] text-slate">{t.id}</td>
                      <td className="font-mono font-semibold text-[var(--text-primary)]">{t.account}</td>
                      <td>
                        <span className="rounded bg-[var(--bg-chip)] px-1.5 py-0.5 font-mono text-[11px] font-medium text-slate">
                          {t.channel}
                        </span>
                      </td>
                      <td className="text-right font-mono font-medium text-[var(--text-primary)]">{inr(t.amount)}</td>
                      <td className="text-right font-mono font-bold">
                        <Tone tone={t.pCalib >= 0.62 ? "bad" : t.pCalib >= 0.32 ? "warn" : "good"}>
                          {(t.pCalib * 100).toFixed(1)}%
                        </Tone>
                      </td>
                      <td className="text-[11px]">
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                            t.route === "FIREWALL_FIRST"
                              ? "bg-warn/20 text-warn"
                              : "bg-slate/15 text-slate"
                          }`}
                        >
                          {t.route === "FIREWALL_FIRST" ? "FIREWALL" : "ML_FLOW"}
                        </span>
                      </td>
                      <td>
                        <StatusPill status={t.status} />
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel
            title="Focused Incident Case"
            hint={`Transaction ${selected.id}`}
            action={<StatusPill status={selected.status} />}
          >
            <div className="space-y-3.5 text-[13px]">
              <div className="flex items-baseline justify-between border-b border-[var(--line)] pb-3">
                <div>
                  <div className="sf-kicker">Target Account</div>
                  <span className="font-mono text-[16px] font-bold text-[var(--text-primary)]">{selected.account}</span>
                </div>
                <div className="text-right">
                  <div className="sf-kicker">Calibrated Risk</div>
                  <span className="font-mono text-[24px] font-bold tabular-nums text-crimson">
                    {(selected.pCalib * 100).toFixed(1)}
                    <span className="ml-1 text-[12px] font-normal text-slate">%</span>
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[12px]">
                <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] p-2">
                  <div className="sf-kicker">Channel & Volume</div>
                  <div className="mt-1 font-mono font-medium text-[var(--text-primary)]">
                    {selected.channel} · {inr(selected.amount)}
                  </div>
                </div>
                <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] p-2">
                  <div className="sf-kicker">Routing Pipeline</div>
                  <div className="mt-1 font-mono font-medium text-[var(--text-primary)]">
                    {selected.route === "FIREWALL_FIRST" ? "Firewall First" : "ML Stream"}
                  </div>
                </div>
                <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] p-2">
                  <div className="sf-kicker">Regulatory Blacklist</div>
                  <div className="mt-1 font-mono">
                    <span className={selected.govHit ? "text-crimson font-bold" : "text-slate"}>
                      {selected.govHit ? "I4C/NCRP Match" : "Clear (No match)"}
                    </span>
                  </div>
                </div>
                <div className="rounded border border-[var(--line)] bg-[var(--bg-card)] p-2">
                  <div className="sf-kicker">Latency SLA</div>
                  <div className="mt-1 font-mono font-medium text-[var(--text-primary)]">{selected.latencyMs.toFixed(1)} ms</div>
                </div>
              </div>

              {selected.tmsFlags.length > 0 ? (
                <div className="rounded border border-warn/30 bg-warn/10 p-2">
                  <div className="sf-kicker text-warn font-semibold">TMS Triggered Rules</div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    {selected.tmsFlags.map((flag, idx) => (
                      <span key={`tms-${flag}-${idx}`} className="rounded bg-warn/20 px-1.5 py-0.5 text-[10px] font-semibold text-warn">
                        {flag}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}

              <Link
                href={`/analysis/${selected.id}`}
                onClick={() => selectTx(selected.id)}
                className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-[var(--navy)] py-2 text-[13px] font-semibold text-white no-underline shadow-sm transition hover:opacity-90 dark:bg-foam dark:text-navy-deep"
              >
                <span>Inspect Full Case Dossier</span>
                <span>→</span>
              </Link>
            </div>
          </Panel>

          <Panel
            title="HITL Alert Priority Queue"
            hint="Human-in-the-loop review triage"
            action={
              <span className="rounded bg-crimson/15 px-2 py-0.5 text-[10px] font-bold text-crimson">
                {queue.length} Pending
              </span>
            }
          >
            <div className="max-h-[200px] space-y-1.5 overflow-auto">
              {queue.length === 0 ? (
                <p className="py-4 text-center text-[12px] text-slate">Review queue is currently clear.</p>
              ) : (
                queue.map((t, idx) => (
                  <Link
                    key={`queue-${t.id}-${idx}`}
                    href={`/analysis/${t.id}`}
                    onClick={() => selectTx(t.id)}
                    className="flex items-center justify-between rounded-md border border-[var(--line)] bg-[var(--bg-card)] px-3 py-2 text-[12px] no-underline transition hover:border-slate hover:bg-slate-500/10"
                  >
                    <div>
                      <div className="font-mono font-semibold text-[var(--text-primary)]">{t.account}</div>
                      <div className="text-[11px] text-slate">
                        {t.channel} · {inr(t.amount)}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <StatusPill status={t.status} />
                      <span className="font-mono text-[13px] font-bold text-crimson">
                        {(t.pCalib * 100).toFixed(0)}%
                      </span>
                    </div>
                  </Link>
                ))
              )}
            </div>
          </Panel>

          <Panel title="Interactive Simulation Sandbox" hint="POST /score test harness">
            <div className="space-y-2.5">
              <div>
                <label className="sf-kicker mb-1 block">Select Benchmark Account</label>
                <select
                  className="sf-select"
                  value={sim.account}
                  onChange={(e) => setSim({ ...sim, account: e.target.value })}
                >
                  {SAMPLE_ACCOUNTS.map((a, idx) => (
                    <option key={`sample-${a.account}-${idx}`} value={a.account}>
                      {a.account} — {a.note}
                    </option>
                  ))}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="sf-kicker mb-1 block">Channel</label>
                  <select
                    className="sf-select"
                    value={sim.channel}
                    onChange={(e) =>
                      setSim({ ...sim, channel: e.target.value as Transaction["channel"] })
                    }
                  >
                    {CHANNELS.map((c, idx) => (
                      <option key={`sim-channel-${c}-${idx}`} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="sf-kicker mb-1 block">Amount (₹)</label>
                  <input
                    className="sf-input font-mono"
                    type="number"
                    min={100}
                    step={1000}
                    value={sim.amount}
                    onChange={(e) => setSim({ ...sim, amount: Number(e.target.value) })}
                  />
                </div>
              </div>
              <button
                onClick={runScore}
                disabled={scoring}
                className="w-full rounded-md bg-[var(--navy)] py-2 text-[13px] font-semibold text-white shadow-sm transition hover:opacity-90 disabled:opacity-50"
              >
                {scoring ? "Scoring via Pipeline…" : "Execute Transaction Risk Assessment"}
              </button>
              {simOut ? (
                <div className="mt-2 rounded border border-[var(--line)] bg-[var(--bg-card)] p-2.5 text-center">
                  <div className="font-mono text-[12px] font-bold text-[var(--text-primary)]">
                    {simOut.id} · P={(simOut.pCalib * 100).toFixed(1)}% ({simOut.status})
                  </div>
                  <Link
                    href={`/analysis/${simOut.id}`}
                    className="mt-1 block text-[11px] font-medium text-slate hover:text-[var(--text-primary)]"
                  >
                    View SHAP & Ego Graph Decomposition →
                  </Link>
                </div>
              ) : null}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
