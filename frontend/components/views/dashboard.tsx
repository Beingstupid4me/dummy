"use client";

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

  function openCase(id: string) {
    selectTx(id);
    router.push(`/analysis/${id}`);
  }

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PageHeader kicker="SecOps" title="Dashboard">
        <p className="max-w-md text-right text-[12px] leading-5 text-slate">
          Live core stream and HITL queue. Open a row for the case file.
        </p>
      </PageHeader>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi
          label="Chrono PR-AUC"
          value={active.prAuc ? active.prAuc.toFixed(3) : "—"}
          hint="vs report 0.710 · same windows"
        />
        <Kpi
          label="Mule F1"
          value={active.minorityF1 ? active.minorityF1.toFixed(3) : "—"}
          hint={`${active.id} · nested threshold`}
        />
        <Kpi label="Escrow holds" value={String(holds)} hint="15-min outbound webhook" />
        <Kpi label="p99 latency" value={`${p99.toFixed(1)} ms`} hint="SLA < 100 ms" />
      </div>

      <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.85fr)]">
        <Panel title="Core stream" hint={apiLive ? "live SSE · POST /score overlay" : "mock SSE · API offline"}>
          <div className="max-h-[min(620px,calc(100vh-280px))] overflow-auto">
            <table className="w-full text-left text-[12px]">
              <thead className="sticky top-0 bg-[#0e1a2c] text-[10px] uppercase tracking-[0.14em] text-slate">
                <tr>
                  <th className="py-2 font-medium">Time</th>
                  <th className="font-medium">Account</th>
                  <th className="font-medium">Ch</th>
                  <th className="text-right font-medium">Amount</th>
                  <th className="text-right font-medium">P</th>
                  <th className="font-medium">Route</th>
                  <th className="font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {[...feed].reverse().map((t) => (
                  <tr
                    key={t.id}
                    onClick={() => openCase(t.id)}
                    className={`cursor-pointer border-t border-white/[0.04] ${
                      t.id === selected.id ? "bg-crimson/[0.07]" : "hover:bg-white/[0.03]"
                    }`}
                  >
                    <td className="py-1.5 font-mono text-[11px] text-slate">{utcStamp(t.ts)}</td>
                    <td className="font-mono">{t.account}</td>
                    <td className="text-slate">{t.channel}</td>
                    <td className="text-right font-mono">{inr(t.amount)}</td>
                    <td className="text-right font-mono">
                      <Tone tone={t.pCalib > 0.6 ? "bad" : t.pCalib > 0.3 ? "warn" : "good"}>
                        {t.pCalib.toFixed(2)}
                      </Tone>
                    </td>
                    <td className="text-[11px] text-slate">
                      {t.route === "FIREWALL_FIRST" ? "FW" : "ML"}
                    </td>
                    <td>
                      <StatusPill status={t.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel
            title="Focused case"
            hint={selected.id}
            action={<StatusPill status={selected.status} />}
          >
            <div className="space-y-3 text-[13px]">
              <div className="flex items-baseline justify-between">
                <span className="font-mono">{selected.account}</span>
                <span className="font-mono text-[22px] tabular-nums">
                  {(selected.pCalib * 100).toFixed(1)}
                  <span className="ml-1 text-[11px] text-slate">%</span>
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[12px] text-slate">
                <span>{selected.channel}</span>
                <span className="text-right font-mono text-foam">{inr(selected.amount)}</span>
                <span>{selected.govHit ? "I4C match" : "No gov match"}</span>
                <span className="text-right">{selected.latencyMs.toFixed(1)} ms</span>
              </div>
              <Link
                href={`/analysis/${selected.id}`}
                onClick={() => selectTx(selected.id)}
                className="inline-flex w-full items-center justify-center rounded-md bg-foam py-2 text-[13px] font-medium text-navy-deep no-underline hover:bg-paper"
              >
                Open detailed analysis
              </Link>
            </div>
          </Panel>

          <Panel title="Review queue" hint="HITL · non-clear only">
            <div className="max-h-[220px] space-y-1 overflow-auto">
              {queue.length === 0 ? (
                <p className="text-[12px] text-slate">Queue empty.</p>
              ) : (
                queue.map((t) => (
                  <Link
                    key={t.id}
                    href={`/analysis/${t.id}`}
                    onClick={() => selectTx(t.id)}
                    className="flex items-center justify-between rounded-md px-2 py-1.5 text-[12px] no-underline hover:bg-white/[0.04]"
                  >
                    <span className="font-mono text-foam">{t.account}</span>
                    <span className="flex items-center gap-2">
                      <StatusPill status={t.status} />
                      <span className="font-mono text-crimson">{(t.pCalib * 100).toFixed(0)}%</span>
                    </span>
                  </Link>
                ))
              )}
            </div>
          </Panel>

          <Panel title="Score a transaction" hint="POST /score">
            <div className="space-y-2.5">
              <select
                className="sf-select"
                value={sim.account}
                onChange={(e) => setSim({ ...sim, account: e.target.value })}
              >
                {SAMPLE_ACCOUNTS.map((a) => (
                  <option key={a.account} value={a.account}>
                    {a.account} — {a.note}
                  </option>
                ))}
              </select>
              <div className="grid grid-cols-2 gap-2">
                <select
                  className="sf-select"
                  value={sim.channel}
                  onChange={(e) =>
                    setSim({ ...sim, channel: e.target.value as Transaction["channel"] })
                  }
                >
                  {CHANNELS.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
                <input
                  className="sf-input font-mono"
                  type="number"
                  min={100}
                  step={500}
                  value={sim.amount}
                  onChange={(e) => setSim({ ...sim, amount: Number(e.target.value) })}
                />
              </div>
              <button
                onClick={runScore}
                className="w-full rounded-md border border-[var(--line)] py-2 text-[13px] hover:border-slate"
              >
                {scoring ? "Scoring…" : "Submit /score"}
              </button>
              {simOut ? (
                <Link
                  href={`/analysis/${simOut.id}`}
                  className="block text-center text-[11px] text-slate no-underline hover:text-foam"
                >
                  {simOut.id} scored · P {simOut.pCalib.toFixed(3)} · view analysis
                </Link>
              ) : null}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
