import type { ModelCard, ModelId, SimInput, Transaction } from "@/lib/types";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "/sf-api";

export async function apiHealth(): Promise<{ ok: boolean; active?: string }> {
  const r = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function fetchRegistry(): Promise<ModelCard[]> {
  const r = await fetch(`${API_BASE}/registry`, { cache: "no-store" });
  if (!r.ok) throw new Error(`registry ${r.status}`);
  return r.json();
}

export async function setActiveModel(model_id: ModelId): Promise<void> {
  const r = await fetch(`${API_BASE}/registry/active`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id }),
  });
  if (!r.ok) throw new Error(await r.text());
}

export async function postScore(input: SimInput): Promise<Transaction> {
  const r = await fetch(`${API_BASE}/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      account: input.account,
      channel: input.channel,
      amount: input.amount,
      beneficiary: "UPI@okhdfc",
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return mapScore(await r.json());
}

export async function postRetrain(body: {
  include_elapsed: boolean;
  include_leaky: boolean;
  smote_ratio: number;
}): Promise<{ job_id: string }> {
  const r = await fetch(`${API_BASE}/retrain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, n_estimators: 120 }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function retrainStatus(jobId: string): Promise<{
  job_id: string;
  status: string;
  log: string[];
  metrics: Record<string, number> | null;
  elapsed_s: number | null;
}> {
  const r = await fetch(`${API_BASE}/retrain/${jobId}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`retrain ${r.status}`);
  return r.json();
}

export function mapScore(raw: Record<string, unknown>): Transaction {
  const graph = (raw.graph ?? {}) as { nodes?: Transaction["graph"]["nodes"]; edges?: Array<Record<string, unknown>> };
  const hold = raw.holdUntil;
  return {
    id: String(raw.id),
    ts: String(raw.ts),
    account: String(raw.account),
    channel: raw.channel as Transaction["channel"],
    amount: Number(raw.amount),
    beneficiary: String(raw.beneficiary ?? "UPI@okhdfc"),
    pRaw: Number(raw.pRaw),
    pCalib: Number(raw.pCalib),
    latencyMs: Number(raw.latencyMs),
    latency: raw.latency as Transaction["latency"],
    route: raw.route as Transaction["route"],
    status: raw.status as Transaction["status"],
    holdUntil: hold ? String(hold) : undefined,
    tmsFlags: (raw.tmsFlags as string[]) ?? [],
    govHit: Boolean(raw.govHit),
    shap: (raw.shap as Transaction["shap"]) ?? [],
    graph: {
      nodes: graph.nodes ?? [],
      edges: (graph.edges ?? []).map((e) => ({
        from: String(e.from ?? e.from_),
        to: String(e.to),
        amount: Number(e.amount),
        channel: String(e.channel),
        ttl: (e.ttl as Transaction["graph"]["edges"][number]["ttl"]) ?? "24h",
      })),
    },
    profile: (raw.profile as Transaction["profile"]) ?? {
      upiL7: 0,
      atmL7: 0,
      cardL7: 0,
      netL7: 0,
      vCross: 0,
      accel: 0,
    },
  };
}
