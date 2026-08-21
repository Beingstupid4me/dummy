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
  features_on?: string[];
  features_off?: string[];
  include_elapsed: boolean;
  include_leaky: boolean;
  include_tms?: boolean;
  smote_ratio: number;
}): Promise<{ job_id: string }> {
  const r = await fetch(`${API_BASE}/retrain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      features_on: body.features_on ?? [],
      features_off: body.features_off ?? [],
      include_elapsed: body.include_elapsed,
      include_leaky: body.include_leaky,
      include_tms: body.include_tms ?? false,
      smote_ratio: body.smote_ratio,
      n_estimators: 120,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchGovTickets(): Promise<import("@/lib/types").GovTicket[]> {
  const r = await fetch(`${API_BASE}/feeds/gov`, { cache: "no-store" });
  if (!r.ok) throw new Error(`gov ${r.status}`);
  return r.json();
}

export async function postGovTicket(ticket: {
  id: string;
  kind: "IP_SUBNET" | "DEVICE" | "ACCOUNT";
  value: string;
  src: "I4C" | "NCRP";
}): Promise<{ ok: boolean }> {
  const r = await fetch(`${API_BASE}/feeds/gov`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...ticket, ts: new Date().toISOString() }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchEscrowWebhooks(): Promise<import("@/lib/types").WebhookEvent[]> {
  const r = await fetch(`${API_BASE}/webhooks/escrow`, { cache: "no-store" });
  if (!r.ok) throw new Error(`webhooks ${r.status}`);
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
  const rawTs = raw.ts ? String(raw.ts) : "";
  const validTs = rawTs && !isNaN(new Date(rawTs).getTime()) ? rawTs : new Date().toISOString();
  return {
    id: String(raw.id ?? `TX-${Math.floor(100000 + Math.random() * 900000)}`),
    ts: validTs,
    account: String(raw.account ?? "XXXX0000"),
    channel: (raw.channel as Transaction["channel"]) ?? "UPI",
    amount: Number(raw.amount ?? 0),
    beneficiary: String(raw.beneficiary ?? "UPI@okhdfc"),
    pRaw: Number(raw.pRaw ?? 0),
    pCalib: Number(raw.pCalib ?? 0),
    latencyMs: Number(raw.latencyMs ?? 0),
    latency: (raw.latency as Transaction["latency"]) ?? {
      redis: 0,
      reconstruct: 0,
      gbdt: 0,
      pchip: 0,
      shap: 0,
    },
    route: (raw.route as Transaction["route"]) ?? "ML_FIRST",
    status: (raw.status as Transaction["status"]) ?? "CLEAR",
    holdUntil: hold ? String(hold) : undefined,
    tmsFlags: (raw.tmsFlags as string[]) ?? [],
    govHit: Boolean(raw.govHit),
    shap: (raw.shap as Transaction["shap"]) ?? [],
    graph: {
      nodes: graph.nodes ?? [],
      edges: (graph.edges ?? []).map((e) => ({
        from: String(e.from ?? e.from_),
        to: String(e.to),
        amount: Number(e.amount ?? 0),
        channel: String(e.channel ?? "UPI"),
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
