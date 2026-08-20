import type { FeatureToggle, GovTicket, ModelCard, Transaction } from "./types";

const CHANNELS = ["UPI", "ATM", "IMPS", "NEFT", "NETBANK"] as const;

const SHAP_POOL = [
  { feature: "V_cross", label: "ATM+IMPS out / UPI in" },
  { feature: "txn_accel", label: "Txn acceleration / age" },
  { feature: "F3889_comp_lag", label: "Account recency lag" },
  { feature: "iso_anomaly_score", label: "Isolation Forest" },
  { feature: "F3888_month_sin", label: "Open-date seasonality" },
  { feature: "row_zero_rate", label: "Dormant / zero-rate" },
  { feature: "ch_upi_in_L7D", label: "UPI inbound L7" },
  { feature: "F3891_ord", label: "Occupation encoding" },
];

function hash(n: number) {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

export function makeGraph(seed: number, highRisk: boolean): Transaction["graph"] {
  const ego = `A${(100000 + seed).toString().slice(-6)}`;
  const n1 = Array.from({ length: 4 }, (_, i) => `B${seed}${i}`);
  const n2 = Array.from({ length: 5 }, (_, i) => `C${seed}${i}`);
  const nodes = [
    { id: ego, kind: "ego" as const, label: ego, risk: highRisk ? 0.86 : 0.12 },
    ...n1.map((id, i) => ({
      id,
      kind: "hop1" as const,
      label: id,
      risk: highRisk ? 0.35 + hash(seed + i) * 0.5 : 0.08,
    })),
    ...n2.map((id, i) => ({
      id,
      kind: i === 0 && highRisk ? ("blacklist" as const) : ("hop2" as const),
      label: id,
      risk: highRisk && i === 0 ? 0.97 : 0.1 + hash(seed + 20 + i) * 0.2,
    })),
  ];
  const edges = [
    ...n1.map((id, i) => ({
      from: ego,
      to: id,
      amount: Math.round(8000 + hash(seed + i) * 180000),
      channel: CHANNELS[(seed + i) % CHANNELS.length],
      ttl: i % 2 === 0 ? ("24h" as const) : ("7d" as const),
    })),
    ...n2.map((id, i) => ({
      from: n1[i % n1.length],
      to: id,
      amount: Math.round(3000 + hash(seed + 9 + i) * 90000),
      channel: CHANNELS[(seed + 2 + i) % CHANNELS.length],
      ttl: "7d" as const,
    })),
  ];
  return { nodes, edges };
}

export function makeTx(i: number, now = Date.now()): Transaction {
  const h = hash(i + 17);
  const high = h > 0.82;
  const mid = h > 0.7;
  const pCalib = high ? 0.72 + h * 0.25 : mid ? 0.28 + h * 0.25 : 0.01 + h * 0.08;
  const govHit = high && hash(i + 3) > 0.55;
  const route = govHit || pCalib > 0.55 ? "FIREWALL_FIRST" : "ML_FIRST";
  const status = pCalib >= 0.62 ? "ESCROW" : pCalib >= 0.32 ? "QUEUE" : "CLEAR";
  const redis = 1.1 + hash(i + 21) * 2.0;
  const reconstruct = 4.0 + hash(i + 22) * 4.0;
  const gbdt = 11.0 + hash(i + 23) * 8.0;
  const pchip = 0.25 + hash(i + 24) * 0.35;
  const shap = high ? 16 + hash(i + 25) * 8 : 0;
  const latency = { redis, reconstruct, gbdt, pchip, shap };
  const shapBars = SHAP_POOL.map((s, k) => ({
    ...s,
    value: (hash(i * 13 + k) - 0.42) * (high ? 0.22 : 0.08),
  })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return {
    id: `TX-${(900000 + i).toString()}`,
    ts: new Date(now - (40 - i) * 1400).toISOString(),
    account: `XXXX${(2200 + (i % 780)).toString().padStart(4, "0")}`,
    channel: CHANNELS[i % CHANNELS.length],
    amount: Math.round(500 + h * (high ? 420000 : 48000)),
    beneficiary: `UPI@${["okhdfc", "ybl", "axl", "ibl"][i % 4]}`,
    pRaw: Math.min(0.99, pCalib * 1.15),
    pCalib,
    latencyMs: redis + reconstruct + gbdt + pchip + shap,
    latency,
    route,
    status,
    holdUntil: status === "ESCROW" ? new Date(now + 15 * 60 * 1000 - (i % 7) * 40000).toISOString() : undefined,
    tmsFlags: high ? ["VEL_L7", "OFF_HOUR"].slice(0, 1 + (i % 2)) : i % 9 === 0 ? ["NEW_PAYEE"] : [],
    govHit,
    shap: shapBars,
    graph: makeGraph(i, high),
    profile: {
      upiL7: Math.round(hash(i + 30) * 48),
      atmL7: Math.round(hash(i + 31) * 12),
      cardL7: Math.round(hash(i + 32) * 20),
      netL7: Math.round(hash(i + 33) * 9),
      vCross: Number((hash(i + 34) * 4).toFixed(2)),
      accel: Number((hash(i + 35) * 1.8).toFixed(2)),
    },
  };
}

const T0 = Date.UTC(2026, 7, 20, 11, 30, 0);

export const SEED_FEED: Transaction[] = Array.from({ length: 28 }, (_, i) => makeTx(i, T0));

export const MODELS: ModelCard[] = [
  {
    id: "M1",
    name: "Anchor-free robust",
    tag: "DEFAULT",
    sizeMb: 3.8,
    prAuc: 0.738,
    rocAuc: 0.867,
    macroF1: 0.874,
    minorityF1: 0.75,
    cost: 27,
    p99Ms: 49.8,
    notes: "105 non-leaky tracks. Cyclical F3888. No elapsed_days. Production default.",
  },
  {
    id: "M2",
    name: "Chronological baseline",
    tag: "ANCHOR ON",
    sizeMb: 3.9,
    prAuc: 0.71,
    rocAuc: 0.855,
    macroF1: 0.904,
    minorityF1: 0.81,
    cost: 24,
    p99Ms: 51.2,
    notes: "Same GBDT with absolute elapsed_days kept as a drift ablation.",
  },
  {
    id: "M3",
    name: "Audited leaky",
    tag: "DEMO ONLY",
    sizeMb: 4.1,
    prAuc: 0.868,
    rocAuc: 0.984,
    macroF1: 0.915,
    minorityF1: 0.831,
    cost: 12,
    p99Ms: 48.1,
    notes: "F3912, F2230, F3886, F3889, F3891, F3892 left on — judge leakage demo.",
  },
  {
    id: "M4",
    name: "User-custom retrain",
    tag: "EMPTY",
    sizeMb: 0,
    prAuc: 0,
    rocAuc: 0,
    macroF1: 0,
    minorityF1: 0,
    cost: 0,
    p99Ms: 0,
    notes: "Filled by POST /retrain. 90-day label-delay buffer. Slot <5 MB.",
  },
];

export const FEATURE_TOGGLES: FeatureToggle[] = [
  { id: "moments", label: "14 behavioral moments", on: true, leaky: false },
  { id: "gaps", label: "Missingness gap flags", on: true, leaky: false },
  { id: "cats", label: "Fold-safe ordinal / Laplace TE", on: true, leaky: false },
  { id: "cyclical", label: "Cyclical F3888 sin/cos", on: true, leaky: false },
  { id: "iso", label: "Isolation Forest (normal class)", on: true, leaky: false },
  { id: "vcross", label: "V_cross / txn_accel", on: true, leaky: false },
  { id: "elapsed", label: "elapsed_days (M2 anchor)", on: false, leaky: false },
  { id: "F3912", label: "F3912 target proxy", on: false, leaky: true },
  { id: "F2230", label: "F2230 month stamp", on: false, leaky: true },
  { id: "F3886", label: "F3886 account-type leak flag", on: false, leaky: true },
  { id: "F3889", label: "F3889 raw (use lag instead)", on: false, leaky: true },
  { id: "F3891", label: "F3891 prior-audit invention", on: false, leaky: true },
  { id: "F3892", label: "F3892 gender leak tag", on: false, leaky: true },
];

export function makeGov(i: number, now = Date.now()): GovTicket {
  const kinds = ["IP_SUBNET", "DEVICE", "ACCOUNT"] as const;
  const kind = kinds[i % 3];
  const value =
    kind === "IP_SUBNET"
      ? `103.21.${i % 250}.0/24`
      : kind === "DEVICE"
        ? `IMEI-35${(100000 + i).toString().slice(-6)}`
        : `XXXX${(4400 + (i % 300)).toString().padStart(4, "0")}`;
  return {
    id: `I4C-${8000 + i}`,
    ts: new Date(now - i * 9000).toISOString(),
    kind,
    value,
    src: i % 2 === 0 ? "I4C" : "NCRP",
  };
}

export const SEED_GOV: GovTicket[] = Array.from({ length: 8 }, (_, i) =>
  makeGov(i, Date.UTC(2026, 7, 20, 11, 30, 0)),
);

/** Firewall/blacklist lookup vs GBDT inference, relative rupee-units. */
export const C_SEARCH = 0.4;
export const C_INFERENCE = 2.8;

export const SAMPLE_ACCOUNTS = [
  { account: "XXXX2203", note: "Retail · UPI-heavy" },
  { account: "XXXX2411", note: "Salary · ATM + NEFT" },
  { account: "XXXX2688", note: "New payee burst" },
  { account: "XXXX2904", note: "Dormant wake-up" },
] as const;

export const PIPELINE = [
  "Redis profile",
  "14 moments",
  "Cyclical F3888",
  "Laplace TE",
  "Isolation Forest",
  "XGB+LGB",
  "PCHIP",
  "Cost threshold",
] as const;

export function scoreSim(
  input: { account: string; channel: Transaction["channel"]; amount: number; i: number },
  now = Date.now(),
): Transaction {
  const base = makeTx(input.i, now);
  const amountRisk = Math.min(0.42, input.amount / 650000);
  const chBoost = input.channel === "UPI" || input.channel === "IMPS" ? 0.06 : 0;
  const pCalib = Math.min(0.97, 0.04 + amountRisk + chBoost + hash(input.i) * 0.12);
  const govHit = pCalib > 0.7 && hash(input.i + 3) > 0.4;
  const route = govHit || pCalib > 0.55 ? "FIREWALL_FIRST" : "ML_FIRST";
  const status = pCalib >= 0.62 ? "ESCROW" : pCalib >= 0.32 ? "QUEUE" : "CLEAR";
  const shapScale = pCalib > 0.45 ? 0.22 : 0.08;
  return {
    ...base,
    account: input.account,
    channel: input.channel,
    amount: input.amount,
    pCalib,
    pRaw: Math.min(0.99, pCalib * 1.12),
    route,
    status,
    govHit,
    holdUntil:
      status === "ESCROW" ? new Date(now + 15 * 60 * 1000).toISOString() : undefined,
    shap: base.shap
      .map((s, k) => ({ ...s, value: (hash(input.i * 7 + k) - 0.42) * shapScale }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
    graph: makeGraph(input.i, pCalib > 0.55),
  };
}
