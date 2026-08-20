export type RoutePath = "ML_FIRST" | "FIREWALL_FIRST";
export type AlertStatus = "CLEAR" | "QUEUE" | "ESCROW";
export type ModelId = "M1" | "M2" | "M3" | "M4";

export type ShapBar = {
  feature: string;
  label: string;
  value: number;
};

export type GraphNode = {
  id: string;
  kind: "ego" | "hop1" | "hop2" | "blacklist";
  label: string;
  risk: number;
};

export type GraphEdge = {
  from: string;
  to: string;
  amount: number;
  channel: string;
  ttl: "24h" | "7d";
};

export type LatencyBreak = {
  redis: number;
  reconstruct: number;
  gbdt: number;
  pchip: number;
  shap: number;
};

export type RedisProfile = {
  upiL7: number;
  atmL7: number;
  cardL7: number;
  netL7: number;
  vCross: number;
  accel: number;
};

export type Transaction = {
  id: string;
  ts: string;
  account: string;
  channel: "UPI" | "ATM" | "IMPS" | "NEFT" | "NETBANK";
  amount: number;
  beneficiary: string;
  pRaw: number;
  pCalib: number;
  latencyMs: number;
  latency: LatencyBreak;
  route: RoutePath;
  status: AlertStatus;
  holdUntil?: string;
  tmsFlags: string[];
  govHit: boolean;
  shap: ShapBar[];
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  profile: RedisProfile;
};

export type GovTicket = {
  id: string;
  ts: string;
  kind: "IP_SUBNET" | "DEVICE" | "ACCOUNT";
  value: string;
  src: "I4C" | "NCRP";
};

export type WebhookEvent = {
  id: string;
  txId: string;
  ts: string;
  endpoint: string;
  holdMin: number;
};

export type ModelCard = {
  id: ModelId;
  name: string;
  tag: string;
  sizeMb: number;
  prAuc: number;
  rocAuc: number;
  macroF1: number;
  minorityF1: number;
  cost: number;
  p99Ms: number;
  notes: string;
  active?: boolean;
};

export type FeatureToggle = {
  id: string;
  label: string;
  on: boolean;
  leaky: boolean;
};

export type SimInput = {
  account: string;
  channel: Transaction["channel"];
  amount: number;
};
