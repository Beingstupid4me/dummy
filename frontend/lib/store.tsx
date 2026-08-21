"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  API_BASE,
  apiHealth,
  fetchRegistry,
  mapScore,
  postRetrain,
  postScore,
  retrainStatus,
  setActiveModel,
} from "@/lib/api";
import {
  C_INFERENCE,
  C_SEARCH,
  FEATURE_TOGGLES,
  MODELS,
  SAMPLE_ACCOUNTS,
  SEED_FEED,
  SEED_GOV,
  makeGov,
  makeTx,
  scoreSim,
} from "@/lib/demo";
import { holdLeft } from "@/lib/format";
import type {
  FeatureToggle,
  GovTicket,
  ModelCard,
  ModelId,
  Transaction,
  WebhookEvent,
} from "@/lib/types";

export type SimForm = {
  account: string;
  channel: Transaction["channel"];
  amount: number;
};

type ConsoleState = {
  theme: "dark" | "light";
  feed: Transaction[];
  selected: Transaction;
  sel: string;
  model: ModelId;
  models: ModelCard[];
  active: ModelCard;
  toggles: FeatureToggle[];
  retrain: "idle" | "run" | "done";
  log: string[];
  clock: Date | null;
  gov: GovTicket[];
  hooks: WebhookEvent[];
  toast: string | null;
  smote: number;
  sim: SimForm;
  simOut: Transaction | null;
  scoring: boolean;
  apiLive: boolean;
  queue: Transaction[];
  holds: number;
  p99: number;
  escrowLeft: string | null;
  unitCost: number;
  selectTx: (id: string) => void;
  swapModel: (id: ModelId) => void;
  setSim: (next: SimForm) => void;
  runScore: () => void;
  toggleFeature: (id: string) => void;
  setSmote: (n: number) => void;
  startRetrain: () => void;
  toggleTheme: () => void;
};

const Ctx = createContext<ConsoleState | null>(null);

function pushEscrow(tx: Transaction, setter: (fn: (prev: WebhookEvent[]) => WebhookEvent[]) => void) {
  if (tx.status !== "ESCROW") return;
  setter((prev) =>
    [
      ...prev,
      {
        id: `WH-${tx.id}`,
        txId: tx.id,
        ts: tx.ts,
        endpoint: "POST /webhooks/escrow",
        holdMin: 15,
      },
    ].slice(-12),
  );
}

export function ConsoleProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [feed, setFeed] = useState<Transaction[]>(SEED_FEED);
  const [sel, setSel] = useState(SEED_FEED[SEED_FEED.length - 1].id);
  const [model, setModel] = useState<ModelId>("M1");
  const [toggles, setToggles] = useState<FeatureToggle[]>(FEATURE_TOGGLES);
  const [retrain, setRetrain] = useState<"idle" | "run" | "done">("idle");
  const [log, setLog] = useState<string[]>([]);
  const [models, setModels] = useState(MODELS);
  const [clock, setClock] = useState<Date | null>(null);
  const [gov, setGov] = useState<GovTicket[]>(SEED_GOV);
  const [hooks, setHooks] = useState<WebhookEvent[]>([]);
  const [toast, setToast] = useState<string | null>(null);
  const [smote, setSmote] = useState(0);
  const [sim, setSim] = useState<SimForm>({
    account: SAMPLE_ACCOUNTS[0].account,
    channel: "UPI",
    amount: 185000,
  });
  const [simOut, setSimOut] = useState<Transaction | null>(null);
  const [scoring, setScoring] = useState(false);
  const [apiLive, setApiLive] = useState(false);
  const apiLiveRef = useRef(false);
  const mockI = useRef(SEED_FEED.length);

  useEffect(() => {
    try {
      const savedTheme = localStorage.getItem("sf_theme") as "dark" | "light" | null;
      if (savedTheme === "light" || savedTheme === "dark") {
        setTheme(savedTheme);
        document.documentElement.setAttribute("data-theme", savedTheme);
      } else {
        document.documentElement.setAttribute("data-theme", "dark");
      }
    } catch {
      /* ignore */
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      try {
        localStorage.setItem("sf_theme", next);
        document.documentElement.setAttribute("data-theme", next);
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  const selected = useMemo(
    () => feed.find((t) => t.id === sel) ?? feed[feed.length - 1],
    [feed, sel],
  );
  const active = models.find((m) => m.id === model) ?? models[0] ?? MODELS[0];
  const queue = useMemo(
    () => feed.filter((t) => t.status !== "CLEAR").slice(-10).reverse(),
    [feed],
  );
  const holds = feed.filter((t) => t.status === "ESCROW").length;
  const p99 = feed.slice(-24).reduce((a, t) => Math.max(a, t.latencyMs), 0);
  const escrowLeft = holdLeft(selected.holdUntil, clock);
  const leakyOn = toggles.some((t) => t.leaky && t.on);
  const unitCost = selected.route === "FIREWALL_FIRST" ? C_SEARCH : C_INFERENCE;

  useEffect(() => {
    const tick = window.setInterval(() => setClock(new Date()), 1000);
    let mockId: number | undefined;
    let es: EventSource | null = null;
    let cancelled = false;

    function startMock() {
      if (mockId !== undefined) return;
      mockId = window.setInterval(() => {
        mockI.current += 1;
        const now = Date.now();
        const tx = makeTx(mockI.current, now);
        setFeed((prev) => [...prev.slice(-47), tx]);
        pushEscrow(tx, setHooks);
        if (mockI.current % 5 === 0) setGov((prev) => [makeGov(mockI.current, now), ...prev].slice(0, 12));
      }, 1800);
    }

    async function boot() {
      try {
        const health = await apiHealth();
        if (cancelled) return;
        const live = Boolean(health.ok);
        apiLiveRef.current = live;
        setApiLive(live);
        const cards = await fetchRegistry();
        if (!cancelled && Array.isArray(cards) && cards.length) {
          setModels(cards);
          const act = cards.find((c) => (c as ModelCard & { active?: boolean }).active);
          if (act?.id) setModel(act.id);
        }
        // Fetch live gov tickets from backend
        try {
          const { fetchGovTickets, fetchEscrowWebhooks } = await import("@/lib/api");
          const govTickets = await fetchGovTickets();
          if (!cancelled && Array.isArray(govTickets) && govTickets.length) {
            setGov(govTickets);
          }
          const hooksList = await fetchEscrowWebhooks();
          if (!cancelled && Array.isArray(hooksList) && hooksList.length) {
            setHooks(hooksList);
          }
        } catch {
          /* fallback to seeds */
        }
      } catch {
        if (!cancelled) {
          apiLiveRef.current = false;
          setApiLive(false);
        }
      }

      if (cancelled) return;
      if (apiLiveRef.current) {
        es = new EventSource(`${API_BASE}/stream`);
        es.onmessage = (ev) => {
          try {
            const raw = JSON.parse(ev.data) as Record<string, unknown>;
            if (raw.error) return;
            const tx = mapScore(raw);
            setFeed((prev) => [...prev.slice(-47), tx]);
            pushEscrow(tx, setHooks);
          } catch {
            /* ignore malformed SSE */
          }
        };
        es.onerror = () => {
          es?.close();
          es = null;
          apiLiveRef.current = false;
          setApiLive(false);
          startMock();
        };
      } else {
        startMock();
      }
    }

    void boot();
    return () => {
      cancelled = true;
      window.clearInterval(tick);
      if (mockId !== undefined) window.clearInterval(mockId);
      es?.close();
    };
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), 2200);
    return () => window.clearTimeout(t);
  }, [toast]);

  const selectTx = useCallback((id: string) => setSel(id), []);

  const swapModel = useCallback((id: ModelId) => {
    if (id === "M4" && models.find((m) => m.id === "M4")?.tag === "EMPTY") return;
    void (async () => {
      if (apiLiveRef.current) {
        try {
          await setActiveModel(id);
        } catch {
          setToast(`${id} empty · train or bootstrap first`);
          return;
        }
      }
      setModel((from) => {
        if (from !== id) setToast(`Hot-swap ${from} → ${id} · 0 ms downtime`);
        return id;
      });
    })();
  }, [models]);

  const runScore = useCallback(() => {
    if (scoring) return;
    setScoring(true);
    void (async () => {
      try {
        let tx: Transaction;
        if (apiLiveRef.current) {
          tx = await postScore(sim);
        } else {
          tx = scoreSim({ ...sim, i: feed.length + 80 }, Date.now());
        }
        setSimOut(tx);
        setFeed((prev) => [...prev.slice(-47), tx]);
        setSel(tx.id);
        pushEscrow(tx, setHooks);
      } catch {
        const tx = scoreSim({ ...sim, i: feed.length + 80 }, Date.now());
        setSimOut(tx);
        setFeed((prev) => [...prev.slice(-47), tx]);
        setSel(tx.id);
        setToast("API /score failed · mock result");
      } finally {
        setScoring(false);
      }
    })();
  }, [feed.length, scoring, sim]);

  const toggleFeature = useCallback((id: string) => {
    setToggles((prev) => prev.map((x) => (x.id === id ? { ...x, on: !x.on } : x)));
  }, []);

  const startRetrain = useCallback(() => {
    if (retrain === "run") return;
    setRetrain("run");
    setLog([]);
    const smoteLine =
      smote > 0
        ? `SMOTE ratio ${smote.toFixed(2)} · minority upsample`
        : "SMOTE skipped · scale_pos_weight";

    if (apiLiveRef.current) {
      void (async () => {
        try {
          const featuresOn = toggles.filter((t) => t.on).map((t) => t.id);
          const featuresOff = toggles.filter((t) => !t.on).map((t) => t.id);
          const { job_id } = await postRetrain({
            features_on: featuresOn,
            features_off: featuresOff,
            include_elapsed: toggles.some((t) => t.id === "elapsed" && t.on),
            include_leaky: leakyOn,
            include_tms: toggles.some((t) => t.id === "tms" && t.on),
            smote_ratio: smote,
          });
          setLog([`HTTP 202  POST /retrain  ${job_id}`, "90-day label-delay buffer applied", smoteLine]);
          
          // Connect to SSE stream for live retrain progress
          let retrainES: EventSource | null = null;
          try {
            retrainES = new EventSource(`${API_BASE}/retrain/${job_id}/stream`);
            retrainES.onmessage = (ev) => {
              try {
                const data = JSON.parse(ev.data);
                if (data.line) {
                  setLog((prev) => [...prev, data.line]);
                }
                if (data.done || ["done", "aborted", "error"].includes(data.status)) {
                  retrainES?.close();
                  setRetrain(data.status === "done" ? "done" : "idle");
                  if (data.status === "done") {
                    setToast("M4 registered · 0 ms cutover");
                    void fetchRegistry().then((cards) => {
                      setModels(cards);
                      setModel("M4");
                    });
                  } else {
                    setToast(`Retrain ${data.status}`);
                  }
                }
              } catch {
                /* ignore */
              }
            };
            retrainES.onerror = () => {
              retrainES?.close();
              // fallback to polling if SSE drops
              startPollingFallback(job_id);
            };
          } catch {
            startPollingFallback(job_id);
          }

          function startPollingFallback(id: string) {
            const t0 = performance.now();
            const poll = window.setInterval(() => {
              void (async () => {
                try {
                  const st = await retrainStatus(id);
                  if (st.log?.length) setLog(st.log);
                  if (["done", "aborted", "error"].includes(st.status)) {
                    window.clearInterval(poll);
                    setRetrain(st.status === "done" ? "done" : "idle");
                    if (st.status === "done") {
                      setToast("M4 registered · 0 ms cutover");
                      const cards = await fetchRegistry();
                      setModels(cards);
                      setModel("M4");
                    } else {
                      setToast(`Retrain ${st.status}`);
                    }
                  }
                  if ((performance.now() - t0) / 1000 > 240) {
                    window.clearInterval(poll);
                    setLog((prev) => [...prev, "still running in worker — check GET /retrain/{id}"]);
                    setRetrain("idle");
                  }
                } catch {
                  window.clearInterval(poll);
                  setRetrain("idle");
                  setToast("Retrain poll failed");
                }
              })();
            }, 1600);
          }
        } catch {
          setLog((prev) => [...prev, "API /retrain failed · demo playback"]);
          setRetrain("idle");
        }
      })();
      return;
    }

    const steps = [
      "HTTP 202  POST /retrain  accepted",
      "90-day label-delay buffer applied",
      smoteLine,
      leakyOn
        ? "Leaky tracks included — demo only"
        : "Feature reconstruction · moments / TE / V_cross",
      "XGB+LGB 0.6/0.4  ·  ProcessPoolExecutor",
      "PCHIP isotonic  ·  f′(x) > 0  passed",
      "Cost threshold search  ·  Model 4 registered < 5 MB",
    ];
    const t0 = performance.now();
    steps.forEach((line, idx) => {
      window.setTimeout(() => {
        setLog((prev) => [...prev, line]);
        if (idx === steps.length - 1) {
          const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
          setLog((prev) => [...prev, `Registered M4 in ${elapsed}s  ·  SLA 15s`]);
          setRetrain("done");
          setModel("M4");
          setToast("M4 registered · 0 ms cutover");
          setModels((prev) =>
            prev.map((m) =>
              m.id === "M4"
                ? {
                    ...m,
                    tag: "CUSTOM",
                    sizeMb: 3.7,
                    prAuc: leakyOn ? 0.821 : 0.741,
                    rocAuc: leakyOn ? 0.94 : 0.87,
                    macroF1: leakyOn ? 0.89 : 0.838,
                    minorityF1: leakyOn ? 0.79 : 0.68,
                    cost: leakyOn ? 18 : 26,
                    p99Ms: 47.4,
                    notes: leakyOn
                      ? "Operator retrain with leaky tracks. Not deployable."
                      : "Operator retrain. Anchor-free. 90-day buffer enforced.",
                  }
                : m,
            ),
          );
        }
      }, 520 * (idx + 1));
    });
  }, [leakyOn, retrain, smote, toggles]);

  const value: ConsoleState = {
    theme,
    feed,
    selected,
    sel,
    model,
    models,
    active,
    toggles,
    retrain,
    log,
    clock,
    gov,
    hooks,
    toast,
    smote,
    sim,
    simOut,
    scoring,
    apiLive,
    queue,
    holds,
    p99,
    escrowLeft,
    unitCost,
    selectTx,
    swapModel,
    setSim,
    runScore,
    toggleFeature,
    setSmote,
    startRetrain,
    toggleTheme,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useConsole() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useConsole must be used within ConsoleProvider");
  return ctx;
}
