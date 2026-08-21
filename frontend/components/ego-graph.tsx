"use client";

import { useEffect, useRef } from "react";
import { useConsole } from "@/lib/store";
import type { Transaction } from "@/lib/types";

export function EgoGraph({ tx, className = "h-[240px]" }: { tx: Transaction; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { theme } = useConsole();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const isLight = theme === "light";

    const nodes = tx.graph.nodes.map((n) => {
      const ring = n.kind === "ego" ? 0 : n.kind === "hop1" ? 1 : 2;
      const kind = n.kind === "blacklist" ? "hop2" : n.kind;
      const count = tx.graph.nodes.filter((x) =>
        kind === "ego" ? x.kind === "ego" : kind === "hop1" ? x.kind === "hop1" : x.kind !== "ego" && x.kind !== "hop1",
      ).length;
      const idx = tx.graph.nodes.filter((x) =>
        kind === "ego" ? x.kind === "ego" : kind === "hop1" ? x.kind === "hop1" : x.kind !== "ego" && x.kind !== "hop1",
      ).indexOf(n);
      const r = ring === 0 ? 0 : ring === 1 ? 72 : 118;
      const a = (idx / Math.max(count, 1)) * Math.PI * 2 - Math.PI / 2;
      return { ...n, x: w / 2 + Math.cos(a) * r, y: h / 2 + Math.sin(a) * r };
    });
    const pos = Object.fromEntries(nodes.map((n) => [n.id, n]));

    ctx.clearRect(0, 0, w, h);

    for (const e of tx.graph.edges) {
      const a = pos[e.from];
      const b = pos[e.to];
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.strokeStyle =
        e.amount > 80000
          ? isLight
            ? "rgba(220, 38, 38, 0.75)"
            : "rgba(230, 57, 70, 0.65)"
          : isLight
            ? "rgba(71, 85, 105, 0.45)"
            : "rgba(69, 123, 157, 0.45)";
      ctx.lineWidth = e.amount > 80000 ? 1.8 : 1.1;
      ctx.setLineDash(e.ttl === "24h" ? [4, 3] : []);
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    for (const n of nodes) {
      const color =
        n.kind === "ego"
          ? isLight
            ? "#dc2626"
            : "#e63946"
          : n.kind === "blacklist"
            ? isLight
              ? "#d97706"
              : "#e9c46a"
            : n.kind === "hop1"
              ? isLight
                ? "#1d3557"
                : "#457b9d"
              : isLight
                ? "#64748b"
                : "#7ea3bb";
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(n.x, n.y, n.kind === "ego" ? 8 : 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = isLight ? "#0f172a" : "#f1faee";
      ctx.font = "500 10px IBM Plex Mono, ui-monospace, monospace";
      ctx.fillText(n.label, n.x + 9, n.y + 3);
    }
  }, [tx, theme]);

  return (
    <div className={`relative w-full ${className}`}>
      <canvas ref={ref} className="h-full w-full" />
      <div className="pointer-events-none absolute bottom-1.5 left-3 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-slate">
        <span className="text-crimson font-medium">● ego</span>
        <span className="font-medium">● 1-hop</span>
        <span className="text-warn font-medium">● I4C</span>
        <span>● 2-hop</span>
        <span className="opacity-70">dashed 24h TTL · solid 7d</span>
      </div>
    </div>
  );
}
