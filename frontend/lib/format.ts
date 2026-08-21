export function inr(n: number | undefined | null) {
  if (n === undefined || n === null || isNaN(Number(n))) return "₹0";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(Number(n));
}

export function clockTime(d: Date | null | undefined) {
  if (!d) return "—:—:—";
  try {
    if (isNaN(d.getTime())) return "—:—:—";
    return d.toLocaleTimeString("en-IN", { hour12: false });
  } catch {
    return "—:—:—";
  }
}

export function utcStamp(iso: string | null | undefined) {
  if (!iso) return "—:—:—";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "—:—:—";
    return d.toISOString().slice(11, 19);
  } catch {
    return "—:—:—";
  }
}

export function holdLeft(holdUntil: string | undefined | null, now: Date | null | undefined) {
  if (!holdUntil || !now) return null;
  try {
    const d = new Date(holdUntil);
    if (isNaN(d.getTime()) || isNaN(now.getTime())) return null;
    const ms = d.getTime() - now.getTime();
    if (ms <= 0) return "released";
    const m = Math.floor(ms / 60000);
    const s = Math.floor((ms % 60000) / 1000);
    return `${m}:${s.toString().padStart(2, "0")}`;
  } catch {
    return null;
  }
}
