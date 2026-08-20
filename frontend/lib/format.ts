export function inr(n: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(n);
}

export function clockTime(d: Date | null) {
  return d ? d.toLocaleTimeString("en-IN", { hour12: false }) : "—:—:—";
}

export function utcStamp(iso: string) {
  return new Date(iso).toISOString().slice(11, 19);
}

export function holdLeft(holdUntil: string | undefined, now: Date | null) {
  if (!holdUntil || !now) return null;
  const ms = new Date(holdUntil).getTime() - now.getTime();
  if (ms <= 0) return "released";
  const m = Math.floor(ms / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
