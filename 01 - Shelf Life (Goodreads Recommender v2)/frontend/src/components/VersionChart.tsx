import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Markdown } from "./Markdown";

/**
 * Model versions (x) against an evaluation metric (y) for the Evaluation page, one line per dataset "era" (each
 * era has its own test split, so scores are only compared within an era; a divider separates them). Each era's
 * line starts at the 2023 Book Recommender scored on that era's test (a hollow diamond, before v1), then, after the
 * first era, the previous era's last version on the new test (the bridge, a hollow circle), then the versions.
 * A details panel beside the chart (below it on phones) shows the current champion by default; hovering a point
 * with a mouse shows it (and it stays when the mouse moves away), and clicking (or Enter/Space) pins it until
 * clicked again, × or Escape. Dashed lines are the era's simple baselines (popular books; the book average).
 * `track` picks ranking (NDCG / Precision / Recall) or rating prediction (MAE and friends, several lower-is-better).
 * The chart and panel share one height that fits the viewport; the SVG is drawn at its measured size.
 */

export interface VersionCommit { short: string; subject: string; url: string }
export interface VersionDecision { id: string; title: string; sections: { label: string; text: string }[] }
export interface Version {
  id: string; date: string; title: string; summary: string; settings: string; rescored: boolean; report: string;
  commits: VersionCommit[]; decisions: VersionDecision[]; champion: boolean; n_users: number;
  metrics: Record<string, Record<string, number>>;            // n ("1".."25", "-1" = full history) -> metric -> value
  rmetrics: Record<string, Record<string, number>> | null;     // rating track, same shape
  ci_vs_previous?: Record<string, PairedCI>;                    // per-user NDCG@10 vs the previous point
  rci_vs_previous?: Record<string, PairedCI>;                   // per-user MAE vs the previous point
  dataset: string; previous: string;                            // previous point on the era's line ("2023", "v6*", "v5")
}
interface PairedCI { mean_diff: number; ci95: [number, number]; win: number; loss: number }
export type Track = "ranking" | "rating";
type Metrics = Record<string, Record<string, number>>;
export interface Reference { key: string; label: string; metrics: Metrics }
interface EraPoint {
  id: string; label: string; kind: "baseline" | "bridge"; metrics: Metrics; rmetrics: Metrics; report: string;
  ranking_method?: string | null; rating_method?: string | null; of?: string;
}
export interface Era {
  id: string; label: string; short: string; versions: string[]; baseline: EraPoint; bridge: EraPoint | null;
  references: Reference[]; rating_references: Reference[];
}
/** One point on the chart: a version, an era's 2023 point or its bridge. */
interface Point { key: string; kind: "version" | "baseline" | "bridge"; era: number; name: string; sub: string; v?: Version; e?: EraPoint }

export const METRIC_LABEL: Record<string, string> = {
  "ndcg@10": "NDCG@10", "precision@10": "Precision@10", "recall@10": "Recall@10",
  "ndcg@20": "NDCG@20", "precision@20": "Precision@20", "recall@20": "Recall@20",
};
export const RATING_LABEL: Record<string, string> = {
  mae: "MAE", rmse: "RMSE", nmae: "MAE / reader's spread", pearson: "Correlation (Pearson)",
  spearman: "Per-reader rank correlation", within1: "Within ±1★",
};
const LOWER_BETTER = new Set(["mae", "rmse", "nmae"]);
const TRACK = {
  ranking: { labels: METRIC_LABEL, first: "ndcg@10", ci: "NDCG@10", pci: (v: Version) => v.ci_vs_previous,
             get: (p: Point): Metrics => (p.v ? p.v.metrics : p.e!.metrics), refs: (e: Era) => e.references },
  rating: { labels: RATING_LABEL, first: "mae", ci: "MAE", pci: (v: Version) => v.rci_vs_previous,
            get: (p: Point): Metrics => (p.v ? p.v.rmetrics ?? {} : p.e!.rmetrics), refs: (e: Era) => e.rating_references },
};
const fmtDate = (d: string) => new Date(`${d}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
const PAD = { l: 76, r: 20, t: 30, b: 54 };
const PHONE = "(max-width: 640px)";

/** Axis from 0 in exactly five even, round steps, topping out comfortably above `max` (at least ~12% headroom),
 * so every metric gets the same gridlines. */
function niceTicks(max: number): number[] {
  const raw = (max * 1.12) / 5;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].map((m) => m * pow).find((s) => s >= raw - 1e-12) ?? raw;
  return Array.from({ length: 6 }, (_, i) => +(i * step).toFixed(6));
}

/** Rating track: values sit in a narrow band (MAE ~0.6-0.8), so the axis spans just that band, still in five
 * even, round steps, with room above and below for labels. */
function bandTicks(min: number, max: number): number[] {
  const pad = Math.max((max - min) * 0.25, 0.01);
  const raw = (max - min + 2 * pad) / 5;
  const pow = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20]) {
    const step = m * pow, lo = Math.max(0, Math.floor((min - pad) / step) * step);
    if (lo + 5 * step >= max + pad - 1e-12) return Array.from({ length: 6 }, (_, i) => +(lo + i * step).toFixed(6));
  }
  return niceTicks(max);
}

export function VersionChart({ versions, eras, buckets, onViewReport, track = "ranking" }: {
  versions: Version[]; eras: Era[]; buckets: number[]; onViewReport: (reportId: string) => void; track?: Track;
}) {
  const T = TRACK[track];
  const [metricBy, setMetricBy] = useState<Record<Track, string>>({ ranking: TRACK.ranking.first, rating: TRACK.rating.first });
  const metric = metricBy[track];
  const setMetric = (m: string) => setMetricBy((s) => ({ ...s, [track]: m }));
  const lower = LOWER_BETTER.has(metric) && track === "rating";
  const axisLabel = `${T.labels[metric]}${lower ? " (lower is better)" : ""}`;
  const [n, setN] = useState("-1");
  const [openId, setOpenId] = useState<string | null>(null);
  const [pinned, setPinned] = useState(false);
  const [height, setHeight] = useState<number | undefined>(undefined);
  const [size, setSize] = useState({ w: 560, h: 340 });
  const section = useRef<HTMLElement>(null);
  const plot = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLElement>(null);

  // Fit the section into the viewport below where it starts (desktop); on phones the chart and panel stack at
  // their natural height.
  useLayoutEffect(() => {
    const fit = () => {
      const el = section.current;
      if (!el) return;
      if (window.matchMedia(PHONE).matches) { setHeight(undefined); return; }
      const top = el.getBoundingClientRect().top + window.scrollY;
      setHeight(Math.round(Math.min(Math.max(window.innerHeight - top - 20, 460), 780)));
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);
  // Draw the SVG at the plot area's measured size, so it fills the space without stretching the text.
  useEffect(() => {
    const el = plot.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      const w = Math.round(e.contentRect.width), h = Math.round(e.contentRect.height);
      if (w > 0) setSize({ w, h: window.matchMedia(PHONE).matches || h < 120 ? Math.round(w * 0.75) : h });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    if (!openId) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { setOpenId(null); setPinned(false); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openId]);

  // The points, era by era: 2023, the bridge (later eras), then the era's versions.
  const byId = Object.fromEntries(versions.map((v) => [v.id, v]));
  const points: Point[] = eras.flatMap((era, ei) => [
    { key: `2023:${era.id}`, kind: "baseline" as const, era: ei, name: "v0", sub: "2023 baseline", e: era.baseline },
    ...(era.bridge ? [{ key: `bridge:${era.id}`, kind: "bridge" as const, era: ei, name: era.bridge.of ?? "", sub: "retest", e: era.bridge }] : []),
    ...era.versions.filter((id) => byId[id]).map((id) => ({
      key: id, kind: "version" as const, era: ei, name: id, sub: fmtDate(byId[id].date).replace(/, \d{4}$/, ""), v: byId[id] })),
  ]);
  const { w: W, h: H } = size;
  const compact = W < 520;            // phones: no value labels or dates under points (the panel has them)
  const val = (m: Metrics) => m[n]?.[metric] ?? 0;
  const pv = (p: Point) => val(T.get(p));
  const ys = [...points.map(pv), ...eras.flatMap((e) => T.refs(e).map((r) => val(r.metrics)))];
  const ticks = track === "rating" ? bandTicks(Math.min(...ys), Math.max(...ys)) : niceTicks(Math.max(...ys, 1e-6));
  const ymin = ticks[0], ymax = ticks[ticks.length - 1];
  const decimals = track === "rating" ? Math.max(2, -Math.floor(Math.log10(ticks[1] - ticks[0]) + 1e-9)) : ymax < 0.1 ? 3 : 2;
  // x: one slot per point, a little extra room after each v0 (its "2023 baseline" label is wider), a gap between eras.
  const GAP = 0.6, AFTER_V0 = 0.25;
  const slot = points.map((p, i) => i + p.era * GAP
    + AFTER_V0 * points.slice(0, i).filter((q) => q.kind === "baseline").length);
  const span = Math.max(slot[slot.length - 1] ?? 0, 1);
  const x0 = PAD.l + 28, x1 = W - PAD.r - 28;
  const x = (i: number) => x0 + (slot[i] / span) * (x1 - x0);
  const y = (v: number) => H - PAD.b - ((v - ymin) / (ymax - ymin)) * (H - PAD.t - PAD.b);
  const yMid = (PAD.t + H - PAD.b) / 2;
  const eraIdx = (ei: number) => points.map((p, i) => (p.era === ei ? i : -1)).filter((i) => i >= 0);
  const eraX = eras.map((_, ei) => {
    const idx = eraIdx(ei);
    const lo = ei === 0 ? PAD.l : (x(idx[0]) + x(eraIdx(ei - 1).slice(-1)[0])) / 2;
    const hi = ei === eras.length - 1 ? W - PAD.r : (x(idx[idx.length - 1]) + x(eraIdx(ei + 1)[0])) / 2;
    return { lo, hi };
  });
  // Baseline labels: above or below their line, wherever they don't cover a point, its value label or another label.
  type Box = { x0: number; x1: number; y0: number; y1: number };
  const hit = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
  const taken: Box[] = points.map((p, k) => ({ x0: x(k) - (compact ? 8 : 30), x1: x(k) + (compact ? 8 : 30),
                                               y0: y(pv(p)) - (compact ? 9 : 34), y1: y(pv(p)) + 8 }));
  const labelPos: Record<string, { x: number; y: number; anchor: "start" | "middle" | "end"; text: string } | null> = {};
  eras.forEach((era, ei) => T.refs(era).forEach((r) => {
    const { lo, hi } = eraX[ei];
    const short = `${r.label.replace(/ books$/, "")} ${val(r.metrics).toFixed(3)}`;
    const ly = y(val(r.metrics));
    type Opt = { x: number; y: number; anchor: "start" | "middle" | "end" };
    const options: Opt[] = [
      { x: lo + 8, y: ly - 6, anchor: "start" }, { x: lo + 8, y: ly + 27, anchor: "start" },
      { x: hi - 8, y: ly - 6, anchor: "end" }, { x: hi - 8, y: ly + 27, anchor: "end" },
      { x: (lo + hi) / 2, y: ly - 6, anchor: "middle" }, { x: (lo + hi) / 2, y: ly + 27, anchor: "middle" },
      ...[0.25, 0.75].flatMap((f): Opt[] => [{ x: lo + f * (hi - lo), y: ly - 6, anchor: "middle" },
                                             { x: lo + f * (hi - lo), y: ly + 27, anchor: "middle" }]),
    ];
    // The full label where it fits, else a short one; with no room at all (a narrow era on a phone), just the line.
    for (const text of compact ? [short] : [`Baseline: ${r.label} (${val(r.metrics).toFixed(3)})`, `Baseline: ${short}`, short]) {
      const w = 5.7 * text.length;
      const left = (o: Opt) => (o.anchor === "start" ? o.x : o.anchor === "end" ? o.x - w : o.x - w / 2);
      const box = (o: Opt): Box => ({ x0: left(o), x1: left(o) + w, y0: o.y - 11, y1: o.y + 3 });
      const inside = (o: Opt) => left(o) >= lo + 2 && left(o) + w <= hi + 2
        && o.y - 11 >= PAD.t + 4 && o.y + 3 <= H - PAD.b - 6;          // clear of the era headers and the x axis
      const pick = options.find((o) => inside(o) && !taken.some((t) => hit(t, box(o))));
      if (pick) {
        taken.push(box(pick));
        labelPos[`${era.id}:${r.key}`] = { ...pick, text };
        return;
      }
    }
    labelPos[`${era.id}:${r.key}`] = null;
  }));

  const show = (id: string) => { if (!pinned) setOpenId(id); };
  const toggle = (id: string) => {
    if (pinned && openId === id) { setOpenId(null); setPinned(false); }
    else { setOpenId(id); setPinned(true); }
  };
  // The panel shows the pinned or last-hovered point (it stays after the mouse leaves), else the current champion.
  const defaultKey = (versions.find((vv) => vv.champion) ?? versions[versions.length - 1])?.id ?? null;
  const shownKey = openId ?? defaultKey;
  const i = points.findIndex((p) => p.key === shownKey);
  const p = i >= 0 ? points[i] : null;
  const prev = p && i > 0 && points[i - 1].era === p.era ? points[i - 1] : null;
  const v = p?.v ?? null;
  const ci = v ? T.pci(v)?.[n] : undefined;
  const label = (q: Point) => (q.kind === "baseline" ? "v0" : q.kind === "bridge" ? `${q.name} retest` : q.name);
  useEffect(() => { panel.current?.scrollTo({ top: 0 }); }, [shownKey]);
  const histLabel = n === "-1" ? "full history" : `${n} most recent rating${n === "1" ? "" : "s"}`;

  const scoreTable = (rows: Point[], withChange: boolean) => (
    <table className="viz-table ev-delta">
      <thead><tr><th>Metric</th>{rows.map((q) => <th key={q.key}>{label(q)}{q.v?.champion ? " 🏆" : ""}</th>)}{withChange && <th>Change</th>}</tr></thead>
      <tbody>
        {Object.keys(T.labels).map((mk) => {
          const a = T.get(rows[0])[n]?.[mk] ?? 0, b = rows[1] ? T.get(rows[1])[n]?.[mk] ?? 0 : 0;
          const d = a - b, pct = b ? (100 * d) / b : 0;
          const better = track === "rating" && LOWER_BETTER.has(mk) ? d < 0 : d > 0;
          return (
            <tr key={mk} className={mk === metric ? "sel" : ""}>
              <td>{T.labels[mk]}{track === "rating" && LOWER_BETTER.has(mk) && <span className="muted small"> ↓</span>}
                {mk === metric && <span className="ev-onchart" title="The metric shown on the chart"> on chart</span>}</td>
              {rows.map((q) => <td key={q.key}>{(T.get(q)[n]?.[mk] ?? 0).toFixed(4)}</td>)}
              {withChange && (
                <td className={Math.abs(d) < 5e-5 ? "" : better ? "up" : "down"}>
                  {d >= 0 ? "+" : "−"}{Math.abs(d).toFixed(4)}{b ? <span className="ev-pct"> ({pct >= 0 ? "+" : "−"}{Math.abs(pct).toFixed(1)}%)</span> : null}
                </td>)}
            </tr>
          );
        })}
      </tbody>
    </table>
  );

  return (
    <section className="ev-chart" aria-label="Model versions over time" ref={section} style={{ height }}>
      <div className="ev-controls">
        <label>Metric{" "}
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {Object.entries(T.labels).map(([k, l]) => <option key={k} value={k}>{l}{track === "rating" && LOWER_BETTER.has(k) ? " (lower is better)" : ""}</option>)}
          </select>
        </label>
        <label>History{" "}
          <select value={n} onChange={(e) => setN(e.target.value)}>
            {[...buckets].reverse().map((b) => (
              <option key={b} value={String(b)}>{b < 0 ? "Full history" : `${b} most recent rating${b === 1 ? "" : "s"}`}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="ev-body">
        <div className="ev-left">
          <div className="ev-plot" ref={plot}>
            <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${axisLabel} by model version`}>
              {ticks.map((t) => (
                <g key={t}>
                  {t > ymin && <line className="ev-grid" x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} />}
                  <text className="ev-tick" x={PAD.l - 10} y={y(t) + 4} textAnchor="end">{t.toFixed(decimals)}</text>
                </g>
              ))}
              <line className="ev-axisline" x1={PAD.l} x2={PAD.l} y1={PAD.t - 6} y2={H - PAD.b} />
              <line className="ev-axisline" x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} />
              <text className="ev-axis" x={18} y={yMid} transform={`rotate(-90 18 ${yMid})`} textAnchor="middle">{axisLabel}</text>
              {eras.length > 1 && eras.map((era, ei) => (
                <g key={era.id} className={`ev-era ev-era-${ei}`}>
                  {ei > 0 && <line className="ev-era-divider" x1={eraX[ei].lo} x2={eraX[ei].lo} y1={PAD.t - 22} y2={H - PAD.b} />}
                  <text x={eraX[ei].lo + (ei ? 8 : 10)} y={PAD.t - 12} textAnchor="start">
                    <title>{era.label}</title>{compact ? era.short.replace(/^Trained on /, "") : era.short}
                  </text>
                  <line className="ev-era-rule" x1={eraX[ei].lo + (ei ? 6 : 8)} x2={eraX[ei].hi - 6} y1={PAD.t - 5} y2={PAD.t - 5} />
                </g>
              ))}
              {eras.map((era, ei) => T.refs(era).map((r, k) => {
                const lp = labelPos[`${era.id}:${r.key}`];
                return (
                  <g key={`${era.id}:${r.key}`} className={`ev-ref ev-ref-${k + 1}`}>
                    <line x1={eraX[ei].lo + 4} x2={eraX[ei].hi - 4} y1={y(val(r.metrics))} y2={y(val(r.metrics))} />
                    {lp && <text x={lp.x} y={lp.y} textAnchor={lp.anchor}>{lp.text}</text>}
                  </g>
                );
              }))}
              {eras.map((_, ei) => {
                const idx = eraIdx(ei);
                // 2023 -> the first point: dashed (it's the earlier project, not a version of this one).
                const lead = idx.length > 1 ? `M${x(idx[0])},${y(pv(points[idx[0]]))} L${x(idx[1])},${y(pv(points[idx[1]]))}` : "";
                const rest = idx.slice(1).map((k, j) => `${j ? "L" : "M"}${x(k)},${y(pv(points[k]))}`).join(" ");
                return (
                  <g key={ei}>
                    {lead && <path className="ev-line ev-line-lead" d={lead} />}
                    {idx.length > 2 && <path className="ev-line" d={rest} />}
                  </g>
                );
              })}
              {points.map((q, k) => {
                const cy = y(pv(q));
                return (
                  <g key={q.key} className={`ev-point ev-${q.kind}${q.v?.champion ? " champion" : ""}${shownKey === q.key ? " on" : ""}`}
                     role="button" tabIndex={0}
                     aria-label={`${q.kind === "version" ? `${q.name}: ${q.v!.title}` : q.kind === "baseline" ? "v0: the 2023 Book Recommender" : `${q.name} on the new test`}, ${T.labels[metric]} ${pv(q).toFixed(4)}`}
                     aria-pressed={openId === q.key && pinned}
                     onPointerEnter={(e) => { if (e.pointerType === "mouse") show(q.key); }}
                     onFocus={(e) => { if ((e.currentTarget as Element).matches(":focus-visible")) show(q.key); }}
                     onClick={() => toggle(q.key)}
                     onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(q.key); } }}>
                    <circle className="ev-hit" cx={x(k)} cy={cy} r={18} />
                    {q.kind === "baseline"
                      ? <path className="ev-diamond" d={`M${x(k)},${cy - (compact ? 5.5 : 7.5)} L${x(k) + (compact ? 5.5 : 7.5)},${cy} L${x(k)},${cy + (compact ? 5.5 : 7.5)} L${x(k) - (compact ? 5.5 : 7.5)},${cy} Z`} />
                      : q.v?.champion
                        ? <text className={`ev-trophy${compact ? " small" : ""}`} x={x(k)} y={cy} textAnchor="middle" dominantBaseline="central">🏆</text>
                        : <circle className="ev-dot" cx={x(k)} cy={cy} r={compact ? 4.5 : 6.5} />}
                    {!compact && <text className="ev-val" x={x(k)} y={cy - (q.v?.champion ? 20 : 14)} textAnchor="middle">{pv(q).toFixed(3)}</text>}
                    <text className={`ev-xlabel${compact ? " small" : ""}`} x={x(k)} y={H - PAD.b + 20} textAnchor="middle">
                      {q.name}</text>
                    {!compact && <text className="ev-xsub" x={x(k)} y={H - PAD.b + 38} textAnchor="middle">{q.sub}</text>}
                  </g>
                );
              })}
            </svg>
          </div>
          <p className="muted small ev-hint">Hover or tap a point to see what changed; click to keep it in the panel.
            {eras.length > 1 && " Scores are compared within a dataset only: each has its own test readers."}</p>
        </div>

        {p && (
          <aside className="ev-pop" ref={panel} aria-live="polite" aria-label={v ? `${v.id}: ${v.title}` : label(p)}>
            <div className="ev-pop-head">
              <div>
                <span className="ev-pop-id">
                  {v ? <>{v.champion ? "🏆 " : ""}{v.id}{v.champion ? " · current champion" : ""}</>
                    : p.kind === "baseline" ? "v0 · 2023 baseline" : `${p.name} · on the new test`}
                </span>
                <h3>{v ? v.title : p.kind === "baseline" ? "2023 Book Recommender" : `${p.name} on the new test`}</h3>
                <span className="muted small">{v ? fmtDate(v.date) : eras[p.era].label}{pinned ? " · 📌 pinned" : ""}</span>
              </div>
              {openId && <button type="button" className="ms-close" aria-label="Back to the current champion" title="Back to the current champion"
                                 onClick={() => { setOpenId(null); setPinned(false); }}>×</button>}
            </div>
            {v ? (
              <>
                <p>{v.summary}</p>
                <p className="muted small"><strong>Settings:</strong> {v.settings}
                  {v.rescored && <> <span className="ev-badge">Re-scored on today&apos;s data</span></>}</p>
              </>
            ) : p.kind === "baseline" ? (
              <p>The original 2023 project&apos;s best method on this track ({(track === "ranking" ? p.e!.ranking_method : p.e!.rating_method)?.replace(/^2023: /, "") ?? "similar readers"}),
                re-implemented on this project&apos;s training data and scored on the same test readers: where Shelf Life started from.</p>
            ) : (
              <p>{p.name} as it runs on the earlier data, scored on this dataset&apos;s test readers (its books translated by
                Goodreads id). It&apos;s the fair comparison for the next version: same readers, same hidden books.</p>
            )}
            <h4>{prev ? `Change from ${label(prev)}` : "Scores"} · {histLabel}</h4>
            {scoreTable(prev ? [p, prev] : [p], !!prev)}
            {track === "rating" && <p className="muted small">↓ lower is better.</p>}
            {ci && prev && (
              <p className="muted small">
                Per reader, {T.ci} vs {label(prev)}: {ci.mean_diff >= 0 ? "+" : "−"}{Math.abs(ci.mean_diff).toFixed(4)}{" "}
                (95% CI {ci.ci95[0].toFixed(4)} to {ci.ci95[1].toFixed(4)}); better for{" "}
                {(100 * ci.win).toFixed(1)}% of readers, worse for {(100 * ci.loss).toFixed(1)}%.
              </p>
            )}
            {v && v.decisions.length > 0 && (
              <>
                <h4>Decisions</h4>
                <ul className="ev-list">
                  {v.decisions.map((d) => (
                    <li key={d.id}>
                      <details>
                        <summary><span className="cl-hash">{d.id}</span> {d.title}</summary>
                        {d.sections.map((s) => <div key={s.label} className="ev-sec"><div className="ev-sec-label">{s.label}</div><Markdown text={s.text} /></div>)}
                      </details>
                    </li>
                  ))}
                </ul>
              </>
            )}
            {v && v.commits.length > 0 && (
              <>
                <h4>Commits</h4>
                <ul className="ms-commits">
                  {v.commits.map((c) => <li key={c.short}><a className="cl-hash" href={c.url} target="_blank" rel="noreferrer">{c.short}</a> {c.subject}</li>)}
                </ul>
              </>
            )}
            <button type="button" className="link ms-show" onClick={() => onViewReport(v ? v.report : p.e!.report)}>View report ↓</button>
          </aside>
        )}
      </div>
    </section>
  );
}
