import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Markdown } from "./Markdown";

/**
 * Model versions (x) against an evaluation metric (y) for the Evaluation page. A details panel beside the chart
 * (below it on phones) shows the current champion by default; hovering a version's point with a mouse shows it
 * (and it stays when the mouse moves away), and clicking (or Enter/Space) pins it until clicked again, × or Escape. The panel never covers the chart.
 * Labelled dashed lines are baselines for comparison on the same test readers. `track` picks the evaluation track:
 * ranking (NDCG / Precision / Recall; baselines the 2023 Book Recommender and popular books) or rating prediction
 * (MAE and friends, several lower-is-better; baselines the book average and the 2023 Book Recommender).
 * The chart and panel share one height that fits the viewport; the SVG is drawn at its measured size.
 */

export interface VersionCommit { short: string; subject: string; url: string }
export interface VersionDecision { id: string; title: string; sections: { label: string; text: string }[] }
export interface Version {
  id: string; date: string; title: string; summary: string; settings: string; rescored: boolean; report: string;
  commits: VersionCommit[]; decisions: VersionDecision[]; champion: boolean; n_users: number;
  metrics: Record<string, Record<string, number>>;            // n ("1".."25", "-1" = full history) -> metric -> value
  rmetrics: Record<string, Record<string, number>> | null;     // rating track, same shape
  ci_vs_previous?: Record<string, PairedCI>;                    // per-user NDCG@10 vs the previous version
  rci_vs_previous?: Record<string, PairedCI>;                   // per-user MAE vs the previous version
}
interface PairedCI { mean_diff: number; ci95: [number, number]; win: number; loss: number }
export type Track = "ranking" | "rating";
export interface Reference { key: string; label: string; metrics: Record<string, Record<string, number>> }

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
  ranking: { labels: METRIC_LABEL, first: "ndcg@10", ci: "NDCG@10", get: (v: Version) => v.metrics, pci: (v: Version) => v.ci_vs_previous },
  rating: { labels: RATING_LABEL, first: "mae", ci: "MAE", get: (v: Version) => v.rmetrics ?? {}, pci: (v: Version) => v.rci_vs_previous },
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

export function VersionChart({ versions, references, buckets, onViewReport, track = "ranking" }: {
  versions: Version[]; references: Reference[]; buckets: number[]; onViewReport: (reportId: string) => void; track?: Track;
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

  const { w: W, h: H } = size;
  const val = (m: Record<string, Record<string, number>>) => m[n]?.[metric] ?? 0;
  const ys = [...versions.map((v) => val(T.get(v))), ...references.map((r) => val(r.metrics))];
  const ticks = track === "rating" ? bandTicks(Math.min(...ys), Math.max(...ys)) : niceTicks(Math.max(...ys, 1e-6));
  const ymin = ticks[0], ymax = ticks[ticks.length - 1];
  const decimals = track === "rating" ? Math.max(2, -Math.floor(Math.log10(ticks[1] - ticks[0]) + 1e-9)) : ymax < 0.1 ? 3 : 2;
  const x0 = PAD.l + 28, x1 = W - PAD.r - 28;
  const x = (i: number) => (versions.length === 1 ? (x0 + x1) / 2 : x0 + (i / (versions.length - 1)) * (x1 - x0));
  const y = (v: number) => H - PAD.b - ((v - ymin) / (ymax - ymin)) * (H - PAD.t - PAD.b);
  const path = versions.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(val(T.get(v)))}`).join(" ");
  const yMid = (PAD.t + H - PAD.b) / 2;
  // Baseline labels: just above or below their line, at the left or right end, wherever they don't cover a
  // version's point, its value label or another baseline's label.
  type Box = { x0: number; x1: number; y0: number; y1: number };
  const hit = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
  const taken: Box[] = versions.map((vv, k) => ({ x0: x(k) - 30, x1: x(k) + 30, y0: y(val(T.get(vv))) - 34, y1: y(val(T.get(vv))) + 8 }));
  const labelPos: Record<string, { x: number; y: number; anchor: "start" | "middle" | "end"; text: string }> = {};
  references.forEach((r) => {
    const full = `Baseline: ${r.label} (${val(r.metrics).toFixed(3)})`;
    const text = W >= 520 ? full : `${r.label.replace(/ performance$/, "")} (${val(r.metrics).toFixed(3)})`;   // short on phones
    const ly = y(val(r.metrics)), w = 6.1 * text.length;
    type Opt = { x: number; y: number; anchor: "start" | "middle" | "end" };
    const options: Opt[] = [
      { x: PAD.l + 8, y: ly - 6, anchor: "start" }, { x: PAD.l + 8, y: ly + 27, anchor: "start" },
      { x: W - PAD.r - 8, y: ly - 6, anchor: "end" }, { x: W - PAD.r - 8, y: ly + 27, anchor: "end" },
      ...versions.slice(1).flatMap((_, k): Opt[] => [                  // centred between two versions
        { x: (x(k) + x(k + 1)) / 2, y: ly - 6, anchor: "middle" }, { x: (x(k) + x(k + 1)) / 2, y: ly + 27, anchor: "middle" }]),
    ];
    const left = (o: Opt) => (o.anchor === "start" ? o.x : o.anchor === "end" ? o.x - w : o.x - w / 2);
    const box = (o: Opt): Box => ({ x0: left(o), x1: left(o) + w, y0: o.y - 11, y1: o.y + 3 });
    const inside = (o: Opt) => left(o) >= PAD.l + 2 && left(o) + w <= W - PAD.r + 2;
    const pick = options.find((o) => inside(o) && !taken.some((t) => hit(t, box(o)))) ?? options[0];
    taken.push(box(pick));
    labelPos[r.key] = { ...pick, text };
  });

  const show = (id: string) => { if (!pinned) setOpenId(id); };
  const toggle = (id: string) => {
    if (pinned && openId === id) { setOpenId(null); setPinned(false); }
    else { setOpenId(id); setPinned(true); }
  };
  // The panel shows the pinned or last-hovered version (it stays after the mouse leaves), else the current
  // champion (else the latest version).
  const defaultId = (versions.find((vv) => vv.champion) ?? versions[versions.length - 1])?.id ?? null;
  const shownId = openId ?? defaultId;
  const i = versions.findIndex((v) => v.id === shownId);
  const v = i >= 0 ? versions[i] : null;
  const prev = i > 0 ? versions[i - 1] : null;
  const ci = v ? T.pci(v)?.[n] : undefined;
  useEffect(() => { panel.current?.scrollTo({ top: 0 }); }, [shownId]);

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
              {references.map((r, k) => (
                <g key={r.key} className={`ev-ref ev-ref-${k}`}>
                  <line x1={PAD.l} x2={W - PAD.r} y1={y(val(r.metrics))} y2={y(val(r.metrics))} />
                  <text x={labelPos[r.key].x} y={labelPos[r.key].y} textAnchor={labelPos[r.key].anchor}>{labelPos[r.key].text}</text>
                </g>
              ))}
              <path className="ev-line" d={path} />
              {versions.map((vv, k) => (
                <g key={vv.id} className={`ev-point${vv.champion ? " champion" : ""}${shownId === vv.id ? " on" : ""}`}
                   role="button" tabIndex={0} aria-label={`${vv.id}: ${vv.title}, ${T.labels[metric]} ${val(T.get(vv)).toFixed(4)}`}
                   aria-pressed={openId === vv.id && pinned}
                   onPointerEnter={(e) => { if (e.pointerType === "mouse") show(vv.id); }}
                   onFocus={(e) => { if ((e.currentTarget as Element).matches(":focus-visible")) show(vv.id); }}
                   onClick={() => toggle(vv.id)}
                   onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(vv.id); } }}>
                  <circle className="ev-hit" cx={x(k)} cy={y(val(T.get(vv)))} r={18} />
                  {vv.champion
                    ? <text className="ev-trophy" x={x(k)} y={y(val(T.get(vv)))} textAnchor="middle" dominantBaseline="central">🏆</text>
                    : <circle className="ev-dot" cx={x(k)} cy={y(val(T.get(vv)))} r={6.5} />}
                  <text className="ev-val" x={x(k)} y={y(val(T.get(vv))) - (vv.champion ? 20 : 14)} textAnchor="middle">{val(T.get(vv)).toFixed(3)}</text>
                  <text className="ev-xlabel" x={x(k)} y={H - PAD.b + 22} textAnchor="middle">{vv.champion ? "🏆 " : ""}{vv.id}</text>
                  <text className="ev-xsub" x={x(k)} y={H - PAD.b + 38} textAnchor="middle">{fmtDate(vv.date).replace(/, \d{4}$/, "")}</text>
                </g>
              ))}
            </svg>
          </div>
          <p className="muted small ev-hint">Hover or tap a version to see what changed; click to keep it in the panel.</p>
        </div>

        {v && (
          <aside className="ev-pop" ref={panel} aria-live="polite" aria-label={`${v.id}: ${v.title}`}>
            <div className="ev-pop-head">
              <div>
                <span className="ev-pop-id">{v.champion ? "🏆 " : ""}{v.id}{v.champion ? " · current champion" : ""}</span>
                <h3>{v.title}</h3>
                <span className="muted small">{fmtDate(v.date)}{pinned ? " · 📌 pinned" : ""}</span>
              </div>
              {openId && <button type="button" className="ms-close" aria-label="Back to the current champion" title="Back to the current champion"
                                 onClick={() => { setOpenId(null); setPinned(false); }}>×</button>}
            </div>
            <p>{v.summary}</p>
            <p className="muted small"><strong>Settings:</strong> {v.settings}
              {v.rescored && <> <span className="ev-badge">Re-scored on today&apos;s data</span></>}</p>
            <h4>{prev ? `Change from ${prev.id}` : "Scores"} · {n === "-1" ? "full history" : `${n} most recent rating${n === "1" ? "" : "s"}`}</h4>
            <table className="viz-table ev-delta">
              <thead><tr><th>Metric</th><th>{v.id}{v.champion ? " 🏆" : ""}</th>{prev && <><th>{prev.id}{prev.champion ? " 🏆" : ""}</th><th>Change</th></>}</tr></thead>
              <tbody>
                {Object.keys(T.labels).map((mk) => {
                  const a = T.get(v)[n]?.[mk] ?? 0, b = prev ? T.get(prev)[n]?.[mk] ?? 0 : 0;
                  const d = a - b, pct = b ? (100 * d) / b : 0;
                  const better = track === "rating" && LOWER_BETTER.has(mk) ? d < 0 : d > 0;
                  return (
                    <tr key={mk} className={mk === metric ? "sel" : ""}>
                      <td>{T.labels[mk]}{track === "rating" && LOWER_BETTER.has(mk) && <span className="muted small"> ↓</span>}
                        {mk === metric && <span className="ev-onchart" title="The metric shown on the chart"> on chart</span>}</td>
                      <td>{a.toFixed(4)}</td>
                      {prev && <><td>{b.toFixed(4)}</td>
                        <td className={Math.abs(d) < 5e-5 ? "" : better ? "up" : "down"}>
                          {d >= 0 ? "+" : "−"}{Math.abs(d).toFixed(4)}{b ? <span className="ev-pct"> ({pct >= 0 ? "+" : "−"}{Math.abs(pct).toFixed(1)}%)</span> : null}
                        </td></>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {track === "rating" && <p className="muted small">↓ lower is better.</p>}
            {ci && prev && (
              <p className="muted small">
                Per reader, {T.ci} vs {prev.id}: {ci.mean_diff >= 0 ? "+" : "−"}{Math.abs(ci.mean_diff).toFixed(4)}{" "}
                (95% CI {ci.ci95[0].toFixed(4)} to {ci.ci95[1].toFixed(4)}); better for{" "}
                {(100 * ci.win).toFixed(1)}% of readers, worse for {(100 * ci.loss).toFixed(1)}%.
              </p>
            )}
            {v.decisions.length > 0 && (
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
            {v.commits.length > 0 && (
              <>
                <h4>Commits</h4>
                <ul className="ms-commits">
                  {v.commits.map((c) => <li key={c.short}><a className="cl-hash" href={c.url} target="_blank" rel="noreferrer">{c.short}</a> {c.subject}</li>)}
                </ul>
              </>
            )}
            <button type="button" className="link ms-show" onClick={() => onViewReport(v.report)}>View report ↓</button>
          </aside>
        )}
      </div>
    </section>
  );
}
