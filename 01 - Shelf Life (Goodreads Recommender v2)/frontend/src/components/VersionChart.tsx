import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Markdown } from "./Markdown";

/**
 * Model versions (x) against an evaluation metric (y) for the Evaluation page. A details panel beside the chart
 * (below it on phones) shows the current champion by default; hovering a version's point with a mouse previews
 * it, and clicking (or Enter/Space) pins it until clicked again, × or Escape. The panel never covers the chart.
 * Labelled dashed lines are baselines for comparison (the 2023 Book Recommender, popular books) on the same test readers.
 * The chart and panel share one height that fits the viewport; the SVG is drawn at its measured size.
 */

export interface VersionCommit { short: string; subject: string; url: string }
export interface VersionDecision { id: string; title: string; sections: { label: string; text: string }[] }
export interface Version {
  id: string; date: string; title: string; summary: string; settings: string; rescored: boolean; report: string;
  commits: VersionCommit[]; decisions: VersionDecision[]; champion: boolean; n_users: number;
  metrics: Record<string, Record<string, number>>;            // n ("1".."25", "-1" = full history) -> metric -> value
  ci_vs_previous?: Record<string, { mean_diff: number; ci95: [number, number]; win: number; loss: number }>;
}
export interface Reference { key: string; label: string; metrics: Record<string, Record<string, number>> }

export const METRIC_LABEL: Record<string, string> = {
  "ndcg@10": "NDCG@10", "recall@10": "Recall@10", "precision@10": "Precision@10",
  "ndcg@20": "NDCG@20", "recall@20": "Recall@20", "precision@20": "Precision@20",
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

export function VersionChart({ versions, references, buckets, onViewReport }: {
  versions: Version[]; references: Reference[]; buckets: number[]; onViewReport: (reportId: string) => void;
}) {
  const [metric, setMetric] = useState("ndcg@10");
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
  const ys = [...versions.map((v) => val(v.metrics)), ...references.map((r) => val(r.metrics))];
  const ticks = niceTicks(Math.max(...ys, 1e-6));
  const ymax = ticks[ticks.length - 1];
  const x0 = PAD.l + 28, x1 = W - PAD.r - 28;
  const x = (i: number) => (versions.length === 1 ? (x0 + x1) / 2 : x0 + (i / (versions.length - 1)) * (x1 - x0));
  const y = (v: number) => H - PAD.b - (v / ymax) * (H - PAD.t - PAD.b);
  const path = versions.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(val(v.metrics))}`).join(" ");
  const yMid = (PAD.t + H - PAD.b) / 2;

  const show = (id: string) => { if (!pinned) setOpenId(id); };
  const toggle = (id: string) => {
    if (pinned && openId === id) { setOpenId(null); setPinned(false); }
    else { setOpenId(id); setPinned(true); }
  };
  // The panel shows the hovered or pinned version, else the current champion (else the latest version).
  const defaultId = (versions.find((vv) => vv.champion) ?? versions[versions.length - 1])?.id ?? null;
  const shownId = openId ?? defaultId;
  const i = versions.findIndex((v) => v.id === shownId);
  const v = i >= 0 ? versions[i] : null;
  const prev = i > 0 ? versions[i - 1] : null;
  useEffect(() => { panel.current?.scrollTo({ top: 0 }); }, [shownId]);

  return (
    <section className="ev-chart" aria-label="Model versions over time" ref={section} style={{ height }}>
      <div className="ev-controls">
        <label>Metric{" "}
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {Object.entries(METRIC_LABEL).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
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
          <div className="ev-plot" ref={plot}
               onPointerLeave={(e) => { if (e.pointerType === "mouse" && !pinned) setOpenId(null); }}>
            <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${METRIC_LABEL[metric]} by model version`}>
              {ticks.map((t) => (
                <g key={t}>
                  {t > 0 && <line className="ev-grid" x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} />}
                  <text className="ev-tick" x={PAD.l - 10} y={y(t) + 4} textAnchor="end">{t.toFixed(ymax < 0.1 ? 3 : 2)}</text>
                </g>
              ))}
              <line className="ev-axisline" x1={PAD.l} x2={PAD.l} y1={PAD.t - 6} y2={H - PAD.b} />
              <line className="ev-axisline" x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} />
              <text className="ev-axis" x={18} y={yMid} transform={`rotate(-90 18 ${yMid})`} textAnchor="middle">{METRIC_LABEL[metric]}</text>
              {references.map((r, k) => (
                <g key={r.key} className={`ev-ref ev-ref-${k}`}>
                  <line x1={PAD.l} x2={W - PAD.r} y1={y(val(r.metrics))} y2={y(val(r.metrics))} />
                  <text x={PAD.l + 8} y={y(val(r.metrics)) - 6}>Baseline: {r.label} ({val(r.metrics).toFixed(3)})</text>
                </g>
              ))}
              <path className="ev-line" d={path} />
              {versions.map((vv, k) => (
                <g key={vv.id} className={`ev-point${vv.champion ? " champion" : ""}${shownId === vv.id ? " on" : ""}`}
                   role="button" tabIndex={0} aria-label={`${vv.id}: ${vv.title}, ${METRIC_LABEL[metric]} ${val(vv.metrics).toFixed(4)}`}
                   aria-pressed={openId === vv.id && pinned}
                   onPointerEnter={(e) => { if (e.pointerType === "mouse") show(vv.id); }}
                   onFocus={(e) => { if ((e.currentTarget as Element).matches(":focus-visible")) show(vv.id); }}
                   onClick={() => toggle(vv.id)}
                   onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(vv.id); } }}>
                  <circle className="ev-hit" cx={x(k)} cy={y(val(vv.metrics))} r={18} />
                  <circle className="ev-dot" cx={x(k)} cy={y(val(vv.metrics))} r={vv.champion ? 8 : 6.5} />
                  <text className="ev-val" x={x(k)} y={y(val(vv.metrics)) - 14} textAnchor="middle">{val(vv.metrics).toFixed(3)}</text>
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
              {pinned && <button type="button" className="ms-close" aria-label="Unpin" onClick={() => { setOpenId(null); setPinned(false); }}>×</button>}
            </div>
            <p>{v.summary}</p>
            <p className="muted small"><strong>Settings:</strong> {v.settings}
              {v.rescored && <> <span className="ev-badge">Re-scored on today&apos;s data</span></>}</p>
            <h4>{prev ? `Change from ${prev.id}` : "Scores"} · {n === "-1" ? "full history" : `${n} most recent rating${n === "1" ? "" : "s"}`}</h4>
            <table className="viz-table ev-delta">
              <thead><tr><th>Metric</th><th>{v.id}{v.champion ? " 🏆" : ""}</th>{prev && <><th>{prev.id}{prev.champion ? " 🏆" : ""}</th><th>Change</th></>}</tr></thead>
              <tbody>
                {Object.keys(METRIC_LABEL).map((mk) => {
                  const a = v.metrics[n]?.[mk] ?? 0, b = prev?.metrics[n]?.[mk] ?? 0;
                  const d = a - b, pct = b ? (100 * d) / b : 0;
                  return (
                    <tr key={mk} className={mk === metric ? "sel" : ""}>
                      <td>{METRIC_LABEL[mk]}{mk === metric && <span className="ev-onchart" title="The metric shown on the chart"> on chart</span>}</td>
                      <td>{a.toFixed(4)}</td>
                      {prev && <><td>{b.toFixed(4)}</td>
                        <td className={Math.abs(d) < 5e-5 ? "" : d > 0 ? "up" : "down"}>
                          {d >= 0 ? "+" : "−"}{Math.abs(d).toFixed(4)}{b ? <span className="ev-pct"> ({pct >= 0 ? "+" : "−"}{Math.abs(pct).toFixed(1)}%)</span> : null}
                        </td></>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {v.ci_vs_previous?.[n] && prev && (
              <p className="muted small">
                Per reader, NDCG@10 vs {prev.id}: {v.ci_vs_previous[n].mean_diff >= 0 ? "+" : "−"}{Math.abs(v.ci_vs_previous[n].mean_diff).toFixed(4)}{" "}
                (95% CI {v.ci_vs_previous[n].ci95[0].toFixed(4)} to {v.ci_vs_previous[n].ci95[1].toFixed(4)}); better for{" "}
                {(100 * v.ci_vs_previous[n].win).toFixed(1)}% of readers, worse for {(100 * v.ci_vs_previous[n].loss).toFixed(1)}%.
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
