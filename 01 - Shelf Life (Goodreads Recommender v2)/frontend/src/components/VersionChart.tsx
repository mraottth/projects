import { useEffect, useRef, useState } from "react";

/**
 * Model versions (x) against an evaluation metric (y) for the Evaluation page. A details panel beside the chart
 * (below it on phones) shows the current champion by default; hovering a version's point with a mouse previews
 * it, and clicking (or Enter/Space) pins it until clicked again, × or Escape. The panel never covers the chart.
 * Dashed lines mark reference baselines (popular books, the best 2023 method) on the same test readers.
 */

export interface VersionCommit { short: string; subject: string; url: string }
export interface Version {
  id: string; date: string; title: string; summary: string; settings: string; rescored: boolean; report: string;
  commits: VersionCommit[]; decisions: string[]; champion: boolean; n_users: number;
  metrics: Record<string, Record<string, number>>;            // n ("1".."25", "-1" = full history) -> metric -> value
  ci_vs_previous?: Record<string, { mean_diff: number; ci95: [number, number]; win: number; loss: number }>;
}
export interface Reference { key: string; label: string; metrics: Record<string, Record<string, number>> }

export const METRIC_LABEL: Record<string, string> = {
  "ndcg@10": "NDCG@10", "recall@10": "Recall@10", "precision@10": "Precision@10",
  "ndcg@20": "NDCG@20", "recall@20": "Recall@20", "precision@20": "Precision@20",
};
const fmtDate = (d: string) => new Date(`${d}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
const W = 560, H = 330, PAD = { l: 52, r: 18, t: 34, b: 56 };

function niceTicks(max: number): number[] {
  const raw = max / 5;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? raw;
  return Array.from({ length: Math.floor(max / step) + 1 }, (_, i) => +(i * step).toFixed(6));
}

export function VersionChart({ versions, references, buckets, onViewReport }: {
  versions: Version[]; references: Reference[]; buckets: number[]; onViewReport: (reportId: string) => void;
}) {
  const [metric, setMetric] = useState("ndcg@10");
  const [n, setN] = useState("-1");
  const [openId, setOpenId] = useState<string | null>(null);
  const [pinned, setPinned] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!openId) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { setOpenId(null); setPinned(false); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openId]);

  const val = (m: Record<string, Record<string, number>>) => m[n]?.[metric] ?? 0;
  const ys = [...versions.map((v) => val(v.metrics)), ...references.map((r) => val(r.metrics))];
  const ticks = niceTicks(Math.max(...ys, 1e-6) * 1.12);
  const ymax = ticks[ticks.length - 1];
  const x = (i: number) => PAD.l + (versions.length === 1 ? 0.5 : i / (versions.length - 1)) * (W - PAD.l - PAD.r - 40) + 20;
  const y = (v: number) => H - PAD.b - (v / ymax) * (H - PAD.t - PAD.b);
  const path = versions.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(val(v.metrics))}`).join(" ");

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

  return (
    <section className="ev-chart" aria-label="Model versions over time">
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
        <div className="ev-plot" ref={wrap}
             onPointerLeave={(e) => { if (e.pointerType === "mouse" && !pinned) setOpenId(null); }}>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${METRIC_LABEL[metric]} by model version`}>
            {ticks.map((t) => (
              <g key={t}>
                <line className="ev-grid" x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} />
                <text className="ev-tick" x={PAD.l - 8} y={y(t) + 4} textAnchor="end">{t.toFixed(t < 0.1 ? 3 : 2)}</text>
              </g>
            ))}
            <text className="ev-axis" x={14} y={(H - PAD.b + PAD.t) / 2} transform={`rotate(-90 14 ${(H - PAD.b + PAD.t) / 2})`}
                  textAnchor="middle">{METRIC_LABEL[metric]}</text>
            {references.map((r, k) => (
              <g key={r.key} className={`ev-ref ev-ref-${k}`}>
                <line x1={PAD.l} x2={W - PAD.r} y1={y(val(r.metrics))} y2={y(val(r.metrics))} />
                <text x={PAD.l + 6} y={y(val(r.metrics)) - 5}>{r.label}</text>
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
                <text className="ev-xlabel" x={x(k)} y={H - PAD.b + 22} textAnchor="middle">{vv.id}{vv.champion ? " ★" : ""}</text>
                <text className="ev-xsub" x={x(k)} y={H - PAD.b + 38} textAnchor="middle">{fmtDate(vv.date).replace(/, \d{4}$/, "")}</text>
              </g>
            ))}
          </svg>
          <p className="muted small ev-hint">Hover or tap a version to see what changed; click to keep it in the panel.</p>
        </div>

          {v && (
            <aside className="ev-pop" aria-live="polite" aria-label={`${v.id}: ${v.title}`}>
              <div className="ev-pop-head">
                <div>
                  <span className="ev-pop-id">{v.id}{v.champion ? " · current champion" : ""}</span>
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
                <thead><tr><th>Metric</th><th>{v.id}</th>{prev && <><th>{prev.id}</th><th>Change</th></>}</tr></thead>
                <tbody>
                  {Object.keys(METRIC_LABEL).map((mk) => {
                    const a = v.metrics[n]?.[mk] ?? 0, b = prev?.metrics[n]?.[mk] ?? 0;
                    const d = a - b, pct = b ? (100 * d) / b : 0;
                    return (
                      <tr key={mk} className={mk === metric ? "sel" : ""}>
                        <td>{METRIC_LABEL[mk]}</td><td>{a.toFixed(4)}</td>
                        {prev && <><td>{b.toFixed(4)}</td>
                          <td className={Math.abs(d) < 5e-5 ? "" : d > 0 ? "up" : "down"}>{d >= 0 ? "+" : "−"}{Math.abs(d).toFixed(4)}{b ? ` (${pct >= 0 ? "+" : "−"}${Math.abs(pct).toFixed(1)}%)` : ""}</td></>}
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
              {v.commits.length > 0 && (
                <>
                  <h4>Commits</h4>
                  <ul className="ms-commits">
                    {v.commits.map((c) => <li key={c.short}><a className="cl-hash" href={c.url} target="_blank" rel="noreferrer">{c.short}</a> {c.subject}</li>)}
                  </ul>
                </>
              )}
              {v.decisions.length > 0 && <p className="muted small">Decisions: {v.decisions.join(", ")} in DECISIONS.md</p>}
              <button type="button" className="link ms-show" onClick={() => onViewReport(v.report)}>View report ↓</button>
            </aside>
          )}
      </div>
      <p className="muted small ev-note">
        Scores on {versions[versions.length - 1]?.n_users.toLocaleString()} test readers the models never saw: each reader&apos;s
        most recent 30% of books are hidden and the model ranks books from what they read before. Versions marked
        re-scored were run with their settings on today&apos;s data, so their numbers differ slightly from what they scored at the time.
      </p>
    </section>
  );
}
