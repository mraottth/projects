import { useState } from "react";

interface Row { genre: string; you: number; similar_readers: number }

/**
 * Dumbbell (dot) plot: share of your liked books vs. readers like you, per parent genre.
 * Two series = categorical slots 1-2 (blue / orange, validated); legend + table view + hover tooltip.
 * One shared 0..max axis; thin connecting line in a neutral tone; 10px markers with a 2px surface ring.
 */
export function GenreChart({ rows }: { rows: Row[] }) {
  const [asTable, setAsTable] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(0.05, ...rows.flatMap((r) => [r.you, r.similar_readers]));
  const axisMax = Math.ceil(max * 10) / 10;
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  const x = (v: number) => `${(v / axisMax) * 100}%`;
  const ticks = Array.from({ length: Math.round(axisMax * 10) + 1 }, (_, i) => i / 10).filter((_, i, a) => a.length <= 6 || i % 2 === 0);

  return (
    <figure className="viz-root genre-chart">
      <figcaption>
        <strong>Your genre mix vs. readers like you</strong>
        <span className="muted small"> — share of books you liked (4–5★) and books your nearest readers read</span>
      </figcaption>
      <div className="viz-legend">
        <span><i className="legend-dot" style={{ background: "var(--series-1)" }} /> You</span>
        <span><i className="legend-dot" style={{ background: "var(--series-2)" }} /> Readers like you</span>
        <button type="button" className="link small" onClick={() => setAsTable((t) => !t)}>
          {asTable ? "Show chart" : "Show table"}
        </button>
      </div>
      {asTable ? (
        <table className="viz-table">
          <thead><tr><th>Genre</th><th>You</th><th>Readers like you</th></tr></thead>
          <tbody>{rows.map((r) => <tr key={r.genre}><td>{r.genre}</td><td>{pct(r.you)}</td><td>{pct(r.similar_readers)}</td></tr>)}</tbody>
        </table>
      ) : (
        <div className="dumbbell">
          {rows.map((r, i) => {
            const lo = Math.min(r.you, r.similar_readers), hi = Math.max(r.you, r.similar_readers);
            return (
              <div key={r.genre} className={`db-row${hover === i ? " hover" : ""}`}
                   onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                <span className="db-label">{r.genre}</span>
                <div className="db-track">
                  <span className="db-line" style={{ left: x(lo), width: `calc(${x(hi)} - ${x(lo)})` }} />
                  <span className="db-dot" style={{ left: x(r.similar_readers), background: "var(--series-2)" }} />
                  <span className="db-dot" style={{ left: x(r.you), background: "var(--series-1)" }} />
                  {hover === i && (
                    <span className="db-tip" style={{ left: x(hi) }}>
                      <strong>{r.genre}</strong><br />You {pct(r.you)} · Readers like you {pct(r.similar_readers)}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
          <div className="db-row db-axis">
            <span className="db-label" />
            <div className="db-track">
              {ticks.map((t) => <span key={t} className="db-tick" style={{ left: x(t) }}>{pct(t)}</span>)}
            </div>
          </div>
        </div>
      )}
    </figure>
  );
}
