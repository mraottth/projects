import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Book, ReaderBook, RecBook } from "../api";
import { Cover } from "./Cover";
import { formatCount } from "./Stars";

type AnyBook = Book & Partial<RecBook> & Partial<ReaderBook>;

/**
 * Scatter of the top recommendations: x = Goodreads ratings (log scale), y = Goodreads average.
 * Dots are colored by predicted rating on an ordinal one-hue ramp (dataviz reference blue, steps
 * 250-650); the top five carry their rank as a direct label. Hover/focus grows a dot into its cover
 * with an info card; click opens the full recommendation card. The List view is the table fallback.
 */
const RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"];
const M = { top: 18, right: 22, bottom: 52, left: 62 };
const HEIGHT = 560;

export function RecMap({ books, onSelect }: { books: AnyBook[]; onSelect: (b: AnyBook, rank: number) => void }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);
  const [active, setActive] = useState<number | null>(null);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const geo = useMemo(() => {
    const counts = books.map((b) => Math.log10(Math.max(b.ratings_count, 1)));
    const avgs = books.map((b) => b.avg_rating);
    const x0 = Math.floor(Math.min(...counts) * 4) / 4 - 0.1, x1 = Math.ceil(Math.max(...counts) * 4) / 4 + 0.1;
    const y0 = Math.floor(Math.min(...avgs) * 10) / 10 - 0.05, y1 = Math.ceil(Math.max(...avgs) * 10) / 10 + 0.05;
    const iw = width - M.left - M.right, ih = HEIGHT - M.top - M.bottom;
    const sx = (v: number) => M.left + ((v - x0) / (x1 - x0)) * iw;
    const sy = (v: number) => M.top + (1 - (v - y0) / (y1 - y0)) * ih;
    const xTicks: number[] = [];
    for (let e = Math.floor(x0); e <= x1; e++) for (const k of [1, 3]) { const v = Math.log10(k) + e; if (v >= x0 && v <= x1) xTicks.push(v); }
    const yStep = y1 - y0 > 1 ? 0.2 : 0.1;
    const yTicks: number[] = [];
    for (let v = Math.ceil(y0 / yStep) * yStep; v <= y1 + 1e-9; v += yStep) yTicks.push(Math.round(v * 100) / 100);
    // Ordinal bins of predicted rating (quintiles of the shown set) for color.
    const preds = books.map((b) => b.predicted_rating ?? 0).sort((a, b) => a - b);
    const cuts = [0.2, 0.4, 0.6, 0.8].map((q) => preds[Math.floor(q * (preds.length - 1))]);
    const bin = (p: number) => cuts.filter((c) => p > c).length;
    const ranges = RAMP.map((_, i) => [i === 0 ? preds[0] : cuts[i - 1], i === 4 ? preds[preds.length - 1] : cuts[i]]);
    return { sx, sy, xTicks, yTicks, bin, ranges, iw, ih };
  }, [books, width]);

  if (books.length === 0) return <p className="muted">Nothing to plot for these filters.</p>;
  const hasPred = books.some((b) => b.predicted_rating != null);

  return (
    <figure className="viz-root rec-map">
      <figcaption className="map-caption">
        <span><strong>Your top {books.length}</strong> by Goodreads popularity and rating. Hover a dot for the book,
          click for its full card. Upper right is loved <em>and</em> widely read; upper left is hidden gems.</span>
        {hasPred && (
          <span className="map-legend" aria-label="Color shows predicted rating">
            <span className="muted small">Predicted for you</span>
            {RAMP.map((c, i) => (
              <span key={c} className="ramp-step" title={`${geo.ranges[i][0]?.toFixed(1)}–${geo.ranges[i][1]?.toFixed(1)}`}>
                <i style={{ background: c }} />{i === 0 ? geo.ranges[0][0]?.toFixed(1) : i === 4 ? geo.ranges[4][1]?.toFixed(1) : ""}
              </span>
            ))}
          </span>
        )}
      </figcaption>

      <div className="map-wrap" ref={wrap} style={{ height: HEIGHT }} onMouseLeave={() => setActive(null)}>
        <svg width={width} height={HEIGHT} aria-hidden="true">
          {geo.yTicks.map((v) => (
            <g key={`y${v}`}>
              <line x1={M.left} x2={width - M.right} y1={geo.sy(v)} y2={geo.sy(v)} className="map-grid" />
              <text x={M.left - 8} y={geo.sy(v) + 4} textAnchor="end" className="map-tick">{v.toFixed(1)}</text>
            </g>
          ))}
          {geo.xTicks.map((v) => (
            <g key={`x${v}`}>
              <line x1={geo.sx(v)} x2={geo.sx(v)} y1={M.top} y2={HEIGHT - M.bottom} className="map-grid" />
              <text x={geo.sx(v)} y={HEIGHT - M.bottom + 18} textAnchor="middle" className="map-tick">
                {formatCount(Math.round(10 ** v))}
              </text>
            </g>
          ))}
          <line x1={M.left} x2={width - M.right} y1={HEIGHT - M.bottom} y2={HEIGHT - M.bottom} className="map-axis" />
          <text x={M.left + geo.iw / 2} y={HEIGHT - 10} textAnchor="middle" className="map-axis-title">
            Number of Goodreads ratings (log scale)
          </text>
          <text transform={`translate(16 ${M.top + geo.ih / 2}) rotate(-90)`} textAnchor="middle" className="map-axis-title">
            Goodreads average rating
          </text>
        </svg>

        {books.map((b, i) => {
          const x = geo.sx(Math.log10(Math.max(b.ratings_count, 1)));
          const y = geo.sy(b.avg_rating);
          const color = b.predicted_rating != null ? RAMP[geo.bin(b.predicted_rating)] : RAMP[2];
          const on = active === i;
          const flip = x > width * 0.62;          // info card goes left of the cover near the right edge
          const low = y > HEIGHT * 0.62;          // ...and upward near the bottom
          return (
            <button key={b.id} type="button" className={`map-pt${on ? " on" : ""}`} style={{ left: x, top: y, zIndex: on ? 1000 : books.length - i }}
                    aria-label={`#${i + 1} ${b.title} by ${b.author}: ${b.avg_rating.toFixed(2)} average from ${b.ratings_count.toLocaleString()} ratings${b.predicted_rating != null ? `, predicted ${b.predicted_rating.toFixed(1)} for you` : ""}`}
                    onMouseEnter={() => setActive(i)} onFocus={() => setActive(i)} onBlur={() => setActive(null)}
                    onClick={() => onSelect(b, i + 1)}>
              <span className="map-dot" style={{ background: color }} />
              {i < 5 && !on && <span className="map-rank">{i + 1}</span>}
              {on && (
                <span className={`map-pop${flip ? " flip" : ""}${low ? " low" : ""}`}>
                  <span className="map-cover"><Cover book={b} size="sm" /></span>
                  <span className="map-info">
                    <span className="map-info-rank">#{i + 1}</span>
                    <strong className="map-info-title">{b.title}</strong>
                    <span className="map-info-meta">{b.author}{b.year ? ` · ${b.year}` : ""}</span>
                    <span className="map-info-row"><b>★ {b.avg_rating.toFixed(2)}</b> Goodreads avg · {formatCount(b.ratings_count)} ratings</span>
                    {b.predicted_rating != null && (
                      <span className="map-info-row"><b>{b.predicted_rating.toFixed(1)}</b> predicted for you</span>
                    )}
                    {b.genre && <span className="map-info-meta">{b.genre}</span>}
                  </span>
                </span>
              )}
            </button>
          );
        })}
      </div>
    </figure>
  );
}
