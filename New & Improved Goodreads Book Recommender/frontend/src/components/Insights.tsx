import type React from "react";
import type { Insights as InsightsData } from "../api";
import { GenreChip } from "./GenreChip";

const pct = (x: number) => `${Math.round(x * 100)}%`;
const signed = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(2)}`;

/** Where a value sits in the reader population: a 0–100 track with quartile ticks and one marker. */
function PercentileTrack({ value, left, right, label }: { value: number; left: string; right: string; label: string }) {
  return (
    <div className="ptrack" role="img" aria-label={label}>
      <div className="ptrack-bar">
        {[25, 50, 75].map((t) => <span key={t} className="ptrack-tick" style={{ left: `${t}%` }} />)}
        <span className="ptrack-dot" style={{ left: `${Math.min(99, Math.max(1, value * 100))}%` }} />
      </div>
      <div className="ptrack-ends"><span>{left}</span><span>{right}</span></div>
    </div>
  );
}

export function InsightTiles({ data }: { data: InsightsData }) {
  const b = data.books;
  const h = data.harshness;
  return (
    <>
      <div className="stat-tile insight">
        <span className="tile-title">{b.n.toLocaleString()} books read</span>
        <span className="insight-line">More than <strong>{pct(b.percentile)}</strong> of the {b.n_readers.toLocaleString()} readers in the dataset</span>
        <PercentileTrack value={b.percentile} left="fewer books" right="more books" label={`${pct(b.percentile)} percentile`} />
      </div>

      <div className="stat-tile insight">
        {h ? (
          <>
            <span className="tile-title one-line">
              {h.harsher_than >= 0.5 ? `Harsher than ${pct(h.harsher_than)} of readers` : `Kinder than ${pct(1 - h.harsher_than)} of readers`}
            </span>
            <span className="insight-line">You rate books <strong>{signed(h.bias)}★</strong> vs. their average.</span>
            <PercentileTrack value={h.harsher_than} left="more generous" right="harsher" label={`harsher than ${pct(h.harsher_than)} of readers`} />
          </>
        ) : (
          <>
            <span className="tile-title">How harsh a critic are you?</span>
            <span className="insight-line muted">Rate at least 5 books to see this.</span>
          </>
        )}
      </div>
    </>
  );
}

export const GENRE_ROWS_COLLAPSED = 8;

export function GenreTable({ data, onGenre, active, all, setAll, tableRef }: {
  data: InsightsData; onGenre: (g: string) => void; active: string | null;
  all: boolean; setAll: (v: boolean) => void; tableRef?: React.Ref<HTMLDivElement>;
}) {
  const rows = all ? data.genres : data.genres.slice(0, GENRE_ROWS_COLLAPSED);
  const maxBooks = Math.max(1, ...data.genres.map((g) => g.books));
  if (!data.genres.length) return null;
  return (
    <section className="genre-insights">
      <div className="table-scroll" ref={tableRef}>
        <table className="viz-table genre-table">
          <thead>
            <tr>
              <th>Genre</th><th>Books</th><th className="num">Your avg</th><th className="num">Goodreads<br /><span className="th-sub">same books</span></th>
              <th className="num">Difference</th><th className="num">Genre avg<br /><span className="th-sub">all readers</span></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((g) => {
              const diff = g.your_avg != null && g.goodreads_avg != null ? g.your_avg - g.goodreads_avg : null;
              return (
                <tr key={g.genre} className={active === g.genre ? "on" : ""}>
                  <td><GenreChip genre={g.genre} active={active === g.genre} onClick={() => onGenre(g.genre)} /></td>
                  <td>
                    <span className="count-cell">
                      <span className="count-track"><span className="count-bar" style={{ width: `${(g.books / maxBooks) * 100}%` }} /></span>
                      <span className="count-n">{g.books}{g.rated < g.books && <span className="muted"> ({g.rated} rated)</span>}</span>
                    </span>
                  </td>
                  <td className="num">{g.your_avg != null ? `★ ${g.your_avg.toFixed(2)}` : "—"}</td>
                  <td className="num">{g.goodreads_avg != null ? `★ ${g.goodreads_avg.toFixed(2)}` : "—"}</td>
                  <td className="num">{diff != null ? <span className="diff">{signed(diff)}</span> : "—"}</td>
                  <td className="num">{g.readers_genre_avg != null ? `★ ${g.readers_genre_avg.toFixed(2)}` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {data.genres.length > GENRE_ROWS_COLLAPSED && (
        <button type="button" className="link small" onClick={() => setAll(!all)}>
          {all ? "Show fewer genres" : `Show all ${data.genres.length} genres`}
        </button>
      )}
    </section>
  );
}
