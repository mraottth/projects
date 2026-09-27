import { useState } from "react";
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
        <span className="stat-label">Books read</span>
        <span className="stat-value">{b.n.toLocaleString()}</span>
        <span className="insight-line">
          More than <strong>{pct(b.percentile)}</strong> of the {b.n_readers.toLocaleString()} readers in the dataset
        </span>
        <PercentileTrack value={b.percentile} left="fewer books" right="more books" label={`${pct(b.percentile)} percentile`} />
        <span className="muted tiny">
          Typical reader: {b.median} books; top 10%: {Math.round(b.p90)}+. Dataset readers are counted by the books they
          reviewed through 2017, which undercounts their reading, so this is a rough comparison.
        </span>
      </div>

      <div className="stat-tile insight">
        <span className="stat-label">How harsh a critic are you?</span>
        {h ? (
          <>
            <span className="stat-value">{h.harsher_than >= 0.5 ? `Harsher than ${pct(h.harsher_than)}` : `Kinder than ${pct(1 - h.harsher_than)}`}</span>
            <span className="insight-line">
              of {h.n_readers.toLocaleString()} readers. You rate books <strong>{signed(h.bias)}★</strong> vs. their average,
              compared with {signed(h.median_bias)}★ for the typical reader.
            </span>
            <PercentileTrack value={h.harsher_than} left="more generous" right="harsher" label={`harsher than ${pct(h.harsher_than)} of readers`} />
            <span className="muted tiny">Compares each of your ratings with the book&apos;s average rating in the dataset.</span>
          </>
        ) : (
          <span className="insight-line muted">Rate at least 5 books to see this.</span>
        )}
      </div>
    </>
  );
}

export function GenreTable({ data, onGenre, active }: { data: InsightsData; onGenre: (g: string) => void; active: string | null }) {
  const [all, setAll] = useState(false);
  const rows = all ? data.genres : data.genres.slice(0, 8);
  const maxBooks = Math.max(1, ...data.genres.map((g) => g.books));
  if (!data.genres.length) return null;
  return (
    <section className="genre-insights">
      <h2>By genre</h2>
      <p className="muted small">
        Your average vs. what all Goodreads readers gave <em>the same books</em>, and the typical rating across the whole
        genre. Click a genre to filter your list.
      </p>
      <div className="table-scroll">
        <table className="viz-table genre-table">
          <thead>
            <tr>
              <th>Genre</th><th>Books</th><th className="num">Your avg</th><th className="num">Goodreads avg<br /><span className="th-sub">same books</span></th>
              <th className="num">You vs. Goodreads</th><th className="num">Typical in genre</th>
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
      {data.genres.length > 8 && (
        <button type="button" className="link small" onClick={() => setAll((a) => !a)}>
          {all ? "Show fewer genres" : `Show all ${data.genres.length} genres`}
        </button>
      )}
    </section>
  );
}
