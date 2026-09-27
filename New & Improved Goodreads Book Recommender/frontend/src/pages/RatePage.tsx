import { useEffect, useRef, useState } from "react";
import { api, type Book, type RecBook } from "../api";
import { Cover } from "../components/Cover";
import { AvgRating, Stars } from "../components/Stars";
import { useShelf } from "../store";

const GOAL = 10;

export function RatePage({ go, onOpen }: { go: (v: "recs") => void; onOpen: (id: number) => void }) {
  const shelf = useShelf();
  const [q, setQ] = useState("");
  const [results, setResults] = useState<Book[]>([]);
  const [starter, setStarter] = useState<Book[]>([]);
  const [preview, setPreview] = useState<RecBook[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { api.starter().then(setStarter).catch(() => {}); inputRef.current?.focus(); }, []);

  // Typeahead (debounced 150ms, previous request aborted).
  useEffect(() => {
    if (!q.trim()) { setResults([]); return; }
    const ctl = new AbortController();
    const t = setTimeout(() => api.search(q, ctl.signal).then(setResults).catch(() => {}), 150);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [q]);

  // Live preview of recommendations, refreshed after rating changes (debounced 400ms).
  const body = JSON.stringify(shelf.requestBody());
  useEffect(() => {
    if (!shelf.count) { setPreview([]); return; }
    const ctl = new AbortController();
    const t = setTimeout(() => api.recommend({ ...JSON.parse(body), limit: 12 }, ctl.signal)
      .then((r) => setPreview(r.for_you)).catch(() => {}), 400);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [body, shelf.count]);

  const rated = Object.values(shelf.ratings);
  const unrated = starter.filter((b) => !(b.id in shelf.ratings));

  return (
    <div className="rate-page">
      <div className="rate-main">
        <h1>Rate books you've read</h1>
        <p className="muted">Search for books you know and give them stars. {Math.max(0, GOAL - shelf.count) > 0
          ? `Rate ${GOAL - shelf.count} more for sharper recommendations.` : "Nice — your recommendations are well calibrated."}</p>
        <div className="progress"><span style={{ width: `${Math.min(100, (shelf.count / GOAL) * 100)}%` }} /></div>

        <div className="search-box">
          <input ref={inputRef} type="search" placeholder="Search by title or author…" value={q}
                 onChange={(e) => setQ(e.target.value)} aria-label="Search books" />
        </div>
        {results.length > 0 && (
          <ul className="search-results">
            {results.map((b) => (
              <li key={b.id}>
                <Cover book={b} size="sm" onClick={() => onOpen(b.id)} />
                <div className="sr-info">
                  <button type="button" className="card-title" onClick={() => onOpen(b.id)}>{b.title}</button>
                  <div className="muted small">{b.author}{b.year ? ` · ${b.year}` : ""}{b.series ? ` · ${b.series} #${b.series_pos}` : ""}</div>
                  <AvgRating value={b.avg_rating} count={b.ratings_count} />
                </div>
                <Stars value={shelf.ratings[b.id]?.rating ?? 0} label={`Rate ${b.title}`}
                       onChange={(v) => (v ? shelf.rate(b, v) : shelf.unrate(b.id))} />
              </li>
            ))}
          </ul>
        )}
        {q && results.length === 0 && <p className="muted">No matches in the catalog (books through 2017 with 20+ ratings).</p>}

        {!q && (
          <>
            <h2>Or rate some popular books</h2>
            <div className="starter-grid">
              {unrated.slice(0, 30).map((b) => (
                <div key={b.id} className="starter-item">
                  <Cover book={b} onClick={() => onOpen(b.id)} />
                  <span className="strip-title">{b.title}</span>
                  <Stars value={0} size="sm" label={`Rate ${b.title}`} onChange={(v) => v && shelf.rate(b, v)} />
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <aside className="rate-side">
        <div className="side-head">
          <h2>Your shelf <span className="muted">({shelf.count})</span></h2>
          {shelf.count > 0 && <button type="button" className="primary" onClick={() => go("recs")}>See recommendations →</button>}
        </div>
        {rated.length === 0 && <p className="muted">Books you rate will appear here.</p>}
        <ul className="shelf-list">
          {rated.slice().reverse().map(({ book, rating }) => (
            <li key={book.id}>
              <Cover book={book} size="sm" onClick={() => onOpen(book.id)} />
              <div className="sr-info">
                <span className="shelf-title">{book.title}</span>
                <Stars value={rating} size="sm" onChange={(v) => (v ? shelf.rate(book as Book, v) : shelf.unrate(book.id))} />
              </div>
              <button type="button" className="ghost small" aria-label={`Remove ${book.title}`} onClick={() => shelf.unrate(book.id)}>×</button>
            </li>
          ))}
        </ul>
        {preview.length > 0 && (
          <div className="preview">
            <h3>Top picks so far</h3>
            <div className="preview-grid">
              {preview.map((b) => (
                <div key={b.id} title={`${b.title} — ${b.author}`}>
                  <Cover book={b} size="sm" onClick={() => onOpen(b.id)} />
                </div>
              ))}
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}
