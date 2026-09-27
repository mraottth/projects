import { useEffect, useMemo, useState } from "react";
import { api, type Book, type Insights } from "../api";
import { Cover } from "../components/Cover";
import { GenreChip } from "../components/GenreChip";
import { GoodreadsLink } from "../components/GoodreadsLink";
import { GenreTable, InsightTiles } from "../components/Insights";
import { formatCount, Stars } from "../components/Stars";
import { useShelf } from "../store";

type RatingFilter = "all" | 1 | 2 | 3 | 4 | 5 | "unrated";
type YourSort = "yours" | "loved-more" | "loved-less" | "goodreads" | "year" | "title";
const SORTS: { key: YourSort; label: string }[] = [
  { key: "yours", label: "Your rating" },
  { key: "loved-more", label: "You liked more than most" },
  { key: "loved-less", label: "You liked less than most" },
  { key: "goodreads", label: "Goodreads avg" },
  { key: "year", label: "Newest" },
  { key: "title", label: "Title A–Z" },
];

/** Everything on the user's shelf (rated + read without a rating), with stats, filters and inline editing. */
export function YourBooksPage({ go, onOpen }: { go: (v: "rate" | "import") => void; onOpen: (id: number) => void }) {
  const shelf = useShelf();
  const [books, setBooks] = useState<Record<number, Book>>({});
  const [loading, setLoading] = useState(false);
  const [q, setQ] = useState("");
  const [rating, setRating] = useState<RatingFilter>("all");
  const [genre, setGenre] = useState<string | null>(null);
  const [sort, setSort] = useState<YourSort>("yours");

  // Population comparisons (percentiles, harshness, per-genre averages), refreshed as the shelf changes.
  const [insights, setInsights] = useState<Insights | null>(null);
  const insightsBody = JSON.stringify({ ratings: shelf.requestBody().ratings, read: shelf.read });
  useEffect(() => {
    const ctl = new AbortController();
    const t = setTimeout(() => api.insights(JSON.parse(insightsBody), ctl.signal).then(setInsights).catch(() => {}), 250);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [insightsBody]);

  // Fetch full details (averages, counts, tags, links) for any shelf book we don't have yet.
  const ids = useMemo(() => [...new Set([...Object.keys(shelf.ratings).map(Number), ...shelf.read])], [shelf.ratings, shelf.read]);
  useEffect(() => {
    const missing = ids.filter((id) => !(id in books));
    if (!missing.length) return;
    setLoading(true);
    api.booksBatch(missing)
      .then((rows) => setBooks((prev) => ({ ...prev, ...Object.fromEntries(rows.map((b) => [b.id, b])) })))
      .catch(() => {})
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids]);

  const rows = ids.map((id) => ({ book: books[id], mine: shelf.ratings[id]?.rating ?? 0 })).filter((r) => r.book);
  const rated = rows.filter((r) => r.mine > 0);
  const yourAvg = rated.length ? rated.reduce((s, r) => s + r.mine, 0) / rated.length : null;
  const grAvg = rated.length ? rated.reduce((s, r) => s + r.book.avg_rating, 0) / rated.length : null;
  const dist = [5, 4, 3, 2, 1].map((s) => ({ stars: s, n: rated.filter((r) => r.mine === s).length }));
  const maxN = Math.max(1, ...dist.map((d) => d.n));
  const genreCounts = Object.entries(rows.reduce<Record<string, number>>((acc, r) => {
    if (r.book.genre) acc[r.book.genre] = (acc[r.book.genre] ?? 0) + 1;
    return acc;
  }, {})).sort((a, b) => b[1] - a[1]);

  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = rows
    .filter((r) => rating === "all" || (rating === "unrated" ? r.mine === 0 : r.mine === rating))
    .filter((r) => !genre || r.book.genre === genre)
    .filter((r) => terms.every((t) => `${r.book.title} ${r.book.author} ${r.book.tags.join(" ")}`.toLowerCase().includes(t)))
    .sort((a, b) => {
      const diff = (r: typeof a) => (r.mine ? r.mine - r.book.avg_rating : -Infinity);
      switch (sort) {
        case "yours": return b.mine - a.mine || b.book.avg_rating - a.book.avg_rating;
        case "loved-more": return diff(b) - diff(a);
        case "loved-less": return (a.mine ? diff(a) : Infinity) - (b.mine ? diff(b) : Infinity);
        case "goodreads": return b.book.avg_rating - a.book.avg_rating;
        case "year": return (b.book.year ?? 0) - (a.book.year ?? 0);
        default: return a.book.title.localeCompare(b.book.title);
      }
    });

  if (!ids.length) {
    return (
      <div className="empty">
        <h1>Your books</h1>
        <p>Books you rate or import show up here.</p>
        <p>
          <button type="button" className="primary" onClick={() => go("import")}>Import from Goodreads</button>{" "}
          <button type="button" className="ghost" onClick={() => go("rate")}>Rate books</button>
        </p>
      </div>
    );
  }

  return (
    <div className="yours-page">
      <h1>Your books</h1>
      <p className="muted small">Everything you&apos;ve rated or marked as read here, saved in this browser.</p>

      <section className="yours-stats">
        {insights && <InsightTiles data={insights} />}
        <figure className="viz-root dist-chart" aria-label="Your ratings distribution">
          <figcaption className="stat-label">
            Your {rated.length} ratings{yourAvg != null && grAvg != null && <> · avg ★ {yourAvg.toFixed(2)} (Goodreads readers gave
            the same books {grAvg.toFixed(2)})</>} · click a bar to filter
          </figcaption>
          {dist.map((d) => (
            <button key={d.stars} type="button" className={`dist-row${rating === d.stars ? " on" : ""}`}
                    aria-pressed={rating === d.stars} title={`${d.n} book${d.n === 1 ? "" : "s"} rated ${d.stars}★`}
                    onClick={() => setRating(rating === d.stars ? "all" : (d.stars as RatingFilter))}>
              <span className="dist-label">{d.stars}★</span>
              <span className="dist-track"><span className="dist-bar" style={{ width: `${(d.n / maxN) * 100}%` }} /></span>
              <span className="dist-n">{d.n}</span>
            </button>
          ))}
        </figure>
      </section>

      {insights && <GenreTable data={insights} active={genre} onGenre={(g) => setGenre(genre === g ? null : g)} />}

      <section className="yours-controls">
        <input className="search-within" type="search" placeholder="Search your books (title, author, tag)…" value={q}
               onChange={(e) => setQ(e.target.value)} aria-label="Search your books" />
        <div className="seg seg-wrap" role="group" aria-label="Filter by your rating">
          {(["all", 5, 4, 3, 2, 1, "unrated"] as RatingFilter[]).map((r) => (
            <button key={String(r)} type="button" className={rating === r ? "on" : ""} onClick={() => setRating(r)}>
              {r === "all" ? "All" : r === "unrated" ? "Read, not rated" : `${r}★`}
            </button>
          ))}
        </div>
        <div className="genre-chips">
          {genreCounts.slice(0, 12).map(([g, n]) => (
            <GenreChip key={g} genre={g} count={n} active={genre === g} onClick={() => setGenre(genre === g ? null : g)} />
          ))}
        </div>
        <div className="sort-row">
          <span className="flabel">Sort by</span>
          <div className="seg seg-wrap" role="group" aria-label="Sort by">
            {SORTS.map((s) => (
              <button key={s.key} type="button" className={sort === s.key ? "on" : ""} onClick={() => setSort(s.key)}>{s.label}</button>
            ))}
          </div>
        </div>
      </section>

      <p className="muted small">{shown.length} of {rows.length} books{loading ? " · loading details…" : ""}</p>
      <ul className="yours-list">
        {shown.map(({ book, mine }) => {
          const diff = mine ? mine - book.avg_rating : null;
          return (
            <li key={book.id} className="yours-row">
              <Cover book={book} size="sm" onClick={() => onOpen(book.id)} />
              <div className="yours-main">
                <div className="title-row">
                  <button type="button" className="card-title" onClick={() => onOpen(book.id)}>{book.title}</button>
                  <GoodreadsLink url={book.url} title={book.title} />
                </div>
                <div className="card-author">{book.author}{book.year ? <span className="muted"> · {book.year}</span> : null}</div>
                {book.genre && <div className="card-tags"><GenreChip genre={book.genre} /></div>}
              </div>
              <div className="yours-rating">
                <Stars value={mine} size="sm" label={`Your rating for ${book.title}`}
                       onChange={(v) => { if (v) shelf.rate(book, v); else { shelf.unrate(book.id); shelf.markRead(book.id); } }} />
                <span className="muted small">
                  Goodreads ★ {book.avg_rating.toFixed(2)} ({formatCount(book.ratings_count)})
                  {diff != null && <> · <span className="diff">{diff > 0 ? "+" : ""}{diff.toFixed(1)} vs Goodreads</span></>}
                </span>
                <button type="button" className="ghost small" title="Remove from your shelf (e.g. a wrong match)"
                        onClick={() => shelf.remove(book.id)}>Remove</button>
              </div>
            </li>
          );
        })}
      </ul>
      {!loading && shown.length === 0 && <p className="muted">No books match.</p>}
    </div>
  );
}
