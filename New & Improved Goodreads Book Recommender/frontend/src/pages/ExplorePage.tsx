import { useEffect, useState } from "react";
import { api, type Book, type BrowseSort, type Filters } from "../api";
import { BookCard } from "../components/BookCard";
import { FilterBar } from "../components/FilterBar";
import { useShelf } from "../store";
import { EXPLORE_DEFAULTS } from "../useUrlState";

const PAGE = 40;
const SORTS: { key: BrowseSort; label: string }[] = [
  { key: "popular", label: "Most rated" },
  { key: "rating", label: "Highest rated" },
  { key: "newest", label: "Newest" },
  { key: "oldest", label: "Oldest" },
  { key: "title", label: "Title A–Z" },
];

interface Props {
  filters: Filters;
  setFilters: (f: Filters) => void;
  sort: BrowseSort;
  setSort: (s: BrowseSort) => void;
  onOpen: (id: number) => void;
}

/** Browse the whole catalog with the same filters as recommendations (no personalization in the order). */
export function ExplorePage({ filters, setFilters, sort, setSort, onOpen }: Props) {
  const shelf = useShelf();
  const [books, setBooks] = useState<Book[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [genres, setGenres] = useState<{ name: string; books: number }[]>([]);
  const [wide] = useState(() => window.matchMedia("(min-width: 981px)").matches);

  useEffect(() => { api.genres().then(setGenres).catch(() => {}); }, []);

  // Ratings are sent only so each card can show a predicted rating; they don't change the order.
  const ratings = JSON.stringify(shelf.requestBody().ratings);
  const query = JSON.stringify({ filters, sort, ratings: JSON.parse(ratings) });

  const load = (offset: number, signal?: AbortSignal) => {
    setLoading(true);
    return api.browse({ ...JSON.parse(query), limit: PAGE, offset }, signal)
      .then((r) => {
        setBooks((prev) => (offset === 0 ? r.books : [...prev, ...r.books]));
        setTotal(r.total);
        setError(null);
      })
      .catch((e) => { if (e.name !== "AbortError") setError(e.message); })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    const ctl = new AbortController();
    const t = setTimeout(() => load(0, ctl.signal), 200);
    return () => { clearTimeout(t); ctl.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  return (
    <div className="recs-page">
      <div className="recs-head">
        <div>
          <h1>Explore books</h1>
          <p className="muted small">
            {total != null ? `${total.toLocaleString()} books match` : "Loading…"} · the catalog covers books through 2017
            with at least 20 ratings in the dataset
          </p>
        </div>
      </div>

      <div className="recs-layout">
        <section className="recs-main">
          <div className="sort-row">
            <span className="flabel">Sort by</span>
            <div className="seg seg-wrap" role="group" aria-label="Sort by">
              {SORTS.map((s) => (
                <button key={s.key} type="button" className={sort === s.key ? "on" : ""} onClick={() => setSort(s.key)}>
                  {s.label}
                </button>
              ))}
            </div>
          </div>
          {sort === "rating" && (
            <p className="muted small tab-blurb">
              Highest rated is weighted by number of ratings, so a 4.9 from a few dozen readers doesn't outrank a 4.6
              from hundreds of thousands.
            </p>
          )}
          {error && <p className="error">{error}</p>}
          <ol className={`rank-list${loading && books.length === 0 ? " loading" : ""}`}>
            {books.map((b, i) => <li key={b.id}><BookCard book={b} rank={i + 1} onOpen={onOpen} /></li>)}
          </ol>
          {total === 0 && !loading && <p className="muted">No books match these filters — try loosening them.</p>}
          {loading && books.length === 0 && <div className="spinner" />}
          {total != null && books.length < total && (
            <p className="center">
              <button type="button" className="ghost" disabled={loading} onClick={() => load(books.length)}>
                {loading ? "Loading…" : `Show more (${(total - books.length).toLocaleString()} left)`}
              </button>
            </p>
          )}
        </section>

        <aside className="recs-side">
          <details className="filters-details" open={wide}>
            <summary>Filters</summary>
            <FilterBar filters={filters} onChange={setFilters} genres={genres} defaults={EXPLORE_DEFAULTS} />
          </details>
        </aside>
      </div>
    </div>
  );
}
