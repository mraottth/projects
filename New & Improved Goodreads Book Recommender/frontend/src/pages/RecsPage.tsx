import { useEffect, useState } from "react";
import { api, EMPTY_FILTERS, type Filters, type RecBook, type RecResponse, type SortKey } from "../api";
import { BookCard, type FilterClick } from "../components/BookCard";
import { ActiveFilters } from "../components/ActiveFilters";
import { FilterBar } from "../components/FilterBar";
import { RecMap } from "../components/RecMap";
import { applyFilterClick } from "../filterClick";
import { useShelf } from "../store";
import type { UrlState } from "../useUrlState";

const PAGE = 40;
const MAP_N = 50;   // how many points fit legibly on the map

interface Props {
  tab: UrlState["tab"];
  sort: SortKey;
  setSort: (s: SortKey) => void;
  layout: UrlState["layout"];
  setLayout: (l: UrlState["layout"]) => void;
  filters: Filters;
  setTab: (t: UrlState["tab"]) => void;
  setFilters: (f: Filters) => void;
  go: (v: "rate" | "import") => void;
  onOpen: (id: number) => void;
}

export function RecsPage({ tab, sort, setSort, layout, setLayout, filters, setTab, setFilters, go, onOpen }: Props) {
  const shelf = useShelf();
  const [data, setData] = useState<RecResponse | null>(null);
  const [limit, setLimit] = useState(PAGE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [genres, setGenres] = useState<{ name: string; books: number }[]>([]);
  const [wide] = useState(() => window.matchMedia("(min-width: 981px)").matches);

  useEffect(() => { api.genres().then(setGenres).catch(() => {}); }, []);
  useEffect(() => setLimit(PAGE), [filters, sort]);

  const [selected, setSelected] = useState<{ book: RecBook; rank: number } | null>(null);
  const effLimit = layout === "map" ? MAP_N : limit;
  const body = JSON.stringify({ ...shelf.requestBody(), filters, limit: effLimit, sort });
  useEffect(() => {
    if (!shelf.count) return;
    const ctl = new AbortController();
    setLoading(true);
    const t = setTimeout(() => api.recommend(JSON.parse(body), ctl.signal)
      .then((r) => { setData(r); setError(null); })
      .catch((e) => { if (e.name !== "AbortError") setError(e.message); })
      .finally(() => setLoading(false)), 250);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [body, shelf.count]);

  if (!shelf.count) {
    return (
      <div className="empty">
        <h1>No ratings yet</h1>
        <p>Rate a few books or import your Goodreads library to get recommendations.</p>
        <p>
          <button type="button" className="primary" onClick={() => go("rate")}>Rate books</button>{" "}
          <button type="button" className="ghost" onClick={() => go("import")}>Import from Goodreads</button>
        </p>
      </div>
    );
  }

  const sr = data?.similar_readers;
  const minReaders = data?.meta.min_ratings_for_readers ?? 5;
  const tabs: { key: UrlState["tab"]; label: string; show: boolean }[] = [
    { key: "for-you", label: "For you", show: true },
    { key: "popular", label: "Popular with readers like you", show: true },
    { key: "top-rated", label: "Top rated by readers like you", show: true },
    { key: "to-read", label: `From your to-read (${shelf.toRead.length})`, show: shelf.toRead.length > 0 },
  ];
  const list =
    tab === "for-you" ? data?.for_you
      : tab === "popular" ? sr?.popular
        : tab === "top-rated" ? sr?.top_rated
          : data?.to_read_picks;

  return (
    <div className="recs-page">
      <div className="recs-head">
        <div>
          <h1>Recommendations</h1>
          <p className="muted small">
            Based on {shelf.count} rated book{shelf.count === 1 ? "" : "s"}
            {data && ` · ${data.meta.total_candidates.toLocaleString()} books match your filters · ${data.meta.ms} ms`}
          </p>
        </div>
        <button type="button" className="ghost" onClick={() => go("rate")}>+ Rate more books</button>
      </div>

      <nav className="tabs" role="tablist">
        {tabs.filter((t) => t.show).map((t) => (
          <button key={t.key} type="button" role="tab" aria-selected={tab === t.key}
                  className={tab === t.key ? "on" : ""} onClick={() => setTab(t.key)}>{t.label}</button>
        ))}
      </nav>

      <div className="recs-layout">
        <section className="recs-main">
          <div className="sort-row">
            <span className="flabel">Sort by</span>
            <div className="seg" role="group" aria-label="Sort by">
              <button type="button" className={sort === "match" ? "on" : ""} onClick={() => setSort("match")}>Best match</button>
              <button type="button" className={sort === "predicted" ? "on" : ""} onClick={() => setSort("predicted")}>
                Predicted rating
              </button>
            </div>
            <span className="flabel view-label">View</span>
            <div className="seg" role="group" aria-label="View">
              <button type="button" className={layout === "list" ? "on" : ""} onClick={() => setLayout("list")}>List</button>
              <button type="button" className={layout === "map" ? "on" : ""} onClick={() => setLayout("map")}>Map</button>
            </div>
          </div>
          <ActiveFilters filters={filters} defaults={EMPTY_FILTERS} onChange={setFilters} />
          {error && <p className="error">{error}</p>}

          {(tab === "popular" || tab === "top-rated") && !sr && data && (
            <div className="notice">
              Rate at least {minReaders} books to unlock readers-like-you lists (you have {data.meta.n_ratings}).{" "}
              <button type="button" className="link" onClick={() => go("rate")}>Rate more</button>
            </div>
          )}

          {layout === "map" && list ? (
            <div className={loading ? "loading-fade" : ""}>
              <RecMap books={list.slice(0, MAP_N)} onSelect={(book, rank) => setSelected({ book: book as RecBook, rank })} />
            </div>
          ) : (
            <ol className={`rank-list${loading ? " loading" : ""}`}>
              {list?.map((b, i) => (
                <li key={b.id}><BookCard book={b} rank={i + 1} onOpen={onOpen} onFilter={(f) => setFilters(applyFilterClick(filters, f))} /></li>
              ))}
            </ol>
          )}
          {list && list.length === 0 && !loading && (
            <p className="muted">
              {filters.text
                ? <>No recommendations match &ldquo;{filters.text}&rdquo;. This box searches within your recommendations (title,
                  author, genre and tags), so books you&apos;ve already read are hidden, and the catalog ends in 2017. To look up
                  any book, use <strong>Explore</strong>.</>
                : "Nothing matches these filters. Try loosening them."}
            </p>
          )}
          {!data && loading && <div className="spinner" />}
          {layout === "list" && tab === "for-you" && data && data.for_you.length >= limit && limit < 200 && (
            <p className="center"><button type="button" className="ghost" onClick={() => setLimit((l) => l + PAGE)}>Show more</button></p>
          )}
        </section>

        <aside className="recs-side">
          {/* Filters start collapsed on narrow screens so results stay above the fold. */}
          <details className="filters-details" open={wide}>
            <summary>Filters</summary>
            <FilterBar filters={filters} onChange={setFilters} genres={genres} />
          </details>
        </aside>
      </div>

      {selected && <CardPopover book={selected.book} rank={selected.rank} onClose={() => setSelected(null)} onOpen={onOpen}
                                onFilter={(f) => { setSelected(null); setFilters(applyFilterClick(filters, f)); }} />}
    </div>
  );
}

/** The full recommendation card for a book clicked on the map. */
function CardPopover({ book, rank, onClose, onOpen, onFilter }: {
  book: RecBook; rank: number; onClose: () => void; onOpen: (id: number) => void; onFilter: (f: FilterClick) => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="card-popover" role="dialog" aria-modal="true" aria-label={book.title} onClick={(e) => e.stopPropagation()}>
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close">×</button>
        <BookCard book={book} rank={rank} onOpen={(id) => { onClose(); onOpen(id); }} onFilter={onFilter} />
      </div>
    </div>
  );
}
