import { useCallback, useEffect, useRef, useState } from "react";
import { api, EMPTY_FILTERS, type Filters, type ReaderBook, type ReadersSort, type RecBook, type RecResponse, type SortKey } from "../api";
import { BookCard, type FilterClick } from "../components/BookCard";
import { ActiveFilters } from "../components/ActiveFilters";
import { FilterBar } from "../components/FilterBar";
import { RecMap } from "../components/RecMap";
import { applyFilterClick } from "../filterClick";
import { useShelf } from "../store";
import type { UrlState } from "../useUrlState";

const PAGE = 40;
const MAP_N = 50;   // how many points fit legibly on the map

const READERS_SORTS: { key: ReadersSort; label: string }[] = [
  { key: "popularity", label: "Popularity" }, { key: "rating", label: "Rating" }, { key: "predicted", label: "Predicted rating" },
];

/** One line under the sort controls saying what the From similar readers list is ordered by. */
function readersHint(sort: ReadersSort, relative: boolean, n: number | undefined): string {
  const who = n ? `the ${n} readers most like you` : "readers most like you";
  if (sort === "popularity") return relative
    ? `Books ${who} read far more often than readers overall: what sets them apart.`
    : `Books ${who} read most.`;
  if (sort === "rating") return relative
    ? `Books ${who} rate furthest above their Goodreads average (at least 5 of them rated it).`
    : `Books ${who} rated highest (at least 5 of them rated it).`;
  return relative
    ? `Books ${who} read that you're predicted to rate furthest above their Goodreads average.`
    : `Books ${who} read, by the rating we predict you'd give them.`;
}

const pct = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 0.1 ? v.toFixed(1) : "<0.1");
const signed = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}★`;

/** The card's note for the current sort: the number the list is ordered by, in words. */
function readersStat(b: ReaderBook, sort: ReadersSort, relative: boolean): string | null {
  if (sort === "rating" && b.readers_avg != null) return relative
    ? `Similar readers: ${signed(b.readers_avg - b.avg_rating)} vs Goodreads`
    : `Similar readers rate it ★${b.readers_avg.toFixed(2)}`;
  if (sort === "predicted" && relative && b.predicted_rating != null)
    return `Predicted ${signed(b.predicted_rating - b.avg_rating)} vs Goodreads`;
  if (b.pct_read == null) return null;
  return sort === "popularity" && relative && b.pct_read_overall != null
    ? `${pct(b.pct_read)}% of similar readers read it vs ${pct(b.pct_read_overall)}% of all readers`
    : `${pct(b.pct_read)}% of similar readers read it`;
}

interface Props {
  tab: UrlState["tab"];
  sort: SortKey;
  setSort: (s: SortKey) => void;
  readersSort: ReadersSort;
  setReadersSort: (s: ReadersSort) => void;
  relative: boolean;
  setRelative: (r: boolean) => void;
  layout: UrlState["layout"];
  setLayout: (l: UrlState["layout"]) => void;
  filters: Filters;
  setTab: (t: UrlState["tab"]) => void;
  setFilters: (f: Filters) => void;
  go: (v: "rate" | "import") => void;
  onOpen: (id: number) => void;
}

export function RecsPage({ tab, sort, setSort, readersSort, setReadersSort, relative, setRelative, layout, setLayout,
                          filters, setTab, setFilters, go, onOpen }: Props) {
  const shelf = useShelf();
  const [data, setData] = useState<RecResponse | null>(null);
  const [limit, setLimit] = useState(PAGE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [genres, setGenres] = useState<{ name: string; books: number }[]>([]);
  const [wide] = useState(() => window.matchMedia("(min-width: 981px)").matches);

  useEffect(() => { api.genres().then(setGenres).catch(() => {}); }, []);

  // Tabs scroll sideways on phones: keep the selected one in view and fade whichever edge has more tabs.
  const tabsRef = useRef<HTMLElement>(null);
  const [fade, setFade] = useState({ left: false, right: false });
  const updateFade = useCallback(() => {
    const el = tabsRef.current;
    if (!el) return;
    const left = el.scrollLeft > 2, right = el.scrollLeft + el.clientWidth < el.scrollWidth - 2;
    setFade((f) => (f.left === left && f.right === right ? f : { left, right }));
  }, []);
  useEffect(() => {
    const el = tabsRef.current;
    const on = el?.querySelector<HTMLElement>("[aria-selected=true]");
    if (el && on) {
      const r = on.getBoundingClientRect(), n = el.getBoundingClientRect();
      el.scrollLeft += r.left - n.left - (n.width - r.width) / 2;
    }
    updateFade();
    window.addEventListener("resize", updateFade);
    return () => window.removeEventListener("resize", updateFade);
  }, [tab, shelf.count, updateFade]);
  useEffect(() => setLimit(PAGE), [filters, sort, readersSort, relative]);

  const [selected, setSelected] = useState<{ book: RecBook; rank: number | undefined; stat: string | null } | null>(null);
  const effLimit = layout === "map" ? MAP_N : limit;
  const body = JSON.stringify({ ...shelf.requestBody(), filters, limit: effLimit, sort,
                                readers_sort: readersSort, readers_relative: relative });
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
          <button type="button" className="primary" onClick={() => go("import")}>Import from Goodreads</button>{" "}
          <button type="button" className="ghost" onClick={() => go("rate")}>Rate books</button>
        </p>
      </div>
    );
  }

  const sr = data?.similar_readers;
  const minReaders = data?.meta.min_ratings_for_readers ?? 5;
  const tabs: { key: UrlState["tab"]; label: string; show: boolean }[] = [
    { key: "for-you", label: "For you", show: true },
    { key: "similar", label: "From similar readers", show: true },
    { key: "to-read", label: `From your to-read (${shelf.toRead.length})`, show: shelf.toRead.length > 0 },
  ];
  const similar = tab === "similar";
  const list = tab === "for-you" ? data?.for_you : similar ? sr?.books : data?.to_read_picks;
  const stat = (b: ReaderBook) => (similar ? readersStat(b, readersSort, relative) : null);

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
      {shelf.demo && (
        <p className="note demo-note">
          You&apos;re viewing a demo: recommendations for a real Goodreads library of 770 books.{" "}
          <button type="button" className="link" onClick={() => go("import")}>Import your own</button> or{" "}
          <button type="button" className="link" onClick={() => go("rate")}>rate books</button> to get yours.
        </p>
      )}

      <div className={`tabs-wrap${fade.left ? " fade-left" : ""}${fade.right ? " fade-right" : ""}`}>
      <nav className="tabs" role="tablist" ref={tabsRef} onScroll={updateFade}>
        {tabs.filter((t) => t.show).map((t) => (
          <button key={t.key} type="button" role="tab" aria-selected={tab === t.key}
                  className={tab === t.key ? "on" : ""} onClick={() => setTab(t.key)}>{t.label}</button>
        ))}
      </nav>
      </div>

      <div className="recs-layout">
        <section className="recs-main">
          <div className="sort-row">
            <span className="flabel">Sort by</span>
            {similar ? (
              <>
                <div className="seg" role="group" aria-label="Sort by">
                  {READERS_SORTS.map((o) => (
                    <button key={o.key} type="button" className={readersSort === o.key ? "on" : ""}
                            onClick={() => setReadersSort(o.key)}>{o.label}</button>
                  ))}
                </div>
              </>
            ) : (
              <div className="seg" role="group" aria-label="Sort by">
                <button type="button" className={sort === "match" ? "on" : ""} onClick={() => setSort("match")}>Best match</button>
                <button type="button" className={sort === "predicted" ? "on" : ""} onClick={() => setSort("predicted")}>
                  Predicted rating
                </button>
              </div>
            )}
            <span className="view-group">
              <span className="flabel view-label">View</span>
              <div className="seg" role="group" aria-label="View">
                <button type="button" className={layout === "list" ? "on" : ""} onClick={() => setLayout("list")}>List</button>
                <button type="button" className={layout === "map" ? "on" : ""} onClick={() => setLayout("map")}>Map</button>
              </div>
            </span>
            {similar && (
              <label className="compare-check"
                     title={readersSort === "popularity" ? "Rank by how much more often similar readers read a book than readers overall"
                       : "Rank by how far each rating sits above the book's Goodreads average"}>
                <input type="checkbox" checked={relative} onChange={(e) => setRelative(e.target.checked)} />
                Compared to all readers
              </label>
            )}
          </div>
          {similar && sr && <p className="muted small sort-hint">{readersHint(readersSort, relative, sr.n_neighbors)}</p>}
          <ActiveFilters filters={filters} defaults={EMPTY_FILTERS} onChange={setFilters} />
          {error && <p className="error">{error}</p>}

          {similar && !sr && data && (
            <div className="notice">
              Rate at least {minReaders} books to unlock readers-like-you lists (you have {data.meta.n_ratings}).{" "}
              <button type="button" className="link" onClick={() => go("rate")}>Rate more</button>
            </div>
          )}

          {layout === "map" && list ? (
            <div className={loading ? "loading-fade" : ""}>
              <RecMap books={list.slice(0, MAP_N)}
                      onSelect={(book, rank) => setSelected({ book: book as RecBook, rank, stat: stat(book) })} />
            </div>
          ) : (
            <ol className={`rank-list${loading ? " loading" : ""}`}>
              {list?.map((b) => (
                <li key={b.id}><BookCard book={b} rank={b.rank ?? undefined} stat={stat(b)} onOpen={onOpen}
                                        onFilter={(f) => setFilters(applyFilterClick(filters, f))} /></li>
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

      {selected && <CardPopover book={selected.book} rank={selected.rank} stat={selected.stat} onClose={() => setSelected(null)} onOpen={onOpen}
                                onFilter={(f) => { setSelected(null); setFilters(applyFilterClick(filters, f)); }} />}
    </div>
  );
}

/** The full recommendation card for a book clicked on the map. */
function CardPopover({ book, rank, stat, onClose, onOpen, onFilter }: {
  book: RecBook; rank: number | undefined; stat: string | null; onClose: () => void; onOpen: (id: number) => void; onFilter: (f: FilterClick) => void;
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
        <BookCard book={book} rank={rank} stat={stat} onOpen={(id) => { onClose(); onOpen(id); }} onFilter={onFilter} />
      </div>
    </div>
  );
}
