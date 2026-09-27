import { useEffect, useState } from "react";
import { api, type Filters, type RecResponse } from "../api";
import { BookCard } from "../components/BookCard";
import { FilterBar } from "../components/FilterBar";
import { GenreChart } from "../components/GenreChart";
import { useShelf } from "../store";
import type { UrlState } from "../useUrlState";

const PAGE = 40;

interface Props {
  tab: UrlState["tab"];
  filters: Filters;
  setTab: (t: UrlState["tab"]) => void;
  setFilters: (f: Filters) => void;
  go: (v: "rate" | "import") => void;
  onOpen: (id: number) => void;
}

export function RecsPage({ tab, filters, setTab, setFilters, go, onOpen }: Props) {
  const shelf = useShelf();
  const [data, setData] = useState<RecResponse | null>(null);
  const [limit, setLimit] = useState(PAGE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [genres, setGenres] = useState<{ name: string; books: number }[]>([]);

  useEffect(() => { api.genres().then(setGenres).catch(() => {}); }, []);
  useEffect(() => setLimit(PAGE), [filters]);

  const body = JSON.stringify({ ...shelf.requestBody(), filters, limit });
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
  const blurb = {
    "for-you": "Predicted from your ratings: books that readers who liked what you liked also rated highly.",
    "popular": "What your nearest readers (by taste) have read most, adjusted so bestsellers don't crowd out everything else.",
    "top-rated": "The highest-rated books among your nearest readers, requiring several of them to have rated it.",
    "to-read": "Your Goodreads to-read shelf, ordered by how much the model thinks you'll like each book.",
  }[tab];

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

      <FilterBar filters={filters} onChange={setFilters} genres={genres} />

      <p className="muted small tab-blurb">{blurb}</p>
      {error && <p className="error">{error}</p>}

      {(tab === "popular" || tab === "top-rated") && !sr && data && (
        <div className="notice">
          Rate at least {minReaders} books to unlock readers-like-you lists (you have {data.meta.n_ratings}).{" "}
          <button type="button" className="link" onClick={() => go("rate")}>Rate more</button>
        </div>
      )}

      {tab === "for-you" && sr && sr.genres.length > 0 && <GenreChart rows={sr.genres.slice(0, 6)} />}

      <div className={`grid${loading ? " loading" : ""}`}>
        {list?.map((b, i) => <BookCard key={b.id} book={b} rank={tab === "for-you" ? i + 1 : undefined} onOpen={onOpen} />)}
      </div>
      {list && list.length === 0 && !loading && <p className="muted">Nothing matches these filters — try loosening them.</p>}
      {!data && loading && <div className="spinner" />}
      {tab === "for-you" && data && data.for_you.length >= limit && limit < 200 && (
        <p className="center"><button type="button" className="ghost" onClick={() => setLimit((l) => l + PAGE)}>Show more</button></p>
      )}
    </div>
  );
}
