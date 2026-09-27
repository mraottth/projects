export interface Book {
  id: number;
  title: string;
  full_title: string;
  author: string;
  author_id: number | null;
  year: number | null;
  avg_rating: number;
  ratings_count: number;
  cover_url: string | null;
  cover_url_small: string | null;
  isbn: string | null;
  url: string;
  series: string | null;
  series_pos: number | null;
  genre: string | null;
  tags: string[];
  description?: string | null;
  predicted_rating?: number;
  readers_avg?: number | null;   // average rating among readers like you
  readers_n?: number;            // how many of them rated it
}

export interface RecBook extends Book {
  score: number;
  source: "similar-books" | "taste-model" | "both" | "popular";
  because: { id: number; title: string }[];
  next_in_series: boolean;
}

export interface ReaderBook extends Book {
  pct_read?: number;
}

export interface BookDetail extends Book {
  description: string | null;
  similar: Book[];
}

export type SortKey = "match" | "predicted";
export type BrowseSort = "popular" | "rating" | "newest" | "oldest" | "title";

export interface Filters {
  genres: string[];
  tags: string[];
  authors_include: number[];
  authors_exclude: number[];
  year_min: number | null;
  year_max: number | null;
  min_avg_rating: number | null;
  min_ratings_count: number | null;
  max_ratings_count: number | null;
  text: string;
  include_children: boolean;
  include_comics: boolean;
  include_series_continuations: boolean;
}

export const EMPTY_FILTERS: Filters = {
  genres: [], tags: [], authors_include: [], authors_exclude: [], year_min: null, year_max: null,
  min_avg_rating: null, min_ratings_count: null, max_ratings_count: null, text: "",
  include_children: false, include_comics: false, include_series_continuations: false,
};

export interface RecResponse {
  for_you: RecBook[];
  to_read_picks: Book[];
  similar_readers: null | {
    n_neighbors: number;
    popular: ReaderBook[];
    top_rated: ReaderBook[];
    genres: { genre: string; you: number; similar_readers: number }[];
  };
  meta: { n_ratings: number; sort: SortKey; your_avg: number | null; books_avg: number | null; alpha: number; total_candidates: number; min_ratings_for_readers: number; ms: number };
}

export interface ImportResult {
  rated: (Book & { rating: number; match: string })[];
  read_unrated: number[];
  to_read: number[];
  unmatched: { title: string; author: string; year: string | null }[];
  stats: { rows: number; matched: number; match_rate: number; unmatched_rated_or_read: number };
}

export interface Insights {
  books: { n: number; percentile: number; median: number; p90: number; n_readers: number };
  harshness: null | { bias: number; harsher_than: number; median_bias: number; n_readers: number };
  genres: { genre: string; books: number; rated: number; your_avg: number | null; goodreads_avg: number | null;
            readers_genre_avg: number | null }[];
  genre_mix: { genre: string; you: number; similar_readers: number }[];
}

export interface Author { id: number; name: string; ratings_count: number; n_books: number }

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail ?? msg; } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  search: (q: string, signal?: AbortSignal) =>
    fetch(`/api/search?q=${encodeURIComponent(q)}&limit=8`, { signal }).then(json<Book[]>),
  authors: (q: string, signal?: AbortSignal) =>
    fetch(`/api/authors?q=${encodeURIComponent(q)}&limit=8`, { signal }).then(json<Author[]>),
  authorNames: (ids: number[]) =>
    fetch(`/api/authors/names?${ids.map((i) => `ids=${i}`).join("&")}`).then(json<Record<string, string>>),
  booksBatch: (ids: number[]) =>
    fetch("/api/books/batch", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids }),
    }).then(json<Book[]>),
  insights: (body: object, signal?: AbortSignal) =>
    fetch("/api/insights", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    }).then(json<Insights>),
  tags: () => fetch("/api/tags").then(json<{ name: string; books: number }[]>),
  genres: () => fetch("/api/genres").then(json<{ name: string; books: number }[]>),
  starter: () => fetch("/api/starter").then(json<Book[]>),
  book: (id: number) => fetch(`/api/books/${id}`).then(json<BookDetail>),
  bookPersonal: (id: number, body: object, signal?: AbortSignal) =>
    fetch(`/api/books/${id}/personal`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    }).then(json<{ predicted_rating: number | null; readers_avg: number | null; readers_n: number | null }>),
  importCsv: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return fetch("/api/import", { method: "POST", body }).then(json<ImportResult>);
  },
  browse: (body: object, signal?: AbortSignal) =>
    fetch("/api/browse", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    }).then(json<{ books: Book[]; total: number; ms: number }>),
  recommend: (body: object, signal?: AbortSignal) =>
    fetch("/api/recommend", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
    }).then(json<RecResponse>),
};
