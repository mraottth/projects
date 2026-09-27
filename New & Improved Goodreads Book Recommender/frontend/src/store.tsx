import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Book, ImportResult, OutsideBook } from "./api";

/** The user's shelf lives only in this browser (localStorage); the API is stateless. */
export type ShelfBook = Pick<Book, "id" | "title" | "author" | "cover_url" | "cover_url_small" | "isbn" | "genre" | "year">;

/**
 * Where a shelf entry came from. Re-importing replaces everything from the previous import but keeps
 * what the user did on this site. Entries saved before this field existed have no source and are
 * treated as imported (they can't be told apart).
 */
type Source = "import" | "manual";

interface ShelfState {
  ratings: Record<number, { rating: number; book: ShelfBook; source?: Source }>;
  read: number[];          // read without a rating (from the import and/or marked here)
  manualRead: number[];    // subset of `read` the user marked on this site (kept on re-import)
  toRead: number[];
  dismissed: number[];
  reviews: Record<number, string>;   // from the import; read by the Assistant tab
  outside: OutsideBook[];            // export rows not in the catalog (mostly post-2017); read by the Assistant tab
}

const KEY = "goodrec.shelf.v1";
const EMPTY: ShelfState = { ratings: {}, read: [], manualRead: [], toRead: [], dismissed: [], reviews: {}, outside: [] };
const MAX_REVIEW_CHARS = 800_000;   // keep localStorage well under its ~5 MB quota

function capReviews(reviews: Record<string, string>): Record<number, string> {
  const out: Record<number, string> = {};
  let total = 0;
  for (const [id, text] of Object.entries(reviews)) {
    if ((total += text.length) > MAX_REVIEW_CHARS) break;
    out[Number(id)] = text;
  }
  return out;
}

function load(): ShelfState {
  try {
    return { ...EMPTY, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return EMPTY;
  }
}

const slim = (b: Book): ShelfBook => ({
  id: b.id, title: b.title, author: b.author, cover_url: b.cover_url, cover_url_small: b.cover_url_small,
  isbn: b.isbn, genre: b.genre, year: b.year,
});

export type ImportMode = "update" | "replace";

interface ShelfApi extends ShelfState {
  count: number;
  rate: (book: Book, rating: number) => void;
  unrate: (id: number) => void;
  remove: (id: number) => void;
  dismiss: (id: number) => void;
  undismiss: (id: number) => void;
  markRead: (id: number) => void;
  applyImport: (res: ImportResult, mode: ImportMode) => void;
  clear: () => void;
  requestBody: () => { ratings: { id: number; rating: number }[]; read: number[]; to_read: number[]; dismissed: number[] };
  /** requestBody plus reviews and outside-the-catalog books, for the Assistant tab. */
  chatShelf: () => object;
}

const Ctx = createContext<ShelfApi | null>(null);
const uniq = (xs: number[]) => [...new Set(xs)];

export function ShelfProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<ShelfState>(load);
  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* storage unavailable: keep in memory */ }
  }, [s]);

  const rate = useCallback((book: Book, rating: number) => setS((p) => ({
    ...p,
    ratings: { ...p.ratings, [book.id]: { rating, book: slim(book), source: "manual" } },
    toRead: p.toRead.filter((i) => i !== book.id),
    dismissed: p.dismissed.filter((i) => i !== book.id),
  })), []);
  const unrate = useCallback((id: number) => setS((p) => {
    const { [id]: _, ...rest } = p.ratings;
    return { ...p, ratings: rest };
  }), []);
  /** Take a book off the shelf entirely (e.g. a wrong import match). */
  const remove = useCallback((id: number) => setS((p) => {
    const { [id]: _, ...rest } = p.ratings;
    const drop = (xs: number[]) => xs.filter((i) => i !== id);
    return { ...p, ratings: rest, read: drop(p.read), manualRead: drop(p.manualRead), toRead: drop(p.toRead) };
  }), []);
  const dismiss = useCallback((id: number) => setS((p) => ({ ...p, dismissed: uniq([...p.dismissed, id]) })), []);
  const undismiss = useCallback((id: number) => setS((p) => ({ ...p, dismissed: p.dismissed.filter((i) => i !== id) })), []);
  const markRead = useCallback((id: number) => setS((p) => ({
    ...p, read: uniq([...p.read, id]), manualRead: uniq([...p.manualRead, id]),
  })), []);
  const clear = useCallback(() => setS(EMPTY), []);

  const applyImport = useCallback((res: ImportResult, mode: ImportMode) => setS((p) => {
    // "update": drop everything from earlier imports (including stale matches), keep what the user did here.
    const keep = mode === "replace" ? {} : Object.fromEntries(Object.entries(p.ratings).filter(([, v]) => v.source === "manual"));
    const ratings: ShelfState["ratings"] = { ...keep };
    for (const r of res.rated) {
      if (!(r.id in keep)) ratings[r.id] = { rating: r.rating, book: slim(r), source: "import" };
    }
    const manualRead = mode === "replace" ? [] : p.manualRead;
    return {
      ratings,
      read: uniq([...manualRead, ...res.read_unrated]),
      manualRead,
      toRead: res.to_read.filter((i) => !(i in ratings)),
      dismissed: mode === "replace" ? [] : p.dismissed,
      reviews: capReviews(res.reviews ?? {}),
      outside: [...(res.unmatched ?? []), ...(res.unmatched_to_read ?? [])],
    };
  }), []);

  const value = useMemo<ShelfApi>(() => {
    const requestBody = () => ({
      ratings: Object.entries(s.ratings).map(([id, v]) => ({ id: Number(id), rating: v.rating })),
      read: s.read, to_read: s.toRead, dismissed: s.dismissed,
    });
    return {
      ...s, count: Object.keys(s.ratings).length, rate, unrate, remove, dismiss, undismiss, markRead, applyImport, clear,
      requestBody, chatShelf: () => ({ ...requestBody(), reviews: s.reviews, outside: s.outside }),
    };
  }, [s, rate, unrate, remove, dismiss, undismiss, markRead, applyImport, clear]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useShelf(): ShelfApi {
  const v = useContext(Ctx);
  if (!v) throw new Error("useShelf outside ShelfProvider");
  return v;
}
