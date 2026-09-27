import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Book, ImportResult } from "./api";

/** The user's shelf lives only in this browser (localStorage); the API is stateless. */
export type ShelfBook = Pick<Book, "id" | "title" | "author" | "cover_url" | "cover_url_small" | "isbn" | "genre" | "year">;

interface ShelfState {
  ratings: Record<number, { rating: number; book: ShelfBook }>;
  read: number[];
  toRead: number[];
  dismissed: number[];
}

const KEY = "goodrec.shelf.v1";
const EMPTY: ShelfState = { ratings: {}, read: [], toRead: [], dismissed: [] };

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

interface ShelfApi extends ShelfState {
  count: number;
  rate: (book: Book, rating: number) => void;
  unrate: (id: number) => void;
  dismiss: (id: number) => void;
  undismiss: (id: number) => void;
  applyImport: (res: ImportResult, mode: "merge" | "replace") => void;
  clear: () => void;
  requestBody: () => { ratings: { id: number; rating: number }[]; read: number[]; to_read: number[]; dismissed: number[] };
}

const Ctx = createContext<ShelfApi | null>(null);

export function ShelfProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<ShelfState>(load);
  useEffect(() => localStorage.setItem(KEY, JSON.stringify(s)), [s]);

  const rate = useCallback((book: Book, rating: number) => setS((p) => ({
    ...p,
    ratings: { ...p.ratings, [book.id]: { rating, book: slim(book) } },
    toRead: p.toRead.filter((i) => i !== book.id),
    dismissed: p.dismissed.filter((i) => i !== book.id),
  })), []);
  const unrate = useCallback((id: number) => setS((p) => {
    const { [id]: _, ...rest } = p.ratings;
    return { ...p, ratings: rest };
  }), []);
  const dismiss = useCallback((id: number) => setS((p) => ({ ...p, dismissed: [...new Set([...p.dismissed, id])] })), []);
  const undismiss = useCallback((id: number) => setS((p) => ({ ...p, dismissed: p.dismissed.filter((i) => i !== id) })), []);
  const clear = useCallback(() => setS(EMPTY), []);
  const applyImport = useCallback((res: ImportResult, mode: "merge" | "replace") => setS((p) => {
    const base = mode === "replace" ? EMPTY : p;
    const ratings = { ...base.ratings };
    for (const r of res.rated) ratings[r.id] = { rating: r.rating, book: slim(r) };
    return {
      ratings,
      read: [...new Set([...base.read, ...res.read_unrated])],
      toRead: [...new Set([...base.toRead, ...res.to_read])].filter((i) => !(i in ratings)),
      dismissed: base.dismissed,
    };
  }), []);

  const value = useMemo<ShelfApi>(() => ({
    ...s, count: Object.keys(s.ratings).length, rate, unrate, dismiss, undismiss, applyImport, clear,
    requestBody: () => ({
      ratings: Object.entries(s.ratings).map(([id, v]) => ({ id: Number(id), rating: v.rating })),
      read: s.read, to_read: s.toRead, dismissed: s.dismissed,
    }),
  }), [s, rate, unrate, dismiss, undismiss, applyImport, clear]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useShelf(): ShelfApi {
  const v = useContext(Ctx);
  if (!v) throw new Error("useShelf outside ShelfProvider");
  return v;
}
