import { useEffect, useMemo, useRef } from "react";
import type { Book, ImportResult } from "../api";
import { Cover } from "./Cover";

export type FlowBook = Pick<Book, "id" | "title" | "author" | "cover_url" | "cover_url_small" | "isbn" | "genre">;

export const FLOW_COVERS = 30;   // the user's own covers (topped up from the cover wall)
export const FLOW_RECS = 20;     // then their top recommendations, as a second wave
const FADE_EARLY_MS = 750;       // start the fade-out this much before the flow's natural end

const hasCover = (b: FlowBook) => !!(b.cover_url || b.cover_url_small || b.isbn);

/** The user's highest-rated books with a cover, best first. */
export function topCovers(res: ImportResult): FlowBook[] {
  return [...res.rated].filter(hasCover).sort((a, b) => b.rating - a.rating).slice(0, FLOW_COVERS);
}

/** Same, from the shelf in the store (Rate books page). */
export function shelfCovers(ratings: Record<number, { rating: number; book: FlowBook }>): FlowBook[] {
  return Object.values(ratings).sort((a, b) => b.rating - a.rating).map((r) => r.book).filter(hasCover).slice(0, FLOW_COVERS);
}

/** Top up to FLOW_COVERS with homepage cover-wall books when the user has fewer rated covers. */
export function fillCovers(own: FlowBook[], wall: FlowBook[]): FlowBook[] {
  const seen = new Set(own.map((b) => b.id));
  const extra = wall.filter((b) => hasCover(b) && !seen.has(b.id));
  return [...own, ...extra].slice(0, FLOW_COVERS);
}

/** /api/recommend body for an import result (the store updates asynchronously, so don't read it yet). */
export function importBody(res: ImportResult) {
  return { ratings: res.rated.map((b) => ({ id: b.id, rating: b.rating })), read: res.read_unrated, to_read: res.to_read };
}

// Deterministic "random" per cover, so the flow looks organic without jitter between renders.
// Each cover eases to a pause mid-flight at its own --mid (18-74vw across) so they don't all stack up in the centre.
const rand = (i: number, k: number) => ((Math.sin((i + 1) * 12.9898 * k) * 43758.5453) % 1 + 1) % 1;

/**
 * Transition from an import (or Rate books) to Recommendations: the user's top-rated covers stream across the
 * screen, then their top recommendations follow as a second wave (added when /api/recommend answers). All timings
 * are fractions of `duration`, after an optional `lead` (ms) during which only the caption shows; Recommendations is
 * shown underneath at the halfway point of the flow.
 */
export function CoverFlow({ books, recs, caption, duration, lead = 0, onSwitch, onDone }: {
  books: FlowBook[]; recs: FlowBook[] | null; caption: string; duration: number; lead?: number;
  onSwitch: () => void; onDone: () => void;
}) {
  const start = useRef(performance.now());
  // Ending: at fadeAt a white veil fades in over the covers, then the (now white) overlay fades away to reveal
  // Recommendations. fadeAt is FADE_EARLY_MS before the last 15% of lead + duration.
  const total = lead + duration;
  const fadeLen = total * 0.15;
  const fadeAt = total - fadeLen - FADE_EARLY_MS;
  const step = fadeLen * 0.6;                      // veil in, then overlay out
  useEffect(() => {
    const t1 = window.setTimeout(onSwitch, lead + duration * 0.5);
    const t2 = window.setTimeout(onDone, fadeAt + 2 * step);
    return () => { window.clearTimeout(t1); window.clearTimeout(t2); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const D = duration;
  const own = useMemo(() => books.map((b, i) => ({
    b,
    style: {
      "--y": `${4 + ((i * 5) % 6) * 14 + rand(i, 1) * 5}vh`,
      "--delay": `${lead + D * (0.04 + 0.36 * (i / Math.max(books.length - 1, 1)) + (rand(i, 2) - 0.5) * 0.04)}ms`,
      "--dur": `${D * (0.5 + rand(i, 3) * 0.1)}ms`,
      "--rot": `${(rand(i, 4) - 0.5) * 24}deg`,
      "--lift": `${(rand(i, 5) - 0.5) * 16}vh`,
      "--scale": `${0.85 + rand(i, 6) * 0.35}`,
      "--mid": `${18 + rand(i, 7) * 56}vw`,
    } as React.CSSProperties,
  })), [books, D, lead]);

  // Second wave: recommendations. Delays are relative to when they arrive, aimed at 32-50% of the timeline;
  // if they arrive too late to cross the screen before the fade, they're skipped.
  const late = useMemo(() => {
    if (!recs) return [];
    const elapsed = performance.now() - start.current;
    if (elapsed > lead + D * 0.45) return [];
    const seen = new Set(books.map((b) => b.id));
    const list = recs.filter((b) => hasCover(b) && !seen.has(b.id)).slice(0, FLOW_RECS);
    return list.map((b, j) => {
      const i = j + 100;
      const target = lead + D * (0.32 + 0.18 * (j / Math.max(list.length - 1, 1)));
      return {
        b,
        style: {
          "--y": `${10 + ((j * 7) % 5) * 15 + rand(i, 1) * 5}vh`,
          "--delay": `${Math.max(0, target - elapsed)}ms`,
          "--dur": `${D * (0.42 + rand(i, 3) * 0.05)}ms`,
          "--rot": `${(rand(i, 4) - 0.5) * 20}deg`,
          "--lift": `${(rand(i, 5) - 0.5) * 12}vh`,
          "--scale": `${1 + rand(i, 6) * 0.25}`,
          "--mid": `${18 + rand(i, 7) * 56}vw`,
        } as React.CSSProperties,
      };
    });
  }, [recs, books, D, lead]);

  return (
    <div className="coverflow" aria-hidden="true" style={{ "--fade-at": `${fadeAt}ms`, "--reveal-at": `${fadeAt + step}ms`, "--step": `${step}ms` } as React.CSSProperties}>
      {[...own, ...late].map(({ b, style }) => (
        <div key={b.id} className="coverflow-item" style={style}>
          <Cover book={b} size="md" />
        </div>
      ))}
      <p className="coverflow-caption">{caption}</p>
      <div className="coverflow-veil" />
    </div>
  );
}
