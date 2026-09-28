import { useEffect, useMemo, useRef } from "react";
import type { Book, ImportResult } from "../api";
import { Cover } from "./Cover";

export type FlowBook = Pick<Book, "id" | "title" | "author" | "cover_url" | "cover_url_small" | "isbn" | "genre">;

export const FLOW_COVERS = 30;   // the user's own covers (topped up from the cover wall)
export const FLOW_RECS = 20;     // then their top recommendations, as a second wave
const FADE_EARLY_MS = 750;       // start the fade-out this much before the flow's natural end
const LANE_ORDER = [0, 3, 1, 4, 2, 5];   // six rows, 14vh apart; consecutive covers skip rows
const MIN_GAP = 0.7;             // cover widths of horizontal travel between covers in neighbouring rows

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
  // Spacing: covers launch one "slot" apart, cycling through lanes in LANE_ORDER so covers launched close together
  // are never in neighbouring lanes. The slot gap is at least MIN_GAP cover-widths of travel between neighbouring
  // lanes, so on narrow screens (where covers cross slowly) fewer covers fly instead of piling up.
  const slotMs = useMemo(() => {
    const vw = window.innerWidth;
    const coverW = Math.min(132, Math.max(84, vw * 0.09));
    const speed = (vw + 200) / (D * 0.52);        // px per ms at the average duration below
    return (MIN_GAP * coverW) / speed / 2;        // neighbouring lanes are two slots apart
  }, [D]);
  const slot = (k: number, delay: number) => ({
    "--y": `${4 + LANE_ORDER[k % LANE_ORDER.length] * 14 + rand(k, 1) * 3}vh`,
    "--delay": `${delay}ms`,
    "--dur": `${D * (0.5 + rand(k, 3) * 0.04)}ms`,
    "--rot": `${(rand(k, 4) - 0.5) * 16}deg`,
    "--lift": `${(rand(k, 5) - 0.5) * 6}vh`,
    "--scale": `${0.9 + rand(k, 6) * 0.2}`,
  }) as React.CSSProperties;

  // First wave: the user's covers, over 4-32% of the timeline.
  const OWN_END = 0.32;
  const own = useMemo(() => {
    const span = D * (OWN_END - 0.04);
    const gap = Math.max(slotMs, span / Math.max(books.length - 1, 1));
    const fit = books.slice(0, Math.floor(span / gap) + 1);
    return fit.map((b, i) => ({ b, style: slot(i, lead + D * 0.04 + i * gap) }));
  }, [books, D, lead, slotMs]); // eslint-disable-line react-hooks/exhaustive-deps

  // Second wave: recommendations, continuing the same lane cycle from where the first wave ended (or from when they
  // arrive, if later) until 50% of the timeline; anything that can't launch by then is skipped.
  const late = useMemo(() => {
    if (!recs) return [];
    const elapsed = performance.now() - start.current;
    const from = Math.max(lead + D * OWN_END + slotMs, elapsed), until = lead + D * 0.5;
    if (from > until) return [];
    const seen = new Set(books.map((b) => b.id));
    const list = recs.filter((b) => hasCover(b) && !seen.has(b.id)).slice(0, FLOW_RECS);
    const n = Math.min(list.length, Math.floor((until - from) / slotMs) + 1);
    const gap = n > 1 ? Math.max(slotMs, (until - from) / (n - 1)) : 0;
    return list.slice(0, n).map((b, j) => ({ b, style: slot(own.length + j, from - elapsed + j * gap) }));
  }, [recs, books, D, lead, slotMs, own.length]); // eslint-disable-line react-hooks/exhaustive-deps

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
