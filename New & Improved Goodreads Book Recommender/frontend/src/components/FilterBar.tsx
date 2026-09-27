import { useEffect, useState } from "react";
import { api, EMPTY_FILTERS, type Author, type Filters } from "../api";
import { rememberAuthor, useAuthorNames } from "../authorNames";
import { useShelf } from "../store";
import { GenreChip } from "./GenreChip";
import { formatCount } from "./Stars";

interface Props {
  defaults?: Filters;   // what "Clear all" restores and what counts as unfiltered
  filters: Filters;
  onChange: (f: Filters) => void;
  genres: { name: string; books: number }[];
}

// Log-scale presets for Goodreads-wide rating counts.
const COUNT_PRESETS: { label: string; min: number | null; max: number | null }[] = [
  { label: "Any", min: null, max: null },
  { label: "Hidden gems", min: null, max: 10_000 },
  { label: "Well known", min: 10_000, max: 200_000 },
  { label: "Blockbusters", min: 200_000, max: null },
];
const COUNT_STEPS = [0, 100, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000];
const RATING_MAX = 4.8, RATING_ANY = 2.9;   // slider runs 3.0–4.8; the stop just left of 3.0 = "Any"
const GENRES_SHOWN = 10;

export function FilterBar({ filters, onChange, genres, defaults = EMPTY_FILTERS }: Props) {
  const [text, setText] = useState(filters.text);
  const set = (patch: Partial<Filters>) => onChange({ ...filters, ...patch });

  // Debounce the free-text search-within-results box.
  useEffect(() => setText(filters.text), [filters.text]);
  useEffect(() => {
    if (text === filters.text) return;
    const t = setTimeout(() => set({ text }), 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  const active = activeCount(filters, defaults);
  const toggle = (key: "genres" | "tags", v: string) =>
    set({ [key]: filters[key].includes(v) ? filters[key].filter((x) => x !== v) : [...filters[key], v] });

  // Genres ranked by the user's favorites: books rated 4-5 stars (5 counts double), then catalog size.
  const shelf = useShelf();
  const [allGenres, setAllGenres] = useState(false);
  const affinity: Record<string, number> = {};
  for (const { rating, book } of Object.values(shelf.ratings)) {
    if (book.genre && rating >= 4) affinity[book.genre] = (affinity[book.genre] ?? 0) + (rating - 3);
  }
  const ranked = genres.filter((g) => g.books >= 100)
    .sort((a, b) => (affinity[b.name] ?? 0) - (affinity[a.name] ?? 0) || b.books - a.books);
  const shownGenres = allGenres ? ranked : ranked.filter((g, i) => i < GENRES_SHOWN || filters.genres.includes(g.name));
  const hasFavorites = Object.keys(affinity).length > 0;
  const preset = COUNT_PRESETS.find((p) => p.min === filters.min_ratings_count && p.max === filters.max_ratings_count);

  return (
    <section className="filters">
      <div className="filters-head">
        <h2>Filters{active ? ` (${active})` : ""}</h2>
        {active > 0 && <button type="button" className="link small" onClick={() => onChange(defaults)}>Clear all</button>}
      </div>

      <input className="search-within" type="search" placeholder="Search within results…"
             aria-label="Search within results (title, author, genre, tag)" value={text} onChange={(e) => setText(e.target.value)} />

      <div className="fsection">
        <span className="flabel">Tags</span>
        {filters.tags.length > 0 && (
          <div className="picked">
            {filters.tags.map((t) => (
              <span key={t} className="tag removable">
                {t}
                <button type="button" aria-label={`Remove tag ${t}`} onClick={() => toggle("tags", t)}>×</button>
              </span>
            ))}
          </div>
        )}
        <TagPicker selected={filters.tags} onAdd={(t) => toggle("tags", t)} />
      </div>

      <div className="fsection">
        <span className="flabel">Genre{hasFavorites && <span className="flabel-note"> · your favorites first</span>}</span>
        <div className="genre-chips">
          {shownGenres.map((g) => (
            <GenreChip key={g.name} genre={g.name} active={filters.genres.includes(g.name)} onClick={() => toggle("genres", g.name)} />
          ))}
          {ranked.length > GENRES_SHOWN && (
            <button type="button" className="link small" onClick={() => setAllGenres((v) => !v)}>
              {allGenres ? "Show fewer" : `+${ranked.length - shownGenres.length} more`}
            </button>
          )}
        </div>
      </div>

      <RatingSlider value={filters.min_avg_rating} onChange={(v) => set({ min_avg_rating: v })} />

      <label className="fsection">
        <span className="flabel">Published</span>
        <span className="range-inputs">
          <input type="number" placeholder="from" min={1000} max={2017} value={filters.year_min ?? ""}
                 onChange={(e) => set({ year_min: e.target.value ? Number(e.target.value) : null })} />
          –
          <input type="number" placeholder="to" min={1000} max={2017} value={filters.year_max ?? ""}
                 onChange={(e) => set({ year_max: e.target.value ? Number(e.target.value) : null })} />
        </span>
      </label>

      <div className="fsection">
        <span className="flabel">Number of ratings on Goodreads</span>
        <CountRange min={filters.min_ratings_count} max={filters.max_ratings_count}
                    onChange={(min, max) => set({ min_ratings_count: min, max_ratings_count: max })} />
      </div>

      <div className="fsection">
        <AuthorPicker label="Only these authors" ids={filters.authors_include} onChange={(ids) => set({ authors_include: ids })} />
      </div>

      <div className="fsection toggles">
        <label><input type="checkbox" checked={filters.include_series_continuations}
                      onChange={(e) => set({ include_series_continuations: e.target.checked })} /> Show later books in series</label>
      </div>

      <div className="fsection">
        <span className="flabel">Popularity</span>
        <div className="seg seg-wrap" role="group" aria-label="Popularity">
          {COUNT_PRESETS.map((p) => (
            <button key={p.label} type="button" className={preset === p ? "on" : ""}
                    onClick={() => set({ min_ratings_count: p.min, max_ratings_count: p.max })}>{p.label}</button>
          ))}
        </div>
      </div>
    </section>
  );
}

function activeCount(f: Filters, d: Filters): number {
  return [f.genres.length > 0, f.tags.length > 0, f.authors_include.length > 0, f.authors_exclude.length > 0,
    f.year_min != null, f.year_max != null, f.min_avg_rating != null,
    f.min_ratings_count != null || f.max_ratings_count != null, !!f.text, f.include_children !== d.include_children,
    f.include_series_continuations !== d.include_series_continuations].filter(Boolean).length;
}

/** Minimum Goodreads average: 3.0–4.8 in 0.1 steps; the leftmost stop means "Any". */
function RatingSlider({ value, onChange }: { value: number | null; onChange: (v: number | null) => void }) {
  const [local, setLocal] = useState(value ?? RATING_ANY);
  useEffect(() => setLocal(value ?? RATING_ANY), [value]);
  useEffect(() => {   // commit after the thumb settles, so dragging doesn't fire a request per step
    const v = local <= RATING_ANY + 1e-9 ? null : Math.round(local * 10) / 10;
    if (v === value) return;
    const t = setTimeout(() => onChange(v), 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [local]);
  const label = local <= RATING_ANY + 1e-9 ? "Any" : `★ ${local.toFixed(1)}+`;
  return (
    <div className="fsection">
      <span className="flabel slider-label">Minimum Goodreads rating <span className="slider-value">{label}</span></span>
      <input type="range" className="single-range" min={RATING_ANY} max={RATING_MAX} step={0.1} value={local}
             aria-label="Minimum Goodreads rating" aria-valuetext={label} onChange={(e) => setLocal(Number(e.target.value))} />
      <div className="range-ends"><span>Any</span><span>{RATING_MAX.toFixed(1)}</span></div>
    </div>
  );
}

/** One track, two thumbs (min and max), on log-spaced stops. */
function CountRange({ min, max, onChange }: { min: number | null; max: number | null; onChange: (a: number | null, b: number | null) => void }) {
  const last = COUNT_STEPS.length - 1;
  const toStep = (v: number | null, dflt: number) =>
    v == null ? dflt : COUNT_STEPS.reduce((best, s, i) => (Math.abs(s - v) < Math.abs(COUNT_STEPS[best] - v) ? i : best), 0);
  const lo = toStep(min, 0), hi = toStep(max, last);
  const pct = (i: number) => (i / last) * 100;
  const label = (i: number, end: boolean) => (end && i === last ? "any" : formatCount(COUNT_STEPS[i]));
  return (
    <div className="count-range">
      <div className="dual-range">
        <div className="dual-track" />
        <div className="dual-fill" style={{ left: `${pct(lo)}%`, width: `${pct(hi) - pct(lo)}%` }} />
        <input type="range" min={0} max={last} value={lo} aria-label="Minimum number of ratings"
               onChange={(e) => { const v = Math.min(Number(e.target.value), hi); onChange(v === 0 ? null : COUNT_STEPS[v], max); }} />
        <input type="range" min={0} max={last} value={hi} aria-label="Maximum number of ratings"
               onChange={(e) => { const v = Math.max(Number(e.target.value), lo); onChange(min, v === last ? null : COUNT_STEPS[v]); }} />
      </div>
      <span className="small muted">{label(lo, false)} – {label(hi, true)} ratings</span>
    </div>
  );
}

let tagCache: Promise<{ name: string; books: number }[]> | null = null;

/** Type to find a subgenre tag; the most common tags show when the box is empty. */
function TagPicker({ selected, onAdd }: { selected: string[]; onAdd: (t: string) => void }) {
  const [all, setAll] = useState<{ name: string; books: number }[]>([]);
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  useEffect(() => {
    tagCache ??= api.tags().catch(() => []);
    tagCache.then(setAll);
  }, []);
  const ql = q.trim().toLowerCase();
  // Empty box: the 50 most common tags (scrollable); typing narrows across all tags.
  const matches = all.filter((t) => !selected.includes(t.name) && (!ql || t.name.toLowerCase().includes(ql))).slice(0, 50);
  return (
    <div className="typeahead">
      <input type="search" placeholder="Find a tag, e.g. cozy mystery…" value={q} aria-label="Find a tag"
             onChange={(e) => { setQ(e.target.value); setOpen(true); }} onFocus={() => setOpen(true)}
             onBlur={() => setTimeout(() => setOpen(false), 150)} />
      {open && matches.length > 0 && (
        <ul className="typeahead-list scrollable" aria-label={ql ? "Matching tags" : "Most common tags"}>
          {matches.map((t) => (
            <li key={t.name}>
              <button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => { onAdd(t.name); setQ(""); setOpen(false); }}>
                {t.name} <span className="muted small">{t.books.toLocaleString()} books</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function AuthorPicker({ label, ids, onChange }: { label: string; ids: number[]; onChange: (ids: number[]) => void }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<Author[]>([]);
  // Names for ids restored from a shared URL or added by clicking an author on a card.
  const names = useAuthorNames(ids);
  useEffect(() => {
    if (q.trim().length < 2) { setResults([]); return; }
    const ctl = new AbortController();
    const t = setTimeout(() => api.authors(q, ctl.signal).then(setResults).catch(() => {}), 150);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [q]);
  return (
    <div className="author-picker">
      <span className="flabel">{label}</span>
      <div className="picked">
        {ids.map((id) => (
          <span key={id} className="tag removable">
            {names[id] ?? "…"}
            <button type="button" aria-label="remove" onClick={() => onChange(ids.filter((i) => i !== id))}>×</button>
          </span>
        ))}
      </div>
      <div className="typeahead">
        <input type="search" placeholder="Type an author…" value={q} onChange={(e) => setQ(e.target.value)} />
        {results.length > 0 && (
          <ul className="typeahead-list">
            {results.map((a) => (
              <li key={a.id}>
                <button type="button" onClick={() => { rememberAuthor(a.id, a.name); onChange([...new Set([...ids, a.id])]); setQ(""); }}>
                  {a.name} <span className="muted small">{a.n_books} books</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
