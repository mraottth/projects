import { useEffect, useRef, useState } from "react";
import { api, EMPTY_FILTERS, type Author, type Filters } from "../api";
import { GenreChip } from "./GenreChip";
import { formatCount } from "./Stars";

interface Props {
  filters: Filters;
  onChange: (f: Filters) => void;
  genres: { name: string; books: number }[];
}

const RATING_OPTIONS = [null, 3.5, 3.75, 4.0, 4.25];
// Log-scale presets for Goodreads-wide rating counts.
const COUNT_PRESETS: { label: string; min: number | null; max: number | null }[] = [
  { label: "Any", min: null, max: null },
  { label: "Hidden gems", min: null, max: 10_000 },
  { label: "Well known", min: 10_000, max: 200_000 },
  { label: "Blockbusters", min: 200_000, max: null },
];
const COUNT_STEPS = [0, 100, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000];

export function FilterBar({ filters, onChange, genres }: Props) {
  const [open, setOpen] = useState(false);
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

  const active = activeCount(filters);
  const toggleGenre = (g: string) =>
    set({ genres: filters.genres.includes(g) ? filters.genres.filter((x) => x !== g) : [...filters.genres, g] });
  // Show the 12 largest genres (plus any selected) until expanded.
  const [allGenres, setAllGenres] = useState(false);
  const byCount = genres.filter((g) => g.books >= 100).sort((a, b) => b.books - a.books);
  const shownGenres = allGenres ? byCount : byCount.filter((g, i) => i < 12 || filters.genres.includes(g.name));
  const hiddenCount = byCount.length - shownGenres.length;
  const preset = COUNT_PRESETS.find((p) => p.min === filters.min_ratings_count && p.max === filters.max_ratings_count);

  return (
    <section className="filters">
      <div className="filter-row">
        <input className="search-within" type="search" placeholder="Search within results (title, author, tag)…"
               value={text} onChange={(e) => setText(e.target.value)} />
        <div className="seg" role="group" aria-label="Popularity">
          {COUNT_PRESETS.map((p) => (
            <button key={p.label} type="button" className={preset === p ? "on" : ""}
                    onClick={() => set({ min_ratings_count: p.min, max_ratings_count: p.max })}>{p.label}</button>
          ))}
        </div>
        <button type="button" className={`ghost${open ? " on" : ""}`} onClick={() => setOpen((o) => !o)} aria-expanded={open}>
          More filters{active ? ` (${active})` : ""} {open ? "▴" : "▾"}
        </button>
        {active > 0 && <button type="button" className="link small" onClick={() => onChange(EMPTY_FILTERS)}>Clear all</button>}
      </div>

      <div className="genre-chips">
        {shownGenres.map((g) => (
          <GenreChip key={g.name} genre={g.name} active={filters.genres.includes(g.name)} onClick={() => toggleGenre(g.name)} />
        ))}
        {hiddenCount > 0 && (
          <button type="button" className="link small" onClick={() => setAllGenres((v) => !v)}>
            {allGenres ? "Fewer genres" : `+${hiddenCount} more genres`}
          </button>
        )}
      </div>

      {open && (
        <div className="filter-panel">
          <label>
            <span className="flabel">Published</span>
            <span className="range-inputs">
              <input type="number" placeholder="from" min={1000} max={2017} value={filters.year_min ?? ""}
                     onChange={(e) => set({ year_min: e.target.value ? Number(e.target.value) : null })} />
              –
              <input type="number" placeholder="to" min={1000} max={2017} value={filters.year_max ?? ""}
                     onChange={(e) => set({ year_max: e.target.value ? Number(e.target.value) : null })} />
            </span>
          </label>
          <div>
            <span className="flabel">Minimum average rating</span>
            <div className="seg">
              {RATING_OPTIONS.map((r) => (
                <button key={String(r)} type="button" className={filters.min_avg_rating === r ? "on" : ""}
                        onClick={() => set({ min_avg_rating: r })}>{r == null ? "Any" : `★ ${r.toFixed(2)}+`}</button>
              ))}
            </div>
          </div>
          <div>
            <span className="flabel">Number of ratings on Goodreads</span>
            <CountRange min={filters.min_ratings_count} max={filters.max_ratings_count}
                        onChange={(min, max) => set({ min_ratings_count: min, max_ratings_count: max })} />
          </div>
          <div className="author-filters">
            <AuthorPicker label="Only these authors" ids={filters.authors_include}
                          onChange={(ids) => set({ authors_include: ids })} />
            <AuthorPicker label="Exclude authors" ids={filters.authors_exclude}
                          onChange={(ids) => set({ authors_exclude: ids })} />
          </div>
          <div className="toggles">
            <label><input type="checkbox" checked={filters.include_series_continuations}
                          onChange={(e) => set({ include_series_continuations: e.target.checked })} /> Show later books in series</label>
            <label><input type="checkbox" checked={filters.include_children}
                          onChange={(e) => set({ include_children: e.target.checked })} /> Include children's books</label>
            <label><input type="checkbox" checked={filters.include_comics}
                          onChange={(e) => set({ include_comics: e.target.checked })} /> Include comics &amp; manga</label>
          </div>
        </div>
      )}
    </section>
  );
}

function activeCount(f: Filters): number {
  return [f.genres.length > 0, f.authors_include.length > 0, f.authors_exclude.length > 0, f.year_min != null,
    f.year_max != null, f.min_avg_rating != null, f.min_ratings_count != null || f.max_ratings_count != null,
    !!f.text, f.include_children, f.include_comics, f.include_series_continuations].filter(Boolean).length;
}

function CountRange({ min, max, onChange }: { min: number | null; max: number | null; onChange: (a: number | null, b: number | null) => void }) {
  const toStep = (v: number | null, dflt: number) => (v == null ? dflt : COUNT_STEPS.reduce((best, s, i) => (Math.abs(s - v) < Math.abs(COUNT_STEPS[best] - v) ? i : best), 0));
  const lo = toStep(min, 0), hi = toStep(max, COUNT_STEPS.length - 1);
  const label = (i: number, end: boolean) => (end && i === COUNT_STEPS.length - 1 ? "any" : formatCount(COUNT_STEPS[i]));
  return (
    <div className="count-range">
      <input type="range" min={0} max={COUNT_STEPS.length - 1} value={lo} aria-label="Minimum ratings"
             onChange={(e) => { const v = Math.min(Number(e.target.value), hi); onChange(v === 0 ? null : COUNT_STEPS[v], max); }} />
      <input type="range" min={0} max={COUNT_STEPS.length - 1} value={hi} aria-label="Maximum ratings"
             onChange={(e) => { const v = Math.max(Number(e.target.value), lo); onChange(min, v === COUNT_STEPS.length - 1 ? null : COUNT_STEPS[v]); }} />
      <span className="small muted">{label(lo, false)} – {label(hi, true)} ratings</span>
    </div>
  );
}

function AuthorPicker({ label, ids, onChange }: { label: string; ids: number[]; onChange: (ids: number[]) => void }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<Author[]>([]);
  const names = useRef<Record<number, string>>({});
  const [, force] = useState(0);
  // Resolve names for ids restored from a shared/bookmarked URL.
  useEffect(() => {
    const missing = ids.filter((i) => !(i in names.current));
    if (!missing.length) return;
    api.authorNames(missing).then((m) => {
      for (const [k, v] of Object.entries(m)) names.current[Number(k)] = v;
      force((n) => n + 1);
    }).catch(() => {});
  }, [ids]);
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
            {names.current[id] ?? `Author ${id}`}
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
                <button type="button" onClick={() => { names.current[a.id] = a.name; onChange([...new Set([...ids, a.id])]); setQ(""); }}>
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
