import type { Filters } from "../api";
import { useAuthorNames } from "../authorNames";
import { formatCount } from "./Stars";

interface Props {
  filters: Filters;
  defaults: Filters;
  onChange: (f: Filters) => void;
}

/** "Filtered by" bar above a book list: every active filter as a removable chip, plus Clear all. */
export function ActiveFilters({ filters: f, defaults, onChange }: Props) {
  const names = useAuthorNames([...f.authors_include, ...f.authors_exclude]);
  const set = (patch: Partial<Filters>) => onChange({ ...f, ...patch });
  const chips: { key: string; label: string; remove: () => void }[] = [];

  if (f.text) chips.push({ key: "text", label: `“${f.text}”`, remove: () => set({ text: "" }) });
  for (const t of f.tags) chips.push({ key: `tag:${t}`, label: t, remove: () => set({ tags: f.tags.filter((x) => x !== t) }) });
  for (const g of f.genres) chips.push({ key: `genre:${g}`, label: g, remove: () => set({ genres: f.genres.filter((x) => x !== g) }) });
  for (const a of f.authors_include) {
    chips.push({ key: `a:${a}`, label: `by ${names[a] ?? "…"}`, remove: () => set({ authors_include: f.authors_include.filter((x) => x !== a) }) });
  }
  for (const a of f.authors_exclude) {
    chips.push({ key: `x:${a}`, label: `not ${names[a] ?? "…"}`, remove: () => set({ authors_exclude: f.authors_exclude.filter((x) => x !== a) }) });
  }
  if (f.min_avg_rating != null) chips.push({ key: "rating", label: `★ ${f.min_avg_rating.toFixed(1)}+`, remove: () => set({ min_avg_rating: null }) });
  if (f.year_min != null || f.year_max != null) {
    const label = f.year_min != null && f.year_max != null ? `${f.year_min}–${f.year_max}`
      : f.year_min != null ? `${f.year_min} or later` : `${f.year_max} or earlier`;
    chips.push({ key: "year", label: `Published ${label}`, remove: () => set({ year_min: null, year_max: null }) });
  }
  if (f.min_ratings_count != null || f.max_ratings_count != null) {
    const lo = f.min_ratings_count != null ? formatCount(f.min_ratings_count) : "0";
    const hi = f.max_ratings_count != null ? formatCount(f.max_ratings_count) : "any";
    chips.push({ key: "count", label: `${lo}–${hi} ratings`, remove: () => set({ min_ratings_count: null, max_ratings_count: null }) });
  }
  if (f.include_series_continuations !== defaults.include_series_continuations) {
    chips.push({ key: "series", label: f.include_series_continuations ? "Later books in series shown" : "Later books in series hidden",
                 remove: () => set({ include_series_continuations: defaults.include_series_continuations }) });
  }

  if (!chips.length) return null;
  return (
    <div className="active-filters" role="region" aria-label="Active filters">
      <span className="af-label">Filtered by</span>
      {chips.map((c) => (
        <span key={c.key} className="af-chip">
          {c.label}
          <button type="button" aria-label={`Remove filter: ${c.label}`} onClick={c.remove}>×</button>
        </span>
      ))}
      <button type="button" className="link small" onClick={() => onChange(defaults)}>Clear all</button>
    </div>
  );
}
