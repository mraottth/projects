import { useCallback, useEffect, useState } from "react";
import { EMPTY_FILTERS, type BrowseSort, type Filters, type SortKey } from "./api";

/**
 * View, tab, sort and filters live in the URL query string, so any view can be bookmarked or shared.
 * Recommendations and Explore keep separate filter sets; only the current view's set is in the URL,
 * and the other survives in memory while navigating.
 */
export interface UrlState {
  view: "home" | "rate" | "import" | "recs" | "explore" | "yours" | "chat" | "about";
  tab: "for-you" | "popular" | "top-rated" | "to-read";
  sort: SortKey;
  layout: "list" | "map";
  filters: Filters;          // recommendations
  explore: Filters;          // explore page
  browseSort: BrowseSort;
}

/** Explore browses the whole catalog, so nothing is hidden by default. */
export const EXPLORE_DEFAULTS: Filters = {
  ...EMPTY_FILTERS, include_children: true, include_comics: true, include_series_continuations: true, include_ya: true,
};

const NUM_KEYS = ["year_min", "year_max", "min_avg_rating", "min_ratings_count", "max_ratings_count"] as const;
const LIST_KEYS = ["authors_include", "authors_exclude"] as const;
const BOOL_KEYS = ["include_children", "include_comics", "include_series_continuations", "include_ya"] as const;
const BROWSE_SORTS: BrowseSort[] = ["popular", "rating", "newest", "oldest", "title"];

function parseFilters(p: URLSearchParams, defaults: Filters): Filters {
  const f: Filters = { ...defaults };
  f.genres = p.getAll("genre");
  f.tags = p.getAll("tag");
  f.text = p.get("q") ?? "";
  for (const k of NUM_KEYS) f[k] = p.has(k) ? Number(p.get(k)) : null;
  for (const k of LIST_KEYS) f[k] = p.getAll(k).map(Number).filter(Number.isFinite);
  for (const k of BOOL_KEYS) f[k] = p.get(k) === "1" ? true : p.get(k) === "0" ? false : defaults[k];
  return f;
}

function writeFilters(p: URLSearchParams, f: Filters, defaults: Filters) {
  f.genres.forEach((g) => p.append("genre", g));
  f.tags.forEach((t) => p.append("tag", t));
  if (f.text) p.set("q", f.text);
  for (const k of NUM_KEYS) if (f[k] != null) p.set(k, String(f[k]));
  for (const k of LIST_KEYS) f[k].forEach((v) => p.append(k, String(v)));
  for (const k of BOOL_KEYS) if (f[k] !== defaults[k]) p.set(k, f[k] ? "1" : "0");
}

function parse(search: string): UrlState {
  const p = new URLSearchParams(search);
  const view = (p.get("view") as UrlState["view"]) ?? "home";
  const bs = p.get("bsort") as BrowseSort;
  return {
    view,
    tab: (p.get("tab") as UrlState["tab"]) ?? "for-you",
    sort: p.get("sort") === "predicted" ? "predicted" : "match",
    layout: p.get("layout") === "map" ? "map" : "list",
    filters: view === "recs" ? parseFilters(p, EMPTY_FILTERS) : { ...EMPTY_FILTERS },
    explore: view === "explore" ? parseFilters(p, EXPLORE_DEFAULTS) : { ...EXPLORE_DEFAULTS },
    browseSort: BROWSE_SORTS.includes(bs) ? bs : "popular",
  };
}

function serialize(s: UrlState): string {
  const p = new URLSearchParams();
  if (s.view !== "home") p.set("view", s.view);
  if (s.view === "recs") {
    if (s.tab !== "for-you") p.set("tab", s.tab);
    if (s.sort !== "match") p.set("sort", s.sort);
    if (s.layout !== "list") p.set("layout", s.layout);
    writeFilters(p, s.filters, EMPTY_FILTERS);
  }
  if (s.view === "explore") {
    if (s.browseSort !== "popular") p.set("bsort", s.browseSort);
    writeFilters(p, s.explore, EXPLORE_DEFAULTS);
  }
  const qs = p.toString();
  return qs ? `?${qs}` : window.location.pathname;
}

export function useUrlState() {
  const [state, setState] = useState<UrlState>(() => parse(window.location.search));

  useEffect(() => {
    // Back/forward: restore the URL's view and settings, keeping the other page's in-memory filters.
    const onPop = () => setState((prev) => {
      const next = parse(window.location.search);
      return next.view === "recs" ? { ...next, explore: prev.explore }
        : next.view === "explore" ? { ...next, filters: prev.filters } : { ...next, filters: prev.filters, explore: prev.explore };
    });
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const update = useCallback((patch: Partial<UrlState>, push = false) => {
    setState((prev) => {
      const next = { ...prev, ...patch };
      const url = serialize(next);
      if (push) window.history.pushState(null, "", url);
      else window.history.replaceState(null, "", url);
      return next;
    });
    if (push) window.scrollTo({ top: 0 });
  }, []);

  return [state, update] as const;
}
