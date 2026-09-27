import { useCallback, useEffect, useState } from "react";
import { EMPTY_FILTERS, type Filters } from "./api";

/** Filters + view + tab live in the URL query string, so any view can be bookmarked or shared. */
export interface UrlState {
  view: "home" | "rate" | "import" | "recs";
  tab: "for-you" | "popular" | "top-rated" | "to-read";
  filters: Filters;
}

const NUM_KEYS = ["year_min", "year_max", "min_avg_rating", "min_ratings_count", "max_ratings_count"] as const;
const LIST_KEYS = ["authors_include", "authors_exclude"] as const;
const BOOL_KEYS = ["include_children", "include_comics", "include_series_continuations"] as const;

function parse(search: string): UrlState {
  const p = new URLSearchParams(search);
  const f: Filters = { ...EMPTY_FILTERS };
  f.genres = p.getAll("genre");
  f.text = p.get("q") ?? "";
  for (const k of NUM_KEYS) f[k] = p.has(k) ? Number(p.get(k)) : null;
  for (const k of LIST_KEYS) f[k] = p.getAll(k).map(Number).filter(Number.isFinite);
  for (const k of BOOL_KEYS) f[k] = p.get(k) === "1";
  return {
    view: (p.get("view") as UrlState["view"]) ?? "home",
    tab: (p.get("tab") as UrlState["tab"]) ?? "for-you",
    filters: f,
  };
}

function serialize(s: UrlState): string {
  const p = new URLSearchParams();
  if (s.view !== "home") p.set("view", s.view);
  if (s.view === "recs" && s.tab !== "for-you") p.set("tab", s.tab);
  if (s.view === "recs") {
    const f = s.filters;
    f.genres.forEach((g) => p.append("genre", g));
    if (f.text) p.set("q", f.text);
    for (const k of NUM_KEYS) if (f[k] != null) p.set(k, String(f[k]));
    for (const k of LIST_KEYS) f[k].forEach((v) => p.append(k, String(v)));
    for (const k of BOOL_KEYS) if (f[k]) p.set(k, "1");
  }
  const qs = p.toString();
  return qs ? `?${qs}` : window.location.pathname;
}

export function useUrlState() {
  const [state, setState] = useState<UrlState>(() => parse(window.location.search));

  useEffect(() => {
    const onPop = () => setState(parse(window.location.search));
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
