import type { Filters } from "./api";
import type { FilterClick } from "./components/BookCard";

/** Add the clicked author / genre / tag to the page's filters (no-op if already there) and jump to the top. */
export function applyFilterClick(filters: Filters, f: FilterClick): Filters {
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (f.kind === "author") {
    return filters.authors_include.includes(f.id) ? filters : { ...filters, authors_include: [...filters.authors_include, f.id] };
  }
  const key = f.kind === "genre" ? "genres" : "tags";
  return filters[key].includes(f.value) ? filters : { ...filters, [key]: [...filters[key], f.value] };
}
