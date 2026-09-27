import { useEffect, useState } from "react";
import { api } from "./api";

/** Shared id -> name cache for authors used in filters (names are resolved lazily from the API). */
const cache: Record<number, string> = {};

export function rememberAuthor(id: number, name: string) {
  cache[id] = name;
}

export function useAuthorNames(ids: number[]): Record<number, string> {
  const [, bump] = useState(0);
  const key = ids.join(",");
  useEffect(() => {
    const missing = ids.filter((i) => !(i in cache));
    if (!missing.length) return;
    let live = true;
    api.authorNames(missing).then((m) => {
      for (const [k, v] of Object.entries(m)) cache[Number(k)] = v;
      if (live) bump((n) => n + 1);
    }).catch(() => {});
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return cache;
}
