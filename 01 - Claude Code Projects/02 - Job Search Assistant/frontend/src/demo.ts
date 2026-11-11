/**
 * The public demo (asked 2026-11-11): a static, read-only build (`npm run build:demo`, VITE_DEMO=1)
 * served from GitHub Pages, with no server and no Claude calls.
 *
 * demo/record.py runs the real app on a fictional, seeded database and saves its answer to every GET
 * the pages make (demo-data.json, served next to the page). apiFetch() calls demoFetch() instead of
 * the network: a GET is answered from that recording, and anything else (every change) is refused
 * with a "read-only demo" notice. In a normal build DEMO is false at build time and all of this is
 * left out of the bundle.
 */

export const DEMO = import.meta.env.VITE_DEMO === "1";
export const DEMO_READONLY_EVENT = "jobhunt:demo-readonly";
export const DEMO_READONLY_MESSAGE = "This is a read-only demo, so nothing here can be changed.";

interface Recording {
  recorded_at: string;
  responses: Record<string, { status: number; body: unknown }>;
}

let recording: Promise<Recording> | null = null;

/** A file the recording saved (downloads/...), as a URL under wherever the demo is served. */
export function demoUrl(file: string): string {
  return `${import.meta.env.BASE_URL}${file}`;
}

function load(): Promise<Recording> {
  recording ??= fetch(demoUrl("demo-data.json"))
    .then((res) => res.json() as Promise<Recording>)
    .then(shiftDates);
  return recording;
}

export async function demoFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const method = (init.method ?? "GET").toUpperCase();
  const json = (status: number, body: unknown) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  if (method !== "GET") {
    window.dispatchEvent(new Event(DEMO_READONLY_EVENT));
    return json(403, { detail: DEMO_READONLY_MESSAGE });
  }
  const { responses } = await load();
  const hit = responses[path];
  return hit ? json(hit.status, hit.body) : json(404, { detail: "Not part of the demo." });
}

// ---- Keep the demo's dates current
// The recording was made on one day; visited weeks later, "Last updated" would say weeks ago and
// New & Noteworthy's "today" would be a past date. So every date in it moves forward by the time
// since it was recorded: date-times by exactly that much (so "3 hours ago" stays 3 hours ago),
// plain dates by whole days. The Timeline (/api/changelog) is real history and isn't moved.

const DAY_MS = 86_400_000;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const DB_TIME = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/; // SQLite's datetime('now'), UTC
const ISO_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;
const DIGEST_PATH = /^(\/api\/digest\/)(\d{4}-\d{2}-\d{2})$/;
const NOT_SHIFTED = new Set(["/api/changelog"]);

function shiftDates(rec: Recording): Recording {
  const ms = Date.now() - Date.parse(rec.recorded_at);
  const days = Math.round(ms / DAY_MS);
  if (ms < 60_000) return rec;
  const shiftDate = (d: string) => new Date(Date.parse(`${d}T00:00:00Z`) + days * DAY_MS).toISOString().slice(0, 10);
  const shift = (value: unknown): unknown => {
    if (typeof value === "string") {
      if (DATE.test(value)) return shiftDate(value);
      if (DB_TIME.test(value)) {
        return new Date(Date.parse(`${value.replace(" ", "T")}Z`) + ms).toISOString().slice(0, 19).replace("T", " ");
      }
      if (ISO_TIME.test(value)) return new Date(Date.parse(value) + ms).toISOString().replace(/\.\d{3}Z$/, "+00:00");
      return value;
    }
    if (Array.isArray(value)) return value.map(shift);
    if (value && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, shift(v)]));
    }
    return value;
  };
  const responses: Recording["responses"] = {};
  for (const [path, hit] of Object.entries(rec.responses)) {
    const m = DIGEST_PATH.exec(path);
    const key = m ? `${m[1]}${shiftDate(m[2])}` : path;
    responses[key] = NOT_SHIFTED.has(path) ? hit : { status: hit.status, body: shift(hit.body) };
  }
  return { ...rec, responses };
}
