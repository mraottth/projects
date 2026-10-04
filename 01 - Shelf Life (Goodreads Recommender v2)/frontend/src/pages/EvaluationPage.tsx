import { useEffect, useState } from "react";
import data from "../evaluations.json";
import { Markdown } from "../components/Markdown";
import { VersionChart, type Reference, type Track, type Version } from "../components/VersionChart";

/**
 * Evaluation: how the recommender improved, version by version (chart), and the evaluation reports
 * (eval/reports/*.md) as one card per model version, newest version first (the champion's open): the version's
 * report of record, with its other runs nested inside. Runs before the time-based test get a final card.
 * A Ranking | Rating switch (URL `track=rating`) picks the evaluation track for the intro, the chart, the panel
 * and every card: a card shows its report's shared header, that track's section and the shared notes.
 * Data: scripts/build_evaluations.py.
 */

interface Report {
  id: string; date: string; time: string; kind: string; title: string; model: string | null; set: string | null;
  n_users: number | null; split_hash: string | null; ndcg10: number | null; mae: number | null; champion: boolean;
  head: string; track1: string; track2: string | null; tail: string; version: string | null;
}
interface Group {
  id: string; title: string; date: string; champion: boolean; ndcg10: number | null; mae: number | null;
  main: string | null; others: string[];
}
interface Evaluations {
  versions: Version[]; references: Reference[]; rating_references: Reference[]; reports: Report[]; groups: Group[];
  buckets: number[];
}

const d = data as unknown as Evaluations;
const fmtDate = (s: string) => new Date(`${s}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });

const report = Object.fromEntries(d.reports.map((r) => [r.id, r]));

/** A report's Markdown for one track: the shared header, that track's section, the shared notes. */
function body(r: Report, track: Track): string {
  if (track === "ranking") return r.head + r.track1 + r.tail;
  return r.head + (r.track2 ?? "_No rating evaluation in this report: it was run before the rating-prediction track existed._\n\n") + r.tail;
}

function Score({ track, ndcg10, mae }: { track: Track; ndcg10: number | null; mae: number | null }) {
  if (track === "ranking") return ndcg10 != null ? <span className="ev-report-score">NDCG@10 {ndcg10.toFixed(4)}</span> : null;
  return mae != null ? <span className="ev-report-score">MAE {mae.toFixed(4)}</span> : null;
}

/** Open/close state for <details>, keyed by element id. */
function useOpenSet(initial: string[]) {
  const [open, setOpen] = useState<Set<string>>(() => new Set(initial));
  const toggle = (id: string) => (e: React.SyntheticEvent<HTMLDetailsElement>) => {
    const isOpen = e.currentTarget.open;
    setOpen((s) => { const n = new Set(s); if (isOpen) n.add(id); else n.delete(id); return n; });
  };
  const add = (...ids: string[]) => setOpen((s) => { const n = new Set(s); ids.forEach((i) => n.add(i)); return n; });
  return { open, toggle, add };
}

function RunCard({ r, open, onToggle, track }: {
  r: Report; open: boolean; onToggle: (e: React.SyntheticEvent<HTMLDetailsElement>) => void; track: Track;
}) {
  return (
    <details id={`report-${r.id}`} className="ev-run" open={open} onToggle={onToggle}>
      <summary>
        <span className="ev-report-date">{fmtDate(r.date)}{r.time ? ` · ${r.time}` : ""}</span>
        <span className="ev-kind">{r.kind}</span>
        <span className="ev-report-title">{r.model ?? r.title.replace(/^Eval /, "")}</span>
        <Score track={track} ndcg10={r.ndcg10} mae={r.mae} />
      </summary>
      {open && <div className="ev-report-body"><Markdown text={body(r, track)} heat /></div>}
    </details>
  );
}

export function EvaluationPage({ track, setTrack }: { track: Track; setTrack: (t: Track) => void }) {
  const champion = d.groups.find((g) => g.champion);
  const { open, toggle, add } = useOpenSet(champion ? [`version-${champion.id}`] : []);
  const [focus, setFocus] = useState<string | null>(null);

  useEffect(() => {
    if (!focus) return;
    const el = document.getElementById(focus);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
    el?.classList.add("cl-flash");
    const t = window.setTimeout(() => el?.classList.remove("cl-flash"), 1600);
    return () => window.clearTimeout(t);
  }, [focus]);

  // "View report" from the chart: open the version's card (and the run inside it, if it's not the main report).
  const viewReport = (id: string) => {
    const g = d.groups.find((x) => x.main === id || x.others.includes(id));
    if (!g) return;
    const target = g.main === id ? `version-${g.id}` : `report-${id}`;
    add(`version-${g.id}`, target);
    setFocus(null);
    window.setTimeout(() => setFocus(target), 0);
  };
  const first = d.versions[0], last = d.versions[d.versions.length - 1];
  const gain = first && last ? last.metrics["-1"]["ndcg@10"] / first.metrics["-1"]["ndcg@10"] - 1 : 0;
  const maeOf = (v?: Version) => v?.rmetrics?.["-1"]?.mae;
  const mae0 = maeOf(first), mae1 = maeOf(last);
  const bookAvg = d.rating_references.find((r) => r.key === "book_avg")?.metrics["-1"]?.mae;

  return (
    <article className="evaluation">
      <h1>Evaluation Timeline</h1>
      <div className="seg ev-track" role="tablist" aria-label="Evaluation track">
        {([["ranking", "Ranking"], ["rating", "Rating prediction"]] as [Track, string][]).map(([t, label]) => (
          <button key={t} type="button" role="tab" aria-selected={track === t} className={track === t ? "on" : ""}
                  onClick={() => setTrack(t)}>{label}</button>
        ))}
      </div>
      {track === "ranking" ? (
      <p className="lead-left">
        Every change to the recommender is measured the same way: 10,000 readers were set aside and never used for
        training, and for each one the most recent 30% of the books they read are hidden. The question is whether the
        books they went on to love (4–5★) show up near the top of their recommendations, given everything they read
        before. The main score is NDCG@10; higher is better. Since launch, full-history NDCG@10 has gone from{" "}
        {first?.metrics["-1"]["ndcg@10"].toFixed(3)} to {last?.metrics["-1"]["ndcg@10"].toFixed(3)} ({gain >= 0 ? "+" : ""}
        {(100 * gain).toFixed(0)}%).
      </p>
      ) : (
      <p className="lead-left">
        Every book card shows a predicted star rating. This track asks how close that prediction is to the rating
        the reader actually gave, for every book they read after their split date (not only the ones they loved),
        on the same readers and hidden books as the ranking track. Each prediction uses only the ratings the reader
        had made before. The main score is MAE, the average miss in stars; lower is better.
        {mae0 != null && mae1 != null && <>
          {" "}Since launch, full-history MAE has gone from {mae0.toFixed(3)} to {mae1.toFixed(3)}
          {" "}({mae1 <= mae0 ? "−" : "+"}{Math.abs(100 * (mae1 / mae0 - 1)).toFixed(Math.abs(mae1 / mae0 - 1) < 0.01 ? 1 : 0)}%)
          {bookAvg != null && <>; predicting each book&apos;s average rating scores {bookAvg.toFixed(3)}</>}.
        </>}
        {" "}The first two versions used an earlier rating model, reconstructed for this test on today&apos;s data.
      </p>
      )}

      <VersionChart versions={d.versions} references={track === "ranking" ? d.references : d.rating_references}
                    buckets={d.buckets} onViewReport={viewReport} track={track} />

      <h2>Evaluation reports</h2>
      <p className="muted">
        One card per model version, newest version first, each with the test-set report that scores it. Other runs of
        the same version (confirmation runs, re-scorings and the validation sweeps run on it while tuning the next
        change) are inside its card. The earliest reports used a random holdout instead of the time-based split, so
        their numbers aren&apos;t comparable. Each card shows its report&apos;s {track === "ranking" ? "ranking" : "rating-prediction"}{" "}
        section; switch tracks at the top of the page.
      </p>
      <div className="ev-reports">
        {d.groups.map((g) => {
          const id = `version-${g.id}`;
          const main = g.main ? report[g.main] : null;
          return (
            <details key={g.id} id={id} className={`cl-card ev-report${g.champion ? " champion" : ""}`}
                     open={open.has(id)} onToggle={toggle(id)}>
              <summary>
                <span className="ev-report-date">{fmtDate(g.date)}</span>
                {g.champion && <span className="ev-kind champion">🏆 Current champion</span>}
                <span className="ev-report-title">{g.title}</span>
                {g.others.length > 0 && <span className="muted small">{g.main ? `+ ${g.others.length} other run${g.others.length === 1 ? "" : "s"}` : `${g.others.length} reports`}</span>}
                <Score track={track} ndcg10={g.ndcg10} mae={g.mae} />
              </summary>
              {open.has(id) && (
                <div className="ev-report-body">
                  {main && <Markdown text={body(main, track)} heat />}
                  {g.others.length > 0 && (
                    <>
                      {main && <h4 className="md-h2">Other runs of this version</h4>}
                      <div className="ev-runs">
                        {g.others.map((rid) => (
                          <RunCard key={rid} r={report[rid]} open={open.has(`report-${rid}`)} onToggle={toggle(`report-${rid}`)} track={track} />
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
            </details>
          );
        })}
      </div>
    </article>
  );
}
