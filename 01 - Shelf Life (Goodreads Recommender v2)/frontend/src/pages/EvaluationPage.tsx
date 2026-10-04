import { useEffect, useState } from "react";
import data from "../evaluations.json";
import { Markdown } from "../components/Markdown";
import { VersionChart, type Reference, type Version } from "../components/VersionChart";

/**
 * Evaluation: how the recommender improved, version by version (chart), and every evaluation report
 * (eval/reports/*.md) as an expandable card, the current champion's first and open. Data: scripts/build_evaluations.py.
 */

interface Report {
  id: string; date: string; time: string; kind: string; title: string; model: string | null; set: string | null;
  n_users: number | null; split_hash: string | null; ndcg10: number | null; champion: boolean; markdown: string;
}
interface Evaluations { versions: Version[]; references: Reference[]; reports: Report[]; buckets: number[] }

const d = data as unknown as Evaluations;
const fmtDate = (s: string) => new Date(`${s}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });

export function EvaluationPage() {
  const champion = d.reports.find((r) => r.champion);
  const [open, setOpen] = useState<Set<string>>(() => new Set(champion ? [champion.id] : []));
  const [focus, setFocus] = useState<string | null>(null);

  useEffect(() => {
    if (!focus) return;
    const el = document.getElementById(`report-${focus}`);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
    el?.classList.add("cl-flash");
    const t = window.setTimeout(() => el?.classList.remove("cl-flash"), 1600);
    return () => window.clearTimeout(t);
  }, [focus]);

  const viewReport = (id: string) => {
    setOpen((s) => new Set(s).add(id));
    setFocus(null);
    window.setTimeout(() => setFocus(id), 0);
  };
  const first = d.versions[0], last = d.versions[d.versions.length - 1];
  const gain = first && last ? last.metrics["-1"]["ndcg@10"] / first.metrics["-1"]["ndcg@10"] - 1 : 0;

  return (
    <article className="evaluation">
      <h1>How the model improved</h1>
      <p className="lead-left">
        Every change to the recommender is measured the same way: 10,000 readers were set aside and never used for
        training, and for each one the most recent 30% of the books they read are hidden. The question is whether the
        books they went on to love (4–5★) show up near the top of their recommendations, given everything they read
        before. The main score is NDCG@10; higher is better. Since launch, full-history NDCG@10 has gone from{" "}
        {first?.metrics["-1"]["ndcg@10"].toFixed(3)} to {last?.metrics["-1"]["ndcg@10"].toFixed(3)} ({gain >= 0 ? "+" : ""}
        {(100 * gain).toFixed(0)}%). The first two versions predate this test, so they were re-scored with their own
        settings on today&apos;s data.
      </p>

      <VersionChart versions={d.versions} references={d.references} buckets={d.buckets} onViewReport={viewReport} />

      <h2>Evaluation reports</h2>
      <p className="muted">
        Every run, newest first, with the current champion&apos;s report on top. Test-set reports score a version
        against the baselines; validation runs are the tuning sweeps behind each change; the earliest reports used a
        random holdout instead of the time-based split, so their numbers aren&apos;t comparable.
      </p>
      <div className="ev-reports">
        {d.reports.map((r) => (
          <details key={r.id} id={`report-${r.id}`} className={`cl-card ev-report${r.champion ? " champion" : ""}`}
                   open={open.has(r.id)}
                   onToggle={(e) => {
                     const isOpen = (e.currentTarget as HTMLDetailsElement).open;
                     setOpen((s) => { const n = new Set(s); if (isOpen) n.add(r.id); else n.delete(r.id); return n; });
                   }}>
            <summary>
              <span className="ev-report-date">{fmtDate(r.date)}{r.time ? ` · ${r.time}` : ""}</span>
              <span className={`ev-kind${r.champion ? " champion" : ""}`}>{r.champion ? "🏆 Current champion" : r.kind}</span>
              <span className="ev-report-title">{r.model ?? r.title.replace(/^Eval /, "")}</span>
              {r.ndcg10 != null && <span className="ev-report-score">NDCG@10 {r.ndcg10.toFixed(4)}</span>}
            </summary>
            {open.has(r.id) && <div className="ev-report-body"><Markdown text={r.markdown} /></div>}
          </details>
        ))}
      </div>
    </article>
  );
}
