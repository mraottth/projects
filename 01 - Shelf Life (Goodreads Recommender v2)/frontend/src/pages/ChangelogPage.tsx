import { useEffect, useMemo, useState } from "react";
import data from "../changelog.json";
import { Markdown } from "../components/Markdown";

/**
 * How Shelf Life was built: every prompt given to Claude Code, Claude's replies, the commits they produced and the
 * decisions along the way. Data: frontend/src/changelog.json, built by scripts/build_changelog.py from git,
 * prompts/ and DECISIONS.md.
 */

interface Commit {
  hash: string; short: string; time: string; subject: string; body: string; files: number;
  web_session: boolean; url: string; categories: string[]; prompt: string | null;
}
interface Prompt {
  id: string; time: string; kind: "prompt" | "answer" | "plan-comment" | "plan-feedback"; text: string;
  reply: string | null; asked?: { question: string; options: string[] }[]; plan?: string; categories: string[]; commits: string[];
}
interface Decision {
  id: string; title: string; date: string; categories: string[]; prompts: string[]; commits: string[];
  sections: { label: string; text: string }[];
}
interface Changelog { repo: string; categories: Record<string, string>; commits: Commit[]; prompts: Prompt[]; decisions: Decision[] }

const CL = data as Changelog;
const CATS: Record<string, string> = { ...CL.categories, uncategorized: "Uncategorized" };
const KIND: Record<Prompt["kind"], string> = {
  prompt: "Prompt", answer: "Answer to Claude's question", "plan-comment": "Comment on Claude's plan",
  "plan-feedback": "Feedback on Claude's plan",
};
const commitByShort = new Map(CL.commits.map((c) => [c.short, c]));
const fmtTime = (t: string) => new Date(t).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const fmtDay = (t: string) => new Date(t).toLocaleDateString(undefined, { weekday: "short", month: "long", day: "numeric", year: "numeric" });
const dayKey = (t: string) => new Date(t).toDateString();

type Mode = "timeline" | "category" | "decisions";

function Chips({ cats }: { cats: string[] }) {
  return <span className="cl-chips">{cats.map((c) => <span key={c} className={`cl-chip cl-${c}`}>{CATS[c] ?? c}</span>)}</span>;
}

function CommitRow({ c }: { c: Commit }) {
  return (
    <div className="cl-commit">
      <div className="cl-commit-head">
        <a className="cl-hash" href={c.url} target="_blank" rel="noreferrer">{c.short}</a>
        <span className="cl-subject">{c.subject}</span>
      </div>
      <div className="cl-meta muted small">
        {fmtTime(c.time)} · {c.files} file{c.files === 1 ? "" : "s"}
        {c.web_session && <span className="cl-badge" title="Made in a Claude Code on the web session">Claude Code on the web</span>}
        <Chips cats={c.categories} />
      </div>
      {c.body && (
        <details className="cl-details">
          <summary>Commit message</summary>
          <Markdown text={c.body} />
        </details>
      )}
    </div>
  );
}

function PromptCard({ p, showCommits = true }: { p: Prompt; showCommits?: boolean }) {
  const commits = p.commits.map((s) => commitByShort.get(s)).filter((c): c is Commit => !!c);
  return (
    <article id={p.id} className="cl-card">
      <header className="cl-card-head">
        <span className={`cl-kind${p.kind === "prompt" ? "" : " alt"}`}>{KIND[p.kind]}</span>
        <span className="muted small">{fmtTime(p.time)} · {p.id}</span>
        <Chips cats={p.categories} />
      </header>
      <div className="cl-text">{p.text}</div>
      {p.asked && (
        <div className="cl-asked">
          <span className="muted small">Claude asked</span>
          <ul>{p.asked.map((q, i) => <li key={i}>{q.question} <span className="muted">({q.options.join(" / ")})</span></li>)}</ul>
        </div>
      )}
      {p.reply && (
        <details className="cl-details">
          <summary>Claude&apos;s reply</summary>
          <Markdown text={p.reply} />
        </details>
      )}
      {p.plan && (
        <details className="cl-details">
          <summary>Claude&apos;s plan (approved)</summary>
          <Markdown text={p.plan} />
        </details>
      )}
      {showCommits && commits.length > 0 && (
        <div className="cl-commits">
          <span className="muted small">{commits.length === 1 ? "Commit" : `${commits.length} commits`}</span>
          {commits.map((c) => <CommitRow key={c.hash} c={c} />)}
        </div>
      )}
      {!showCommits && p.commits.length > 0 && (
        <p className="muted small">Commits: {p.commits.map((s) => <a key={s} className="cl-hash" href={commitByShort.get(s)?.url}>{s} </a>)}</p>
      )}
    </article>
  );
}

function DecisionCard({ d, onPrompt }: { d: Decision; onPrompt: (id: string) => void }) {
  return (
    <article id={d.id} className="cl-card">
      <header className="cl-card-head">
        <span className="cl-kind">{d.id}</span>
        <span className="muted small">{d.date}</span>
        <Chips cats={d.categories} />
      </header>
      <h3 className="cl-title">{d.title}</h3>
      {d.sections.map((s) => (
        <div key={s.label} className="cl-section">
          <strong>{s.label}.</strong> <Markdown text={s.text} />
        </div>
      ))}
      {(d.prompts.length > 0 || d.commits.length > 0) && (
        <p className="cl-refs muted small">
          {d.prompts.length > 0 && <>Prompts: {d.prompts.map((p) => (
            <button key={p} type="button" className="link" onClick={() => onPrompt(p)}>{p.split("-").pop()}</button>))}</>}
          {d.commits.length > 0 && <> · Commits: {d.commits.map((s) => (
            <a key={s} className="cl-hash" href={commitByShort.get(s)?.url ?? `${CL.repo}/commit/${s}`} target="_blank" rel="noreferrer">{s}</a>))}</>}
        </p>
      )}
    </article>
  );
}

export function ChangelogPage() {
  const [mode, setMode] = useState<Mode>("timeline");
  const [cats, setCats] = useState<string[]>([]);          // empty = all categories
  const [q, setQ] = useState("");
  const [newest, setNewest] = useState(true);
  const [focus, setFocus] = useState<string | null>(null);

  const match = (itemCats: string[], ...texts: (string | null | undefined)[]) =>
    (cats.length === 0 || itemCats.some((c) => cats.includes(c))) &&
    (!q.trim() || texts.some((t) => t?.toLowerCase().includes(q.trim().toLowerCase())));

  const prompts = useMemo(() => CL.prompts.filter((p) => match(p.categories, p.text, p.reply, p.plan, p.id)),
    [cats, q]); // eslint-disable-line react-hooks/exhaustive-deps
  const commits = useMemo(() => CL.commits.filter((c) => match(c.categories, c.subject, c.body, c.short)),
    [cats, q]); // eslint-disable-line react-hooks/exhaustive-deps
  const decisions = useMemo(() => CL.decisions.filter((d) => match(d.categories, d.title, ...d.sections.map((s) => s.text))),
    [cats, q]); // eslint-disable-line react-hooks/exhaustive-deps

  // Timeline: prompts (with their commits nested) plus commits not tied to a prompt shown here.
  const timeline = useMemo(() => {
    const shownPrompts = new Set(prompts.map((p) => p.id));
    const items: ({ kind: "prompt"; p: Prompt; t: string } | { kind: "commit"; c: Commit; t: string })[] = [
      ...prompts.map((p) => ({ kind: "prompt" as const, p, t: p.time })),
      ...commits.filter((c) => !c.prompt || !shownPrompts.has(c.prompt)).map((c) => ({ kind: "commit" as const, c, t: c.time })),
    ];
    items.sort((a, b) => (newest ? -1 : 1) * (new Date(a.t).getTime() - new Date(b.t).getTime()));
    return items;
  }, [prompts, commits, newest]);

  const counts = useMemo(() => {
    const n: Record<string, number> = {};
    for (const x of [...CL.prompts, ...CL.commits]) for (const c of x.categories) n[c] = (n[c] ?? 0) + 1;
    return n;
  }, []);

  // Jump from a decision to its prompt in the timeline.
  useEffect(() => {
    if (!focus || mode !== "timeline") return;
    const el = document.getElementById(focus);
    if (el) { el.scrollIntoView({ behavior: "smooth", block: "start" }); el.classList.add("cl-flash"); }
    const t = setTimeout(() => { el?.classList.remove("cl-flash"); setFocus(null); }, 2000);
    return () => clearTimeout(t);
  }, [focus, mode]);
  const goPrompt = (id: string) => { setCats([]); setQ(""); setMode("timeline"); setFocus(id); };

  const first = CL.prompts[0]?.time ?? CL.commits[0]?.time;
  const toggleCat = (c: string) => setCats((cs) => (cs.includes(c) ? cs.filter((x) => x !== c) : [...cs, c]));
  let lastDay = "";

  return (
    <div className="changelog">
      <h1>How Shelf Life was built</h1>
      <p className="lead-left">
        Shelf Life was built in conversation with Claude Code. This page logs every prompt, Claude&apos;s replies, the
        commits they produced and the decisions along the way: {CL.prompts.length} prompts, {CL.commits.length} commits
        and {CL.decisions.length} decisions{first ? ` since ${new Date(first).toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" })}` : ""}.
        The same records are in the repo as <a href={`${CL.repo}/tree/main/01%20-%20Shelf%20Life%20%28Goodreads%20Recommender%20v2%29/prompts`} target="_blank" rel="noreferrer">prompts/</a>{" "}
        and <a href={`${CL.repo}/blob/main/01%20-%20Shelf%20Life%20%28Goodreads%20Recommender%20v2%29/DECISIONS.md`} target="_blank" rel="noreferrer">DECISIONS.md</a>.
      </p>

      <div className="cl-controls">
        <div className="seg" role="tablist" aria-label="View">
          {([["timeline", "Timeline"], ["category", "By category"], ["decisions", "Decisions"]] as [Mode, string][]).map(([m, label]) => (
            <button key={m} type="button" role="tab" aria-selected={mode === m} className={mode === m ? "on" : ""} onClick={() => setMode(m)}>{label}</button>
          ))}
        </div>
        <input className="cl-search" type="search" placeholder="Search prompts, replies, commits…" value={q} onChange={(e) => setQ(e.target.value)} />
        {mode === "timeline" && (
          <button type="button" className="ghost small" onClick={() => setNewest((n) => !n)}>{newest ? "Newest first" : "Oldest first"} ⇅</button>
        )}
      </div>
      <div className="cl-filter">
        <button type="button" className={`cl-chip-btn${cats.length === 0 ? " on" : ""}`} onClick={() => setCats([])}>All</button>
        {Object.keys(CL.categories).map((c) => (
          <button key={c} type="button" className={`cl-chip-btn cat cl-${c}${cats.includes(c) ? " on" : ""}`} aria-pressed={cats.includes(c)}
                  onClick={() => toggleCat(c)}>
            {CATS[c]} <span className="muted">{counts[c] ?? 0}</span>
          </button>
        ))}
      </div>

      {mode === "timeline" && (
        <div className="cl-list">
          {timeline.length === 0 && <p className="muted">Nothing matches.</p>}
          {timeline.map((it) => {
            const d = dayKey(it.t);
            const header = d !== lastDay ? <h2 className="cl-day">{fmtDay(it.t)}</h2> : null;
            lastDay = d;
            return (
              <div key={it.kind === "prompt" ? it.p.id : it.c.hash}>
                {header}
                {it.kind === "prompt" ? <PromptCard p={it.p} /> : <div className="cl-card cl-standalone"><CommitRow c={it.c} /></div>}
              </div>
            );
          })}
        </div>
      )}

      {mode === "category" && (
        <div className="cl-list">
          {[...Object.keys(CL.categories), "uncategorized"].map((c) => {
            if (cats.length && !cats.includes(c)) return null;
            const ps = prompts.filter((p) => p.categories.includes(c));
            const cs = commits.filter((x) => x.categories.includes(c));
            if (!ps.length && !cs.length) return null;
            const items = [...ps.map((p) => ({ t: p.time, el: <PromptCard key={p.id} p={p} showCommits={false} /> })),
              ...cs.map((x) => ({ t: x.time, el: <div key={x.hash} className="cl-card cl-standalone"><CommitRow c={x} /></div> }))]
              .sort((a, b) => new Date(b.t).getTime() - new Date(a.t).getTime());
            return (
              <details key={c} className="cl-group" open={cats.length > 0}>
                <summary><h2>{CATS[c]}</h2><span className="muted">{ps.length} prompts · {cs.length} commits</span></summary>
                {items.map((x) => x.el)}
              </details>
            );
          })}
        </div>
      )}

      {mode === "decisions" && (
        <div className="cl-list">
          {decisions.length === 0 && <p className="muted">Nothing matches.</p>}
          {decisions.map((d) => <DecisionCard key={d.id} d={d} onPrompt={goPrompt} />)}
        </div>
      )}
    </div>
  );
}
