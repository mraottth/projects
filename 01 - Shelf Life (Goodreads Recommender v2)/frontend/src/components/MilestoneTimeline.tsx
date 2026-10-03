import { Fragment, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Markdown } from "./Markdown";

/**
 * Visual timeline of major project changes (prompts/milestones.json) for the Changelog page. Milestones scroll
 * horizontally; hovering one previews its details in a side panel to the right (the timeline narrows to make room,
 * so nothing is covered), and clicking pins the panel until the milestone is clicked again, × or Escape. On phones
 * the panel is a bottom sheet. The section fills most of the viewport, leaving the top of the full changelog visible.
 */

export interface Milestone {
  id: string; date: string; icon: string; category: string; title: string; summary: string;
  prompts: string[]; decisions: string[]; commits: { short: string; subject: string; url: string }[];
}
interface PromptLite { id: string; kind: string; text: string; reply: string | null }

const KIND: Record<string, string> = {
  prompt: "Prompt", answer: "Answer to Claude's question", "plan-comment": "Comment on Claude's plan",
  "plan-feedback": "Feedback on Claude's plan",
};
const fmtDay = (d: string) => new Date(`${d}T12:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });
const fmtWeekday = (d: string) => new Date(`${d}T12:00:00`).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
const dayGap = (a: string, b: string) => Math.round((new Date(b).getTime() - new Date(a).getTime()) / 86_400_000);

export function MilestoneTimeline({ milestones, prompts, categories, onShowPrompt, onShowDecision }: {
  milestones: Milestone[]; prompts: Map<string, PromptLite>; categories: Record<string, string>;
  onShowPrompt: (id: string) => void; onShowDecision: (id: string) => void;
}) {
  const [openId, setOpenId] = useState<string | null>(null);
  const [pinned, setPinned] = useState(false);
  const [edges, setEdges] = useState({ left: false, right: true });
  const [progress, setProgress] = useState({ left: 0, width: 100 });   // scroll position, in % of the track
  const [height, setHeight] = useState<number | undefined>(undefined);
  const section = useRef<HTMLElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLElement>(null);
  const nodes = useRef(new Map<string, HTMLButtonElement>());
  const closeTimer = useRef<number | undefined>(undefined);
  const openTimer = useRef<number | undefined>(undefined);

  // Fill most of the viewport, leaving ~90 px so the top of the full changelog peeks in below.
  useLayoutEffect(() => {
    const fit = () => {
      const el = section.current;
      if (!el) return;
      const top = el.getBoundingClientRect().top + window.scrollY;
      setHeight(Math.round(Math.min(Math.max(window.innerHeight - top - 90, 380), 760)));
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);

  const updateEdges = () => {
    const el = scroller.current;
    if (!el) return;
    setEdges({ left: el.scrollLeft > 4, right: el.scrollLeft + el.clientWidth < el.scrollWidth - 4 });
    setProgress({ left: (el.scrollLeft / el.scrollWidth) * 100, width: Math.min(100, (el.clientWidth / el.scrollWidth) * 100) });
  };
  useEffect(() => {
    updateEdges();
    const ro = new ResizeObserver(updateEdges);    // also fires as the panel opens and the timeline narrows
    if (scroller.current) ro.observe(scroller.current);
    return () => ro.disconnect();
  }, []);
  const scrollBy = (dir: number) => scroller.current?.scrollBy({ left: dir * scroller.current.clientWidth * 0.8, behavior: "smooth" });
  // Progress bar: click or drag anywhere on it to scroll the timeline to that point.
  const seek = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = scroller.current;
    if (!el) return;
    const bar = e.currentTarget.getBoundingClientRect();
    const frac = Math.min(Math.max((e.clientX - bar.left) / bar.width, 0), 1);
    el.scrollLeft = frac * el.scrollWidth - el.clientWidth / 2;
  };

  // When a milestone opens, start its details at the top, and once the panel has finished opening, scroll the
  // (now narrower) timeline so the milestone isn't hidden under the faded edges.
  const wasOpen = useRef(false);
  useEffect(() => {
    const opening = !wasOpen.current;
    wasOpen.current = !!openId;
    if (!openId) return;
    panel.current?.scrollTo({ top: 0 });
    const reveal = () => {
      const el = scroller.current, node = nodes.current.get(openId);
      if (!el || !node) return;
      const a = node.getBoundingClientRect(), b = el.getBoundingClientRect();
      const margin = 72;
      if (a.left < b.left + margin) el.scrollBy({ left: a.left - b.left - margin, behavior: "smooth" });
      else if (a.right > b.right - margin) el.scrollBy({ left: a.right - b.right + margin, behavior: "smooth" });
    };
    if (!opening) { reveal(); return; }
    const p = panel.current;
    const t = window.setTimeout(reveal, 450);    // fallback: no transition (e.g. the phone bottom sheet)
    const onEnd = (e: TransitionEvent) => { if (e.target === p && e.propertyName === "flex-basis") { window.clearTimeout(t); reveal(); } };
    p?.addEventListener("transitionend", onEnd);
    return () => { p?.removeEventListener("transitionend", onEnd); window.clearTimeout(t); };
  }, [openId]);

  const close = () => { window.clearTimeout(closeTimer.current); window.clearTimeout(openTimer.current); setOpenId(null); setPinned(false); };
  useEffect(() => {
    if (!openId) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openId]);

  // Opening the panel reflows the timeline, so the first preview waits a moment: sweeping the mouse across the
  // timeline doesn't make it jump. Once open, other milestones preview immediately.
  const preview = (id: string) => {
    window.clearTimeout(closeTimer.current);
    window.clearTimeout(openTimer.current);
    if (pinned) return;
    if (openId) setOpenId(id);
    else openTimer.current = window.setTimeout(() => setOpenId(id), 180);
  };
  // Unpinned previews close shortly after the pointer leaves the whole section (timeline and panel).
  const leaveSoon = () => {
    window.clearTimeout(openTimer.current);
    if (pinned) return;
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setOpenId(null), 300);
  };
  const toggle = (id: string) => {
    if (pinned && openId === id) close();
    else { window.clearTimeout(closeTimer.current); window.clearTimeout(openTimer.current); setOpenId(id); setPinned(true); }
  };

  const m = milestones.find((x) => x.id === openId);
  let prevDate = "";

  return (
    <section className={`ms${m ? " open" : ""}`} aria-label="Project milestones" ref={section} style={{ height }}
             onPointerEnter={() => window.clearTimeout(closeTimer.current)}
             onPointerLeave={(e) => { if (e.pointerType === "mouse") leaveSoon(); }}>
      <div className="ms-head">
        <h2>Milestones</h2>
        <span className="muted small">Hover a milestone to see the prompts and commits behind it; click to keep it open. Scroll sideways for more →</span>
        <span className="ms-arrows">
          <button type="button" className="ghost small" aria-label="Earlier milestones" disabled={!edges.left} onClick={() => scrollBy(-1)}>‹</button>
          <button type="button" className="ghost small" aria-label="Later milestones" disabled={!edges.right} onClick={() => scrollBy(1)}>›</button>
        </span>
      </div>

      <div className="ms-body">
        <div className="ms-left">
          <div className="ms-progress" role="scrollbar" aria-controls="ms-scroller" aria-orientation="horizontal"
               aria-valuenow={Math.round(progress.left / Math.max(100 - progress.width, 1) * 100)} aria-valuemin={0} aria-valuemax={100}
               onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); seek(e); }}
               onPointerMove={(e) => { if (e.buttons) seek(e); }}>
            <div className="ms-thumb" style={{ left: `${progress.left}%`, width: `${progress.width}%` }} />
          </div>
          <div className="ms-viewport">
            <div id="ms-scroller" className={`ms-scroll${edges.left ? " fade-left" : ""}${edges.right ? " fade-right" : ""}`} ref={scroller} onScroll={updateEdges}>
              <ol className="ms-track">
                {milestones.map((x) => {
                  const newDay = x.date !== prevDate;
                  const gap = prevDate ? dayGap(prevDate, x.date) : 0;
                  prevDate = x.date;
                  return (
                    <Fragment key={x.id}>
                      {newDay && gap > 1 && <li className="ms-gap" aria-hidden="true"><span>{gap} days later</span></li>}
                      <li className={`ms-item${newDay ? " new-day" : ""}`}>
                        {newDay && <span className="ms-day">{fmtWeekday(x.date)}</span>}
                        <button type="button" className={`ms-node cl-${x.category}${openId === x.id ? " on" : ""}${openId === x.id && pinned ? " pinned" : ""}`}
                                ref={(el) => { if (el) nodes.current.set(x.id, el); else nodes.current.delete(x.id); }}
                                aria-expanded={openId === x.id} aria-controls="ms-panel" aria-pressed={openId === x.id && pinned}
                                // Hover only for a real mouse and focus only from the keyboard, so a tap on a touch
                                // screen goes straight to the click (pin) instead of an emulated hover.
                                onPointerEnter={(e) => { if (e.pointerType === "mouse") preview(x.id); }}
                                onPointerLeave={() => window.clearTimeout(openTimer.current)}
                                onFocus={(e) => { if (e.currentTarget.matches(":focus-visible")) preview(x.id); }}
                                onClick={() => toggle(x.id)}>
                          <span className="ms-dot" aria-hidden="true">{x.icon}</span>
                          <span className="ms-title">{x.title}</span>
                          <span className="ms-sum">{x.summary}</span>
                          <span className="ms-meta">
                            {x.prompts.length} prompt{x.prompts.length === 1 ? "" : "s"} · {x.commits.length} commit{x.commits.length === 1 ? "" : "s"}
                          </span>
                        </button>
                      </li>
                    </Fragment>
                  );
                })}
              </ol>
            </div>
            {/* Blurred edges where there's more to scroll to. */}
            <div className={`ms-blur left${edges.left ? " on" : ""}`} aria-hidden="true" />
            <div className={`ms-blur right${edges.right ? " on" : ""}`} aria-hidden="true" />
          </div>
        </div>

        <aside id="ms-panel" ref={panel} className={`ms-panel${m ? ` open cl-${m.category}` : ""}`}
               aria-label={m ? `${m.title}: details` : "Milestone details"} aria-hidden={!m}>
          {m && (
            <>
              <div className="ms-pop-head">
                <span className="ms-pop-icon" aria-hidden="true">{m.icon}</span>
                <div>
                  <h3>{m.title}</h3>
                  <span className="muted small">{fmtDay(m.date)} · </span>
                  <span className={`cl-chip cl-${m.category}`}>{categories[m.category] ?? m.category}</span>
                  <span className="muted small ms-pin">{pinned ? " · 📌 pinned" : " · click the milestone to pin"}</span>
                </div>
                <button type="button" className="ms-close" aria-label="Close" onClick={close}>×</button>
              </div>
              <p className="ms-summary">{m.summary}</p>

              <h4>{m.prompts.length === 1 ? "The prompt" : `The ${m.prompts.length} prompts`} behind it</h4>
              <ol className="ms-prompts">
                {m.prompts.map((id) => {
                  const p = prompts.get(id);
                  if (!p) return null;
                  return (
                    <li key={id}>
                      <span className="cl-kind small">{KIND[p.kind] ?? "Prompt"}</span>
                      <p className="ms-ptext">{p.text}</p>
                      <div className="ms-prow">
                        {p.reply && (
                          <details className="cl-details">
                            <summary>Claude&apos;s reply</summary>
                            <Markdown text={p.reply} />
                          </details>
                        )}
                        <button type="button" className="link ms-show" onClick={() => { close(); onShowPrompt(id); }}>Show in changelog ↓</button>
                      </div>
                    </li>
                  );
                })}
              </ol>

              {m.commits.length > 0 && (
                <>
                  <h4>Commits</h4>
                  <ul className="ms-commits">
                    {m.commits.map((c) => (
                      <li key={c.short}><a className="cl-hash" href={c.url} target="_blank" rel="noreferrer">{c.short}</a> {c.subject}</li>
                    ))}
                  </ul>
                </>
              )}
              {m.decisions.length > 0 && (
                <>
                  <h4>Decisions</h4>
                  <div className="ms-pop-foot">
                    {m.decisions.map((d) => (
                      <button key={d} type="button" className="cl-chip-btn" onClick={() => { close(); onShowDecision(d); }}>{d}</button>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </aside>
      </div>
    </section>
  );
}
