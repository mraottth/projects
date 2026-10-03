import { Fragment, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Markdown } from "./Markdown";

/**
 * Visual timeline of major project changes (prompts/milestones.json) for the Changelog page. Nodes scroll
 * horizontally; hovering (or tapping / focusing) a node opens a popover with the prompts and replies behind it,
 * its commits and decisions. The popover renders in a portal with fixed positioning so the scroller can't clip it.
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
const PHONE = "(max-width: 640px)";

export function MilestoneTimeline({ milestones, prompts, categories, onShowPrompt, onShowDecision }: {
  milestones: Milestone[]; prompts: Map<string, PromptLite>; categories: Record<string, string>;
  onShowPrompt: (id: string) => void; onShowDecision: (id: string) => void;
}) {
  const [openId, setOpenId] = useState<string | null>(null);
  const [pinned, setPinned] = useState(false);
  const [pos, setPos] = useState<React.CSSProperties>({});
  const [edges, setEdges] = useState({ left: false, right: true });
  const [progress, setProgress] = useState({ left: 0, width: 100 });   // scroll position, in % of the track
  const scroller = useRef<HTMLDivElement>(null);
  const section = useRef<HTMLElement>(null);
  const [height, setHeight] = useState<number | undefined>(undefined);
  const popRef = useRef<HTMLDivElement>(null);
  const nodes = useRef(new Map<string, HTMLButtonElement>());
  const closeTimer = useRef<number | undefined>(undefined);

  const updateEdges = () => {
    const el = scroller.current;
    if (!el) return;
    setEdges({ left: el.scrollLeft > 4, right: el.scrollLeft + el.clientWidth < el.scrollWidth - 4 });
    setProgress({ left: (el.scrollLeft / el.scrollWidth) * 100, width: Math.min(100, (el.clientWidth / el.scrollWidth) * 100) });
  };
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
  useEffect(() => {
    updateEdges();
    const ro = new ResizeObserver(updateEdges);
    if (scroller.current) ro.observe(scroller.current);
    return () => ro.disconnect();
  }, []);
  // Progress bar: click or drag anywhere on it to scroll the timeline to that point.
  const seek = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = scroller.current;
    if (!el) return;
    const bar = e.currentTarget.getBoundingClientRect();
    const frac = Math.min(Math.max((e.clientX - bar.left) / bar.width, 0), 1);
    el.scrollLeft = frac * el.scrollWidth - el.clientWidth / 2;
  };
  const scrollBy = (dir: number) => scroller.current?.scrollBy({ left: dir * scroller.current.clientWidth * 0.8, behavior: "smooth" });

  // Place the popover beside its node: below if there's room, else above; a bottom sheet on phones.
  const place = useCallback(() => {
    if (!openId) return;
    const node = nodes.current.get(openId);
    if (!node) return;
    if (window.matchMedia(PHONE).matches) { setPos({ left: 0, right: 0, bottom: 0, maxHeight: "70vh" }); return; }
    // Beside the milestone (right if it fits, else left), top-aligned with it and kept on screen.
    const r = node.getBoundingClientRect();
    const w = Math.min(440, window.innerWidth - 24);
    const left = r.right + 8 + w <= window.innerWidth - 12 ? r.right + 8 : Math.max(12, r.left - 8 - w);
    const top = Math.min(Math.max(r.top, 72), window.innerHeight - 320);
    setPos({ left, width: w, top, maxHeight: window.innerHeight - top - 12 });
  }, [openId]);
  useLayoutEffect(place, [place]);
  useEffect(() => {
    if (!openId) return;
    const onMove = () => place();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { setOpenId(null); setPinned(false); } };
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!popRef.current?.contains(t) && ![...nodes.current.values()].some((n) => n.contains(t))) { setOpenId(null); setPinned(false); }
    };
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("scroll", onMove, true); window.removeEventListener("resize", onMove);
      window.removeEventListener("keydown", onKey); window.removeEventListener("pointerdown", onDown);
    };
  }, [openId, place]);

  const show = (id: string) => { window.clearTimeout(closeTimer.current); if (!pinned || openId === id) setOpenId(id); };
  const hideSoon = () => {
    if (pinned) return;
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setOpenId(null), 200);
  };
  const toggle = (id: string) => {
    if (openId === id && pinned) { setOpenId(null); setPinned(false); }
    else { setOpenId(id); setPinned(true); }
  };

  const m = milestones.find((x) => x.id === openId);
  let prevDate = "";

  return (
    <section className="ms" aria-label="Project milestones" ref={section} style={{ height }}>
      <div className="ms-head">
        <h2>Milestones</h2>
        <span className="muted small">Hover or tap a milestone to see the prompts and commits behind it. Scroll sideways for more →</span>
        <span className="ms-arrows">
          <button type="button" className="ghost small" aria-label="Earlier milestones" disabled={!edges.left} onClick={() => scrollBy(-1)}>‹</button>
          <button type="button" className="ghost small" aria-label="Later milestones" disabled={!edges.right} onClick={() => scrollBy(1)}>›</button>
        </span>
      </div>
      <div className="ms-progress" role="scrollbar" aria-controls="ms-scroller" aria-orientation="horizontal"
           aria-valuenow={Math.round(progress.left / Math.max(100 - progress.width, 1) * 100)} aria-valuemin={0} aria-valuemax={100}
           onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); seek(e); }}
           onPointerMove={(e) => { if (e.buttons) seek(e); }}>
        <div className="ms-thumb" style={{ left: `${progress.left}%`, width: `${progress.width}%` }} />
      </div>
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
                  <button type="button" className={`ms-node cl-${x.category}${openId === x.id ? " on" : ""}`}
                          ref={(el) => { if (el) nodes.current.set(x.id, el); else nodes.current.delete(x.id); }}
                          aria-expanded={openId === x.id} aria-controls="ms-pop"
                          // Hover only for a real mouse and focus only from the keyboard: on touch screens the emulated
                          // hover would open the sheet under the finger and swallow the tap.
                          onPointerEnter={(e) => { if (e.pointerType === "mouse") show(x.id); }}
                          onPointerLeave={(e) => { if (e.pointerType === "mouse") hideSoon(); }}
                          onFocus={(e) => { if (e.currentTarget.matches(":focus-visible")) show(x.id); }}
                          onBlur={hideSoon} onClick={() => toggle(x.id)}>
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

      {m && createPortal(
        <div id="ms-pop" ref={popRef} className={`ms-pop cl-${m.category}`} role="dialog" aria-label={m.title} style={pos}
             onPointerEnter={() => window.clearTimeout(closeTimer.current)}
             onPointerLeave={(e) => { if (e.pointerType === "mouse") hideSoon(); }}>
          <div className="ms-pop-head">
            <span className="ms-pop-icon" aria-hidden="true">{m.icon}</span>
            <div>
              <h3>{m.title}</h3>
              <span className="muted small">{fmtDay(m.date)} · </span>
              <span className={`cl-chip cl-${m.category}`}>{categories[m.category] ?? m.category}</span>
            </div>
            <button type="button" className="ms-close" aria-label="Close" onClick={() => { setOpenId(null); setPinned(false); }}>×</button>
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
                    <button type="button" className="link ms-show" onClick={() => { setOpenId(null); setPinned(false); onShowPrompt(id); }}>
                      Show in changelog ↓
                    </button>
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
                  <button key={d} type="button" className="cl-chip-btn" onClick={() => { setOpenId(null); setPinned(false); onShowDecision(d); }}>{d}</button>
                ))}
              </div>
            </>
          )}
        </div>,
        document.body,
      )}
    </section>
  );
}
