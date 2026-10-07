import { useEffect, useLayoutEffect, useRef, useState } from "react";
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
// Hovering a milestone must rest on it while a ring fills around its icon before its details show: SWITCH_MS to
// switch from an open milestone (so crossing milestones on the way to the panel doesn't change it), OPEN_MS to open
// the panel when it's closed (shorter, so the first look isn't slow, and sweeping across doesn't reflow the timeline).
const SWITCH_MS = 550;
const OPEN_MS = Math.round(SWITCH_MS * 0.75);

export function MilestoneTimeline({ milestones, prompts, categories, onShowPrompt, onShowDecision }: {
  milestones: Milestone[]; prompts: Map<string, PromptLite>; categories: Record<string, string>;
  onShowPrompt: (id: string) => void; onShowDecision: (id: string) => void;
}) {
  const [openId, setOpenId] = useState<string | null>(null);
  const [newestFirst, setNewestFirst] = useState(true);
  const shown = newestFirst ? [...milestones].reverse() : milestones;
  const [pinned, setPinned] = useState(false);
  const [pending, setPending] = useState<{ id: string; ms: number } | null>(null);   // hovered milestone and its wait
  const [ringDone, setRingDone] = useState(false);                 // its ring has filled and is fading out
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

  // After the order flips, start at the beginning of the reversed strip (the newest or the oldest milestone).
  const firstOrder = useRef(true);
  useLayoutEffect(() => {
    if (firstOrder.current) { firstOrder.current = false; return; }
    if (scroller.current) scroller.current.scrollLeft = 0;
    updateEdges();
  }, [newestFirst]);

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
  // (now narrower) timeline so the milestone isn't hidden under the faded edges. Hover switches don't scroll: that
  // milestone is under the mouse, and moving the timeline would put a different one there. Clicks and keyboard do.
  const wasOpen = useRef(false);
  const revealNext = useRef(false);
  const quietUntil = useRef(0);   // ignore hovers while the timeline scrolls itself
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
      const dx = a.left < b.left + margin ? a.left - b.left - margin : a.right > b.right - margin ? a.right - b.right + margin : 0;
      const to = Math.min(Math.max(el.scrollLeft + dx, 0), el.scrollWidth - el.clientWidth);
      if (Math.abs(to - el.scrollLeft) < 1) return;
      quietUntil.current = performance.now() + 600;
      el.scrollTo({ left: to, behavior: "smooth" });
    };
    const wanted = revealNext.current;
    revealNext.current = false;
    if (!opening) { if (wanted) reveal(); return; }
    const p = panel.current;
    const t = window.setTimeout(reveal, 450);    // fallback: no transition (e.g. the phone bottom sheet)
    const onEnd = (e: TransitionEvent) => { if (e.target === p && e.propertyName === "flex-basis") { window.clearTimeout(t); reveal(); } };
    p?.addEventListener("transitionend", onEnd);
    return () => { p?.removeEventListener("transitionend", onEnd); window.clearTimeout(t); };
  }, [openId]);

  const cancelPending = () => { window.clearTimeout(openTimer.current); setPending(null); setRingDone(false); };
  const close = () => { window.clearTimeout(closeTimer.current); cancelPending(); setOpenId(null); setPinned(false); };
  useEffect(() => {
    if (!openId) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openId]);

  // Hover waits OPEN_MS (panel closed) or SWITCH_MS (panel open) with a filling ring; keyboard focus shows at once.
  const preview = (id: string, immediate = false) => {
    window.clearTimeout(closeTimer.current);
    cancelPending();
    if (pinned || openId === id) return;
    if (immediate) { revealNext.current = true; setOpenId(id); return; }
    if (performance.now() < quietUntil.current) return;
    const ms = openId ? SWITCH_MS : OPEN_MS;
    setPending({ id, ms });
    openTimer.current = window.setTimeout(() => {
      setOpenId(id);
      setRingDone(true);   // the full ring fades out rather than vanishing as the details appear
      openTimer.current = window.setTimeout(() => { setPending(null); setRingDone(false); }, 260);
    }, ms);
  };
  // Unpinned previews close shortly after the pointer leaves the whole section (timeline and panel).
  const leaveSoon = () => {
    cancelPending();
    if (pinned) return;
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setOpenId(null), 300);
  };
  const toggle = (id: string) => {
    if (pinned && openId === id) close();
    else { window.clearTimeout(closeTimer.current); cancelPending(); revealNext.current = true; setOpenId(id); setPinned(true); }
  };

  const m = milestones.find((x) => x.id === openId);

  return (
    <section className={`ms${m ? " open" : ""}`} aria-label="Product milestones" ref={section} style={{ height }}
             onPointerEnter={() => window.clearTimeout(closeTimer.current)}
             onPointerLeave={(e) => { if (e.pointerType === "mouse") leaveSoon(); }}>
      <div className="ms-head">
        <h2>Product Milestones</h2>
        <span className="muted small">Hover a milestone to see the prompts and commits behind it; click to keep it open. Scroll sideways for {newestFirst ? "older" : "newer"} ones →</span>
        <span className="ms-arrows">
          <label className="ms-order">Display{" "}
            <select value={newestFirst ? "newest" : "oldest"} onChange={(e) => setNewestFirst(e.target.value === "newest")}>
              <option value="oldest">Oldest first</option>
              <option value="newest">Newest first</option>
            </select>
          </label>
          <button type="button" className="ghost small" aria-label={newestFirst ? "Newer milestones" : "Older milestones"}
                  disabled={!edges.left} onClick={() => scrollBy(-1)}>‹</button>
          <button type="button" className="ghost small" aria-label={newestFirst ? "Older milestones" : "Newer milestones"}
                  disabled={!edges.right} onClick={() => scrollBy(1)}>›</button>
        </span>
      </div>

      <div className="ms-body">
        <div className="ms-left">
          <div className="ms-progress-row">
            <span className="ms-end" aria-hidden="true">{newestFirst ? "Newer" : "Older"}</span>
            <div className="ms-progress" role="scrollbar" aria-controls="ms-scroller" aria-orientation="horizontal"
                 aria-valuenow={Math.round(progress.left / Math.max(100 - progress.width, 1) * 100)} aria-valuemin={0} aria-valuemax={100}
                 onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); seek(e); }}
                 onPointerMove={(e) => { if (e.buttons) seek(e); }}>
              <div className="ms-thumb" style={{ left: `${progress.left}%`, width: `${progress.width}%` }} />
            </div>
            <span className="ms-end" aria-hidden="true">{newestFirst ? "Older" : "Newer"}</span>
          </div>
          <div className="ms-viewport">
            <div id="ms-scroller" className={`ms-scroll${edges.left ? " fade-left" : ""}${edges.right ? " fade-right" : ""}`} ref={scroller} onScroll={updateEdges}>
              <ol className="ms-track">
                {shown.map((x) => (
                  <li key={x.id} className="ms-item">
                    <button type="button" className={`ms-node cl-${x.category}${openId === x.id ? " on" : ""}${openId === x.id && pinned ? " pinned" : ""}`}
                            ref={(el) => { if (el) nodes.current.set(x.id, el); else nodes.current.delete(x.id); }}
                            aria-expanded={openId === x.id} aria-controls="ms-panel" aria-pressed={openId === x.id && pinned}
                            // Hover only for a real mouse and focus only from the keyboard, so a tap on a touch
                            // screen goes straight to the click (pin) instead of an emulated hover.
                            onPointerEnter={(e) => { if (e.pointerType === "mouse") preview(x.id); }}
                            onPointerLeave={(e) => { if (e.pointerType === "mouse") cancelPending(); }}
                            onFocus={(e) => { if (e.currentTarget.matches(":focus-visible")) preview(x.id, true); }}
                            onClick={() => toggle(x.id)}>
                      <span className="ms-num">{x.id.replace(/^M0*/, "M")}</span>
                      <span className="ms-dot" aria-hidden="true">
                        {x.icon}
                        {pending?.id === x.id && (
                          <svg className={`ms-ring${ringDone ? " done" : ""}`} viewBox="0 0 100 100" style={{ animationDuration: `${pending.ms}ms` }}>
                            <circle cx="50" cy="50" r="47" pathLength={100} />
                          </svg>
                        )}
                      </span>
                      <span className="ms-title">{x.title}</span>
                      <span className="ms-sum">{x.summary}</span>
                      <span className="ms-meta">
                        {x.prompts.length} prompt{x.prompts.length === 1 ? "" : "s"} · {x.commits.length} commit{x.commits.length === 1 ? "" : "s"}
                      </span>
                    </button>
                  </li>
                ))}
              </ol>
            </div>
            {/* Blurred edges where there's more to scroll to. */}
            <div className={`ms-blur left${edges.left ? " on" : ""}`} aria-hidden="true" />
            <div className={`ms-blur right${edges.right ? " on" : ""}`} aria-hidden="true" />
          </div>
        </div>

        {/* Phones: the details panel is a bottom sheet; tapping the dimmed page around it closes it. */}
        {m && <div className="ms-backdrop" aria-hidden="true" onClick={close} />}
        <aside id="ms-panel" ref={panel} className={`ms-panel${m ? ` open cl-${m.category}` : ""}`}
               aria-label={m ? `${m.title}: details` : "Milestone details"} aria-hidden={!m}>
          {m && (
            <div key={m.id} className="ms-panel-body">
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
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}
