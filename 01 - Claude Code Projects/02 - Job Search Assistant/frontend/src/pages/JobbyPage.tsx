import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  api,
  Application,
  APPLICATION_STATUS_LABELS,
  JobbyChatSummary,
  JobbyMessage,
  JobbyTarget,
  JobbyThread,
} from "../api";
import { JobbyLogo } from "../components/JobbyLogo";
import { Markdown } from "../components/Markdown";
import { Spinner } from "../components/Spinner";

// Jobby, the job-search sidekick (layout asked 2026-11-10). Where you are lives in the URL:
//   /jobby          home: Jobby's welcome, "Start a general chat" or "Choose from My Jobs"
//   ?pick=1         choosing one of My Jobs
//   ?job=42         that job's chat (My Jobs and the job page link straight here)
//   ?chat=7         any chat, reopened from the history on the left
//   ?new=1          a new general chat, created on its first message
// The left tray is the chat history. Chats are saved server-side (core/jobby.py) and only
// deleted from the history, never by going Back.

type Starter = { label: string; blurb: string; text: string };

// For a chat about one job.
const JOB_STARTERS: Starter[] = [
  {
    label: "Mock interview",
    blurb: "Practice with Jobby as your interviewer",
    text: "Let's do a mock interview for this role. You be the interviewer: ask me one question at a time, wait for my answer, and give me quick feedback before the next one.",
  },
  {
    label: "Likely questions",
    blurb: "What they'll probably ask, and how to answer",
    text: "What questions is the interviewer most likely to ask me for this role? Group them, flag the ones I should prepare most carefully given my background, and then offer to work through answers with me.",
  },
  {
    label: "Frame my background",
    blurb: "Your pitch and strongest stories for this role",
    text: "Help me frame my background for this role: draft a 60–90 second \"tell me about yourself\", map my strongest stories to what the posting asks for, and suggest honest ways to handle any gaps.",
  },
  {
    label: "Prep & research plan",
    blurb: "A checklist of what to do beforehand",
    text: "Give me a prep checklist for this interview: what to research about the company (look up anything recent and relevant), skills or examples to brush up on, and logistics to sort out beforehand.",
  },
];

// For a general chat about the search itself (asked 2026-11-10).
const GENERAL_STARTERS: Starter[] = [
  {
    label: "Find organizations",
    blurb: "Places worth watching that fit what you want",
    text: "Help me find organizations worth applying to. Based on my profile, what I'm looking for, and the jobs that score highest for me, suggest specific organizations (look up current ones), say why each fits, and point out patterns I could search for more of.",
  },
  {
    label: "Clarify my goals & strategy",
    blurb: "What you want next, and how to go after it",
    text: "Help me clarify my goals and strategy. Ask me a few questions one at a time about what I want next, then sum up what you hear, where my search could be more focused, and a strategy that fits.",
  },
  {
    label: "Resume feedback",
    blurb: "What's strong, what to sharpen",
    text: "Give me feedback on my current resume for the roles I'm targeting: what's strong, what's missing or unclear, and specific rewrites for the bullets that would gain the most.",
  },
  {
    label: "Cover letter feedback",
    blurb: "Tone, structure and what to cut",
    text: "Give me feedback on my current cover letter: what works, what to tighten or cut, and how to make the opening and closing stronger. Suggest specific rewrites.",
  },
  {
    label: "Networking outreach",
    blurb: "A short note to a contact or a stranger",
    text: "Help me write a short networking message to someone at an organization I'm interested in. Ask me who it's to and what I want from it, then draft a few versions with different tones.",
  },
  {
    label: "Plan my week",
    blurb: "A realistic plan for the next few days",
    text: "Help me plan this week of my job search. Look at the jobs I'm tracking and where each stands, then suggest a realistic plan: what to apply to, follow up on, prepare for, and how to split my time.",
  },
];

// Applications still in play come first in the list; closed-out ones last.
const CLOSED = new Set(["rejected_no_interview", "rejected_interviewed", "withdrawn"]);

const timeAgo = (iso: string) => {
  const then = new Date(iso.replace(" ", "T") + (iso.endsWith("Z") ? "" : "Z")).getTime();
  const minutes = Math.max(0, Math.round((Date.now() - then) / 60_000));
  if (minutes < 60) return minutes <= 1 ? "just now" : `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
};

// A general chat from before titles existed is titled on its next message; until then, "General chat".
const chatTitle = (c: JobbyChatSummary) => c.title || c.job_title || "General chat";

/** The left tray: every chat, newest first, to reopen or delete (asked 2026-11-10). */
function ChatHistory({
  chats,
  activeId,
  activeJobId,
  disabled,
  onOpen,
  onNew,
  onDelete,
}: {
  chats: JobbyChatSummary[] | null;
  activeId: number | null;
  activeJobId: number | null;
  disabled: boolean;
  onOpen: (c: JobbyChatSummary) => void;
  onNew: () => void;
  onDelete: (id: number) => void;
}) {
  const [confirm, setConfirm] = useState<number | null>(null);
  return (
    <div className="card jobby-history">
      <div className="jobby-history-head">
        <span className="flabel">Your chats</span>
        <button className="ghost small" onClick={onNew} disabled={disabled}>
          + New chat
        </button>
      </div>
      {!chats && <Spinner label="Loading..." />}
      {chats && chats.length === 0 && <p className="muted small">Your chats with Jobby will show up here.</p>}
      <ul>
        {chats?.map((c) => {
          const on = c.id === activeId || (c.job_id !== null && c.job_id === activeJobId);
          return (
            <li key={c.id} className={`jobby-history-item${on ? " on" : ""}`}>
              <button className="jobby-history-open" onClick={() => onOpen(c)} disabled={disabled}>
                <span className="jobby-history-title">{chatTitle(c)}</span>
                <span className="muted small jobby-history-meta">
                  {c.job_id !== null ? (
                    <>
                      {c.job_company}
                      {c.application_status && (
                        <span className={`pill pill-${c.application_status}`}>
                          {APPLICATION_STATUS_LABELS[c.application_status]}
                        </span>
                      )}
                    </>
                  ) : (
                    "General"
                  )}{" "}
                  · {timeAgo(c.updated_at)}
                </span>
              </button>
              {confirm === c.id ? (
                <span className="jobby-history-confirm small">
                  Delete?{" "}
                  <button className="link small danger-link" onClick={() => onDelete(c.id)}>
                    Yes
                  </button>{" "}
                  <button className="link small" onClick={() => setConfirm(null)}>
                    No
                  </button>
                </span>
              ) : (
                <button
                  className="jobby-history-delete"
                  onClick={() => setConfirm(c.id)}
                  disabled={disabled}
                  title="Delete this chat"
                  aria-label={`Delete "${chatTitle(c)}"`}
                >
                  ×
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** "Choose from My Jobs": pick the job to chat about. */
function JobChooser({ apps, onPick }: { apps: Application[] | null; onPick: (jobId: number) => void }) {
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const shown = (apps ?? [])
    .filter((a) => !q || `${a.title} ${a.company}`.toLowerCase().includes(q))
    .sort((a, b) => Number(CLOSED.has(a.status)) - Number(CLOSED.has(b.status)));
  return (
    <div className="jobby-chooser">
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search My Jobs..."
        autoFocus
      />
      {!apps && <Spinner label="Loading My Jobs..." />}
      {apps && apps.length === 0 && (
        <p className="muted">
          Jobs you track in <Link to="/tracker">My Jobs</Link> show up here.
        </p>
      )}
      {apps && apps.length > 0 && shown.length === 0 && <p className="muted">No jobs match "{query}".</p>}
      <ul>
        {shown.map((a) => (
          <li key={a.job_id}>
            <button className="jobby-pick" onClick={() => onPick(a.job_id)}>
              <span className="jobby-pick-title">{a.title}</span>
              <span className="muted small">
                {a.company} ·{" "}
                <span className={`pill pill-${a.status}`}>{APPLICATION_STATUS_LABELS[a.status]}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** A user message, with "> " lines (a passage quoted with "Ask about this") set off as a quote. */
function UserText({ text }: { text: string }) {
  const blocks: { quote: boolean; lines: string[] }[] = [];
  for (const line of text.split("\n")) {
    const quote = line.startsWith("> ") || line === ">";
    const last = blocks[blocks.length - 1];
    if (last && last.quote === quote) last.lines.push(quote ? line.replace(/^> ?/, "") : line);
    else blocks.push({ quote, lines: [quote ? line.replace(/^> ?/, "") : line] });
  }
  return (
    <>
      {blocks.map((b, i) =>
        b.quote ? (
          <span key={i} className="jobby-user-quote">
            {b.lines.join("\n")}
          </span>
        ) : (
          <span key={i}>{b.lines.join("\n").replace(/^\n+/, "")}</span>
        ),
      )}
    </>
  );
}

function Bubble({ message }: { message: Pick<JobbyMessage, "role" | "content"> }) {
  if (message.role === "user") {
    return (
      <div className="jobby-msg user">
        <UserText text={message.content} />
      </div>
    );
  }
  return (
    <div className="jobby-msg assistant">
      <JobbyLogo size={28} />
      <div className="jobby-msg-body">
        <Markdown text={message.content} />
      </div>
    </div>
  );
}

function Starters({ starters, onPick }: { starters: Starter[]; onPick: (text: string) => void }) {
  return (
    <div className="jobby-starters">
      {starters.map((s) => (
        <button key={s.label} className="jobby-starter" onClick={() => onPick(s.text)}>
          <span className="jobby-starter-label">{s.label}</span>
          <span className="muted small">{s.blurb}</span>
        </button>
      ))}
    </div>
  );
}

export default function JobbyPage() {
  const [params, setParams] = useSearchParams();
  const jobParam = params.get("job");
  const jobId = jobParam && /^\d+$/.test(jobParam) ? Number(jobParam) : null;
  const chatParam = params.get("chat");
  const chatId = chatParam && /^\d+$/.test(chatParam) ? Number(chatParam) : null;
  const isNew = params.has("new");
  const picking = params.has("pick");
  const inChat = jobId !== null || chatId !== null || isNew;
  const home = !inChat && !picking;

  const [chats, setChats] = useState<JobbyChatSummary[] | null>(null);
  // A new chat that just got its id: the URL switches to ?chat=<id> without reloading it.
  const justCreated = useRef<number | null>(null);
  const [apps, setApps] = useState<Application[] | null>(null);
  const [thread, setThread] = useState<JobbyThread | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<{ user: string; reply: string; status: string | null } | null>(
    null,
  );
  const [sendError, setSendError] = useState<string | null>(null);
  const chatRef = useRef<HTMLElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Whether a streaming reply pulls the view down with it. It stops the moment the reader scrolls
  // up by any means (wheel, trackpad, touch, keys, the scrollbar), and resumes only when they come
  // back to the very bottom themselves, send, or click "Jump to latest". Deciding it from the
  // scroll position alone failed live (asked twice, 2026-11-10): a smooth scroll moves a few pixels
  // a frame, the next chunk arrived while it was still "near the bottom", and snapped it back.
  const following = useRef(true);
  // Set while we move the scroll position ourselves, so that scroll event isn't read as the reader's.
  const selfScroll = useRef(false);
  const [showJump, setShowJump] = useState(false);
  const [chatHeight, setChatHeight] = useState<number | undefined>(undefined);
  // "Ask about this": the selected part of a reply and where to put its button.
  const [quote, setQuote] = useState<{ text: string; top: number; left: number } | null>(null);

  const refreshChats = useCallback(() => {
    api
      .listJobbyChats()
      .then(setChats)
      .catch(() => setChats((c) => c ?? []));
  }, []);

  useEffect(() => {
    refreshChats();
    api
      .listApplications()
      .then(setApps)
      .catch(() => setApps([]));
  }, [refreshChats]);

  useEffect(() => {
    setLoadError(null);
    setSendError(null);
    if (!inChat) {
      setThread(null);
      return;
    }
    if (isNew) {
      setThread({ thread_id: null, job: null, messages: [] });
      return;
    }
    if (chatId !== null && justCreated.current === chatId) {
      justCreated.current = null;
      return;
    }
    setThread(null);
    (chatId !== null ? api.getJobbyChat(chatId) : api.getJobbyThread(jobId))
      .then(setThread)
      .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)));
  }, [inChat, isNew, chatId, jobId]);

  const openChat = (c: JobbyChatSummary) =>
    setParams(c.job_id !== null ? { job: String(c.job_id) } : { chat: String(c.id) });
  const deleteChat = async (id: number) => {
    await api.deleteJobbyChat(id);
    setChats((cs) => (cs ? cs.filter((c) => c.id !== id) : cs));
    // Deleting the chat that's open goes back to Jobby's home.
    if (id === thread?.thread_id) setParams({});
  };

  // The panel fills the window below its top edge, so the message box is always in view at the
  // bottom and only the conversation scrolls (asked 2026-11-10: it could end up below the fold).
  useLayoutEffect(() => {
    const fit = () => {
      const el = chatRef.current;
      if (!el) return;
      const top = el.getBoundingClientRect().top + window.scrollY;
      setChatHeight(Math.max(window.innerHeight - top - 20, 420));
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);

  const scrollToEnd = useCallback(() => {
    const el = messagesRef.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight > 1) {
      selfScroll.current = true;
      el.scrollTop = el.scrollHeight;
    }
    following.current = true;
    setShowJump(false);
  }, []);

  // The reader taking over: any upward scroll gesture stops the following at once, before the
  // next chunk can land.
  useEffect(() => {
    const el = messagesRef.current;
    if (!el) return;
    const stop = () => {
      following.current = false;
    };
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) stop();
    };
    const onKey = (e: KeyboardEvent) => {
      if (["ArrowUp", "PageUp", "Home"].includes(e.key)) stop();
    };
    // A drag on the scrollbar lands on the box itself, not on a message.
    const onPointer = (e: PointerEvent) => {
      if (e.target === el) stop();
    };
    el.addEventListener("wheel", onWheel, { passive: true });
    el.addEventListener("touchmove", stop, { passive: true });
    el.addEventListener("keydown", onKey);
    el.addEventListener("pointerdown", onPointer);
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchmove", stop);
      el.removeEventListener("keydown", onKey);
      el.removeEventListener("pointerdown", onPointer);
    };
  }, [inChat, thread !== null]);

  // Selecting text in one of Jobby's replies offers "Ask about this" next to it (asked 2026-11-10).
  const placeQuote = useCallback(() => {
    const sel = window.getSelection();
    const box = messagesRef.current;
    const chat = chatRef.current;
    const text = sel?.toString().trim() ?? "";
    if (!sel || !box || !chat || !text || sel.rangeCount === 0) {
      setQuote(null);
      return;
    }
    const node = sel.anchorNode instanceof Element ? sel.anchorNode : sel.anchorNode?.parentElement;
    if (!node?.closest(".jobby-msg.assistant") || !box.contains(node)) {
      setQuote(null);
      return;
    }
    const rect = sel.getRangeAt(0).getBoundingClientRect();
    const frame = chat.getBoundingClientRect();
    const boxFrame = box.getBoundingClientRect();
    if (rect.bottom < boxFrame.top || rect.top > boxFrame.bottom) {
      setQuote(null); // scrolled out of view
      return;
    }
    setQuote({
      text,
      top: rect.bottom - frame.top + 6,
      left: Math.min(Math.max(rect.left - frame.left, 12), frame.width - 160),
    });
  }, []);
  useEffect(() => {
    document.addEventListener("selectionchange", placeQuote);
    return () => document.removeEventListener("selectionchange", placeQuote);
  }, [placeQuote]);

  const onMessagesScroll = () => {
    const el = messagesRef.current;
    if (!el) return;
    placeQuote(); // the selection moved with the text: move its button too
    if (selfScroll.current) {
      selfScroll.current = false;
      return;
    }
    // The reader's own scroll: back at the very bottom, follow again; anywhere else, stay put.
    const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < 4;
    following.current = atEnd;
    if (atEnd) setShowJump(false);
  };

  // New text pulls the view down only while following; otherwise the reader stays where they are,
  // with "Jump to latest". Only the conversation box scrolls, never the page.
  useEffect(() => {
    if (following.current) scrollToEnd();
    else setShowJump(true);
  }, [thread?.messages.length, pending?.reply, pending?.status, scrollToEnd]);

  const askAbout = () => {
    if (!quote) return;
    const quoted = quote.text
      .split("\n")
      .map((line) => `> ${line}`)
      .join("\n");
    setDraft((d) => `${quoted}\n\n${d}`.trimStart());
    setQuote(null);
    window.getSelection()?.removeAllRanges();
    requestAnimationFrame(() => {
      const box = inputRef.current;
      if (!box) return;
      box.focus();
      box.setSelectionRange(box.value.length, box.value.length);
    });
  };

  const send = async (text: string) => {
    const content = text.trim();
    if (!content || pending) return;
    setDraft("");
    setQuote(null);
    following.current = true; // sending always brings the conversation back to the newest message
    setSendError(null);
    setPending({ user: content, reply: "", status: null });
    const target: JobbyTarget =
      thread?.thread_id != null ? { threadId: thread.thread_id } : jobId !== null ? { jobId } : { newChat: true };
    try {
      await api.streamJobbyMessage(target, content, (event) => {
        if (event.type === "delta") {
          setPending((p) => (p ? { ...p, reply: p.reply + event.text, status: null } : p));
        } else if (event.type === "status") {
          setPending((p) => (p ? { ...p, status: event.text } : p));
        } else if (event.type === "done") {
          setThread((t) =>
            t ? { ...t, thread_id: event.thread_id, messages: [...t.messages, event.user, event.assistant] } : t,
          );
          setPending(null);
          refreshChats();
          if (isNew) {
            // The new chat exists now: give it its own address, so Back and reloads find it.
            justCreated.current = event.thread_id;
            setParams({ chat: String(event.thread_id) }, { replace: true });
          }
        } else if (event.type === "title") {
          setChats((cs) => (cs ? cs.map((c) => (c.id === event.thread_id ? { ...c, title: event.title } : c)) : cs));
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      });
    } catch (e) {
      // Nothing from a failed turn is saved, so put their message back to retry.
      setSendError(e instanceof Error ? e.message : String(e));
      setDraft(content);
    } finally {
      setPending(null);
    }
  };

  const messages = thread?.messages ?? [];
  const empty = messages.length === 0 && !pending;
  const job = thread?.job;
  const starters = job || jobId !== null ? JOB_STARTERS : GENERAL_STARTERS;
  const current = chats?.find((c) => c.id === thread?.thread_id);

  return (
    <div>
      <div className="jobby-header">
        <JobbyLogo size={56} />
        <div>
          <h1>Jobby</h1>
          <p className="muted" style={{ margin: 0 }}>
            Your job search sidekick. Jobby knows your profile, your documents, and the jobs you're interested in.
          </p>
        </div>
      </div>

      <div className="jobby-layout">
        <aside>
          <ChatHistory
            chats={chats}
            activeId={thread?.thread_id ?? null}
            activeJobId={jobId}
            disabled={!!pending}
            onOpen={openChat}
            onNew={() => setParams({})}
            onDelete={deleteChat}
          />
        </aside>

        <section className="card jobby-chat" ref={chatRef} style={{ height: chatHeight }}>
          {home && (
            <div className="jobby-messages jobby-home">
              <div className="jobby-welcome">
                <JobbyLogo variant="full" size={180} title="Jobby at a laptop" />
                <Bubble
                  message={{
                    role: "assistant",
                    content:
                      "Hi! I'm **Jobby**, your job search sidekick. Want to talk through your search in general, or get ready for one of the jobs on your My Jobs list?",
                  }}
                />
                {/* Plainly buttons, with a line saying to pick one (asked 2026-11-10: a first-time
                    user should see these are how to get started). */}
                <p className="jobby-home-prompt">Choose one to get started:</p>
                <div className="jobby-home-actions">
                  <button className="jobby-home-action" onClick={() => setParams({ new: "1" })}>
                    <span className="jobby-home-icon" aria-hidden="true">💬</span>
                    <span className="jobby-home-text">
                      <span className="jobby-home-label">Start a general chat</span>
                      <span className="muted small">
                        Finding organizations, your goals and strategy, resume and cover letter feedback
                      </span>
                    </span>
                    <span className="jobby-home-go">Start →</span>
                  </button>
                  <button className="jobby-home-action" onClick={() => setParams({ pick: "1" })}>
                    <span className="jobby-home-icon" aria-hidden="true">💼</span>
                    <span className="jobby-home-text">
                      <span className="jobby-home-label">Choose from My Jobs</span>
                      <span className="muted small">Interview prep and research for one job</span>
                    </span>
                    <span className="jobby-home-go">Pick a job →</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {picking && (
            <>
              <div className="jobby-chat-head">
                <button className="ghost small jobby-back" onClick={() => setParams({})}>
                  ← Back
                </button>
                <div className="jobby-chat-title">
                  <div className="card-title">Which job are you prepping for?</div>
                  <div className="muted small">From My Jobs; a job you've chatted about picks up where you left off.</div>
                </div>
              </div>
              <div className="jobby-messages">
                <JobChooser apps={apps} onPick={(id) => setParams({ job: String(id) })} />
              </div>
            </>
          )}

          {inChat && (
            <>
              <div className="jobby-chat-head">
                <button
                  className="ghost small jobby-back"
                  onClick={() => setParams({})}
                  disabled={!!pending}
                  title="Back to Jobby's home. This chat is saved."
                >
                  ← Back
                </button>
                <div className="jobby-chat-title">
                  {job ? (
                    <>
                      <div className="card-title">
                        <Link to={`/jobs/${job.id}`}>{job.title}</Link>
                      </div>
                      <div className="muted small">{job.company}</div>
                    </>
                  ) : (
                    <>
                      <div className="card-title">{current ? chatTitle(current) : "New chat"}</div>
                      <div className="muted small">General chat</div>
                    </>
                  )}
                </div>
              </div>

              <div className="jobby-messages" ref={messagesRef} onScroll={onMessagesScroll} tabIndex={-1}>
                {loadError && <p className="error-text small">{loadError}</p>}
                {!thread && !loadError && <Spinner label="Loading..." />}
                {thread && empty && (
                  <div className="jobby-welcome">
                    <JobbyLogo variant="full" size={180} title="Jobby at a laptop" />
                    <Bubble
                      message={{
                        role: "assistant",
                        content: job
                          ? `Hi! I'm **Jobby**. Let's get you ready for **${job.title}** at **${job.company}**. Pick a place to start, or ask me anything.`
                          : "Hi! I'm **Jobby**. What would help most with your search right now? Pick a place to start, or ask me anything.",
                      }}
                    />
                    <Starters starters={starters} onPick={send} />
                  </div>
                )}
                {messages.map((m) => (
                  <Bubble key={m.id} message={m} />
                ))}
                {pending && (
                  <>
                    <Bubble message={{ role: "user", content: pending.user }} />
                    <div className="jobby-msg assistant">
                      <JobbyLogo size={28} />
                      <div className="jobby-msg-body">
                        {pending.reply && <Markdown text={pending.reply} />}
                        {(pending.status || !pending.reply) && (
                          <Spinner label={pending.status ?? "Jobby is thinking..."} />
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>
              {showJump && (
                <button className="jobby-jump" onClick={scrollToEnd}>
                  Jump to latest ↓
                </button>
              )}
              {quote && (
                <button
                  className="jobby-quote-btn"
                  style={{ top: quote.top, left: quote.left }}
                  // Keep the selection: a mousedown would clear it before the click lands.
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={askAbout}
                >
                  Ask about this
                </button>
              )}

              {!empty && !pending && (
                <div className="jobby-try">
                  <span className="muted small">Try:</span>
                  {starters.map((s) => (
                    <button key={s.label} className="chip-toggle" onClick={() => send(s.text)}>
                      {s.label}
                    </button>
                  ))}
                </div>
              )}

              {sendError && <p className="error-text small">{sendError}</p>}
              <div className="jobby-input">
                <textarea
                  ref={inputRef}
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      send(draft);
                    }
                  }}
                  rows={2}
                  placeholder={pending ? "Jobby is replying..." : "Ask Jobby anything — Enter to send, Shift+Enter for a new line"}
                  disabled={!!pending || !thread}
                />
                <button className="primary" onClick={() => send(draft)} disabled={!!pending || !draft.trim()}>
                  Send
                </button>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
