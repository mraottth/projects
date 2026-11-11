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

// Jobby, the job-search sidekick. Where you are lives in the URL: /jobby is Jobby's home, with your
// past chats (asked 2026-11-10); ?job=42 is that job's chat (My Jobs and the job page link
// straight there); ?chat=7 reopens any chat; ?new=1 is a new general chat. Chats are saved
// server-side (core/jobby.py), so they survive reloads, and are only deleted from the home list.

const STARTERS: { label: string; blurb: string; text: string }[] = [
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

// Applications still in play come first in the picker; closed-out ones last.
const CLOSED = new Set(["rejected_no_interview", "rejected_interviewed", "withdrawn"]);

function JobPicker({
  apps,
  selected,
  onSelect,
  disabled,
}: {
  apps: Application[];
  selected: number | null;
  onSelect: (jobId: number | null) => void;
  disabled: boolean;
}) {
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const shown = apps
    .filter((a) => !q || `${a.title} ${a.company}`.toLowerCase().includes(q))
    .sort((a, b) => Number(CLOSED.has(a.status)) - Number(CLOSED.has(b.status)));

  return (
    <div className="card jobby-picker">
      <span className="flabel">Which job are you prepping for?</span>
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search My Jobs..."
        style={{ width: "100%", marginBottom: "8px" }}
      />
      <ul>
        <li>
          <button
            className={`jobby-pick${selected === null ? " on" : ""}`}
            onClick={() => onSelect(null)}
            disabled={disabled}
          >
            <span className="jobby-pick-title">No specific job</span>
            <span className="muted small">Start a new general chat</span>
          </button>
        </li>
        {shown.map((a) => (
          <li key={a.job_id}>
            <button
              className={`jobby-pick${selected === a.job_id ? " on" : ""}`}
              onClick={() => onSelect(a.job_id)}
              disabled={disabled}
            >
              <span className="jobby-pick-title">{a.title}</span>
              <span className="muted small">
                {a.company} ·{" "}
                <span className={`pill pill-${a.status}`}>{APPLICATION_STATUS_LABELS[a.status]}</span>
              </span>
            </button>
          </li>
        ))}
        {apps.length === 0 && (
          <li className="muted small" style={{ padding: "6px 4px" }}>
            Jobs you track in <Link to="/tracker">My Jobs</Link> show up here.
          </li>
        )}
        {apps.length > 0 && shown.length === 0 && (
          <li className="muted small" style={{ padding: "6px 4px" }}>
            No jobs match "{query}".
          </li>
        )}
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

export default function JobbyPage() {
  const [params, setParams] = useSearchParams();
  const jobParam = params.get("job");
  const jobId = jobParam && /^\d+$/.test(jobParam) ? Number(jobParam) : null;
  const chatParam = params.get("chat");
  const chatId = chatParam && /^\d+$/.test(chatParam) ? Number(chatParam) : null;
  const isNew = params.has("new");
  const home = jobId === null && chatId === null && !isNew;
  const [chats, setChats] = useState<JobbyChatSummary[] | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  // A new chat that just got its id: the URL switches to ?chat=<id> without reloading it.
  const justCreated = useRef<number | null>(null);

  const [apps, setApps] = useState<Application[]>([]);
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
  // Whether the reader is at the bottom of the conversation: only then does a streaming reply
  // pull the view along (asked 2026-11-10: following every chunk fought anyone scrolling up to read).
  const atBottom = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const [chatHeight, setChatHeight] = useState<number | undefined>(undefined);
  // "Ask about this": the selected part of a reply and where to put its button.
  const [quote, setQuote] = useState<{ text: string; top: number; left: number } | null>(null);

  useEffect(() => {
    api.listApplications().then(setApps).catch(() => undefined);
  }, []);

  useEffect(() => {
    setLoadError(null);
    setSendError(null);
    if (home) {
      setThread(null);
      setConfirmDelete(null);
      api
        .listJobbyChats()
        .then(setChats)
        .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)));
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
  }, [home, isNew, chatId, jobId]);

  const openChat = (c: JobbyChatSummary) => setParams(c.job_id !== null ? { job: String(c.job_id) } : { chat: String(c.id) });
  const deleteChat = async (id: number) => {
    await api.deleteJobbyChat(id);
    setChats((cs) => (cs ? cs.filter((c) => c.id !== id) : cs));
    setConfirmDelete(null);
  };

  // The chat fills the window below its top edge, so the message box is always in view at the
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
    if (el) el.scrollTop = el.scrollHeight;
    atBottom.current = true;
    setShowJump(false);
  }, []);

  const onMessagesScroll = () => {
    const el = messagesRef.current;
    if (!el) return;
    atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    if (atBottom.current) setShowJump(false);
    placeQuote(); // the selection moved with the text: move its button too
  };

  // New text follows only a reader who's at the bottom; anyone who scrolled up stays put, with a
  // "Jump to latest" button. Only the conversation box scrolls, never the page (scrollIntoView
  // scrolled the whole page too, which made it jump).
  useEffect(() => {
    if (atBottom.current) scrollToEnd();
    else setShowJump(true);
  }, [thread?.messages.length, pending?.reply, pending?.status, scrollToEnd]);

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

  const selectJob = (id: number | null) => setParams(id === null ? { new: "1" } : { job: String(id) });

  const send = async (text: string) => {
    const content = text.trim();
    if (!content || pending) return;
    setDraft("");
    setQuote(null);
    atBottom.current = true; // sending always brings the conversation back to the newest message
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
          if (isNew) {
            // The new chat exists now: give it its own address, so Back and reloads find it.
            justCreated.current = event.thread_id;
            setParams({ chat: String(event.thread_id) }, { replace: true });
          }
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
          <JobPicker apps={apps} selected={home ? -1 : jobId} onSelect={selectJob} disabled={!!pending} />
        </aside>

        <section className="card jobby-chat" ref={chatRef} style={{ height: chatHeight }}>
          {home ? (
            <ChatList
              chats={chats}
              error={loadError}
              confirmDelete={confirmDelete}
              onOpen={openChat}
              onNew={() => setParams({ new: "1" })}
              onAskDelete={setConfirmDelete}
              onDelete={deleteChat}
            />
          ) : (
          <>
          <div className="jobby-chat-head">
            <button
              className="ghost small jobby-back"
              onClick={() => setParams({})}
              disabled={!!pending}
              title="Back to your chats. This one is saved."
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
                  <div className="card-title">General interview help</div>
                  <div className="muted small">Pick a job on the left for prep tailored to it.</div>
                </>
              )}
            </div>
          </div>

          <div className="jobby-messages" ref={messagesRef} onScroll={onMessagesScroll}>
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
                      : "Hi! I'm **Jobby**, your interview-prep sidekick. Pick a job on the left for prep tailored to it, or start with something general.",
                  }}
                />
                <div className="jobby-starters">
                  {STARTERS.map((s) => (
                    <button key={s.label} className="jobby-starter" onClick={() => send(s.text)}>
                      <span className="jobby-starter-label">{s.label}</span>
                      <span className="muted small">{s.blurb}</span>
                    </button>
                  ))}
                </div>
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
              {STARTERS.map((s) => (
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


const timeAgo = (iso: string) => {
  const then = new Date(iso.replace(" ", "T") + (iso.endsWith("Z") ? "" : "Z")).getTime();
  const minutes = Math.max(0, Math.round((Date.now() - then) / 60_000));
  if (minutes < 60) return minutes <= 1 ? "just now" : `${minutes} minutes ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
};

/** Jobby's home: past chats to reopen or delete, and a new general chat (asked 2026-11-10). */
function ChatList({
  chats,
  error,
  confirmDelete,
  onOpen,
  onNew,
  onAskDelete,
  onDelete,
}: {
  chats: JobbyChatSummary[] | null;
  error: string | null;
  confirmDelete: number | null;
  onOpen: (c: JobbyChatSummary) => void;
  onNew: () => void;
  onAskDelete: (id: number | null) => void;
  onDelete: (id: number) => void;
}) {
  return (
    <>
      <div className="jobby-chat-head">
        <div className="jobby-chat-title">
          <div className="card-title">Your chats</div>
          <div className="muted small">Pick one up where you left off, or pick a job on the left.</div>
        </div>
        <button className="primary" onClick={onNew}>
          New general chat
        </button>
      </div>
      <div className="jobby-messages">
        {error && <p className="error-text small">{error}</p>}
        {!chats && !error && <Spinner label="Loading your chats..." />}
        {chats && chats.length === 0 && (
          <div className="jobby-welcome">
            <JobbyLogo variant="full" size={180} title="Jobby at a laptop" />
            <p className="muted" style={{ textAlign: "center" }}>
              No chats yet. Pick a job on the left to prep for it, or start a general chat.
            </p>
          </div>
        )}
        {chats && chats.length > 0 && (
          <ul className="jobby-chat-list">
            {chats.map((c) => (
              <li key={c.id} className="jobby-chat-item">
                <button className="jobby-chat-open" onClick={() => onOpen(c)}>
                  <span className="jobby-chat-item-title">
                    {c.job_title ? `${c.job_title} — ${c.job_company}` : "General chat"}
                    {c.application_status && (
                      <span className={`pill pill-${c.application_status}`}>
                        {APPLICATION_STATUS_LABELS[c.application_status]}
                      </span>
                    )}
                  </span>
                  <span className="jobby-chat-preview">{c.preview}</span>
                  <span className="muted small">
                    {c.message_count} message{c.message_count === 1 ? "" : "s"} · {timeAgo(c.updated_at)}
                  </span>
                </button>
                {confirmDelete === c.id ? (
                  <span className="jobby-chat-confirm small">
                    Delete this chat?{" "}
                    <button className="link small danger-link" onClick={() => onDelete(c.id)}>
                      Delete
                    </button>{" "}
                    <button className="link small" onClick={() => onAskDelete(null)}>
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button className="link small danger-link" onClick={() => onAskDelete(c.id)} title="Delete this chat">
                    Delete
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </>
  );
}

